#!/usr/bin/env python3
"""
Augment SFT training data with context_recall field.

This script reads a JSONL file with SFT training data (turn-by-turn conversations)
and augments each entry with a "context_recall" field that includes:
1. A structured recap of all customer requests identified so far
2. What has been resolved vs. what's pending
3. Relevant policy excerpts for those requests

The context_recall is generated based on the conversation history up to that point.

Usage:
    python augment_sft_context_recall.py input.jsonl output.jsonl [--model MODEL] [--dry-run]
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from litellm import completion

# Load retail policy
POLICY_PATH = Path(__file__).parent.parent / "data" / "envs" / "retail" / "policy.md"


def load_policy() -> str:
    """Load the retail policy document."""
    with open(POLICY_PATH, "r") as f:
        return f.read()


POLICY_TEXT = load_policy()


def extract_conversation_context(conversations: List[Dict[str, Any]]) -> str:
    """Extract readable conversation context from the conversations list."""
    context_parts = []
    
    for turn in conversations:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls")
        
        if role == "system":
            # Skip system prompt in context (too long)
            continue
        elif role == "user":
            context_parts.append(f"CUSTOMER: {content}")
        elif role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    context_parts.append(f"AGENT ACTION: Called {tool_name}({json.dumps(args)})")
            elif content:
                context_parts.append(f"AGENT: {content}")
        elif role == "tool":
            # Truncate long tool results
            result = content[:500] + "..." if len(content) > 500 else content
            context_parts.append(f"TOOL RESULT: {result}")
    
    return "\n".join(context_parts)


def build_context_recall_prompt(entry: Dict[str, Any]) -> str:
    """Build prompt for generating context_recall from SFT entry."""
    
    conversations = entry.get("conversations", [])
    conversation_context = extract_conversation_context(conversations)
    
    # Get the current answer's reasoning for reference
    answer = entry.get("answer", [{}])[0] if entry.get("answer") else {}
    current_reasoning = answer.get("reasoning_content", "")
    
    # Count turns for context
    non_system = [t for t in conversations if t.get("role") != "system"]
    user_turns = len([t for t in non_system if t.get("role") == "user"])
    
    prompt = f"""You are helping create training data for a customer service AI agent.

Given the conversation history so far, generate a "context_recall" that summarizes:
1. All customer requests identified (even if not explicitly stated, infer from context)
2. Current status of each request (pending, in_progress, completed, blocked)
3. What information has been gathered vs. what's still needed
4. Relevant policy rules that apply

## Conversation History ({user_turns} customer message(s) so far)
{conversation_context}

## Current Agent Reasoning (for reference)
{current_reasoning}

## Policy Document (extract ONLY relevant sections)
{POLICY_TEXT[:8000]}

---

Generate a JSON object with this structure:
```json
{{
    "conversation_stage": "<early|authentication|information_gathering|action_preparation|confirmation|execution|post_action>",
    "request_summary": {{
        "total_requests": <number>,
        "requests": [
            {{
                "request_id": 1,
                "description": "<concise description>",
                "type": "<exchange|return|modify_address|modify_items|cancel|info_lookup|update_profile|authentication>",
                "order_id": "<if known, else null>",
                "status": "<identified|gathering_info|ready_for_action|awaiting_confirmation|completed|blocked>",
                "blocking_reason": "<if blocked, why>",
                "info_gathered": {{
                    "<field>": "<value or 'unknown'>"
                }},
                "info_needed": ["<what's still needed>"]
            }}
        ]
    }},
    "authentication_status": {{
        "authenticated": <true|false>,
        "user_id": "<if known>",
        "method_used": "<email|name_zip|none>",
        "info_available": "<what customer provided>"
    }},
    "tool_results_summary": [
        {{
            "tool": "<tool_name>",
            "key_findings": "<brief summary of what was learned>"
        }}
    ],
    "relevant_policies": [
        {{
            "policy_name": "<short name>",
            "summary": "<1-2 sentence summary>",
            "applies_to": "<which request or general>"
        }}
    ],
    "next_logical_step": "<what the agent should do next>",
    "warnings": ["<any gotchas or things to watch out for>"]
}}
```

Important:
- Be specific about what info has been gathered from tool calls
- Track the exact stage of the conversation
- If this is early in conversation, authentication is likely the first need
- Include warnings about one-time modification rules, sequencing requirements, etc.

Output only the JSON, no additional text."""

    return prompt


def generate_context_recall(
    entry: Dict[str, Any],
    model: str = "gpt-4o-mini",
    provider: str = "openai"
) -> Dict[str, Any]:
    """Generate context_recall for a single SFT entry using LLM."""
    
    prompt = build_context_recall_prompt(entry)
    
    try:
        response = completion(
            model=model,
            custom_llm_provider=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        
        return json.loads(response_text.strip())
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON decode error: {e}")
        return {"error": str(e), "raw_response": response_text[:1000] if 'response_text' in dir() else None}
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return {"error": str(e)}


def get_conversation_preview(conversations: List[Dict[str, Any]], max_len: int = 80) -> str:
    """Get a preview of the last user message."""
    for turn in reversed(conversations):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            return content[:max_len] + "..." if len(content) > max_len else content
    return "No user message"


def augment_jsonl(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    dry_run: bool = False,
    limit: int = None,
    skip_existing: bool = False
):
    """Augment all entries in a JSONL file with context_recall.
    
    Saves progress incrementally to avoid losing work on interruption.
    """
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Model: {model} ({provider})")
    print(f"Dry run: {dry_run}")
    print()
    
    entries = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries")
    
    # Check for existing progress
    existing_entries = []
    if os.path.exists(output_path) and skip_existing:
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    existing_entries.append(json.loads(line))
        print(f"Found {len(existing_entries)} existing entries in output")
    
    if limit:
        entries = entries[:limit]
        print(f"Limited to {limit} entries")
    
    # Open output file in append mode if resuming, write mode otherwise
    mode = "a" if existing_entries and skip_existing else "w"
    start_idx = len(existing_entries) if existing_entries and skip_existing else 0
    
    if start_idx > 0:
        print(f"Resuming from entry {start_idx + 1}")
        entries = entries[start_idx:]
    
    processed = start_idx
    
    with open(output_path, mode) as out_f:
        for i, entry in enumerate(entries):
            actual_idx = start_idx + i
            task_id = entry.get("task_id", actual_idx)
            conv_len = len(entry.get("conversations", []))
            preview = get_conversation_preview(entry.get("conversations", []))
            
            print(f"\n[{actual_idx+1}/{start_idx + len(entries)}] Task {task_id}, {conv_len} turns")
            print(f"  Last user msg: {preview}")
            
            if skip_existing and entry.get("answer") and entry["answer"][0].get("context_recall"):
                print(f"  ⏭ Skipping (already has context_recall)")
                if not dry_run:
                    out_f.write(json.dumps(entry) + "\n")
                    out_f.flush()
                processed += 1
                continue
            
            if dry_run:
                # In dry run, just show the prompt preview
                prompt = build_context_recall_prompt(entry)
                print(f"  Prompt length: {len(prompt)} chars")
            else:
                # Generate context_recall
                context_recall = generate_context_recall(entry, model, provider)
                
                # Add context_recall to the answer object (alongside reasoning_content)
                if entry.get("answer") and len(entry["answer"]) > 0:
                    entry["answer"][0]["context_recall"] = context_recall
                else:
                    # Fallback: create answer structure if missing
                    entry["answer"] = [{"context_recall": context_recall}]
                
                # Write immediately and flush
                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                processed += 1
                
                # Show preview
                if "error" not in context_recall:
                    stage = context_recall.get("conversation_stage", "?")
                    num_requests = context_recall.get("request_summary", {}).get("total_requests", "?")
                    auth = "✓" if context_recall.get("authentication_status", {}).get("authenticated") else "✗"
                    print(f"  ✓ Stage: {stage}, Requests: {num_requests}, Auth: {auth}")
                else:
                    print(f"  ✗ Error: {context_recall.get('error')}")
    
    if not dry_run:
        print(f"\n✓ Wrote {processed} augmented entries to {output_path}")
    else:
        print(f"\n[DRY RUN] Would write entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment SFT training data with context_recall")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without making API calls")
    parser.add_argument("--limit", type=int, help="Limit number of entries to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries that already have context_recall")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    augment_jsonl(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        provider=args.provider,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
