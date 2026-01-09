#!/usr/bin/env python3
"""
Augment SFT training data with scratchpad and relevant_policies fields.

Adds two simple string fields to each answer object:
- scratchpad: Agent's notepad to track situation, progress, and next steps across turns
- relevant_policies: Key policies that apply to this situation

Usage:
    python augment_sft_simple.py input.jsonl output.jsonl [--model MODEL] [--dry-run]
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

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
            result = content[:300] + "..." if len(content) > 300 else content
            context_parts.append(f"TOOL RESULT: {result}")
    
    return "\n".join(context_parts)


def build_prompt(entry: Dict[str, Any]) -> str:
    """Build prompt for generating scratchpad and relevant_policies."""
    
    conversations = entry.get("conversations", [])
    conversation_context = extract_conversation_context(conversations)
    
    prompt = f"""Given this customer service conversation, generate two brief text fields as if you are a human agent taking notes.

## Conversation
{conversation_context}

## Policy (for reference)
{POLICY_TEXT[:6000]}

---

Generate a JSON with exactly two string fields:

```json
{{
    "scratchpad": "<Your personal notepad as a customer service agent. Write notes like a human would: track what's been done, what's pending, customer details gathered, current status of requests, and what you need to do next. Use shorthand, bullet points, or any format that helps you remember the situation in future turns.>",
    "relevant_policies": "<1-3 sentences listing the key policy rules that apply. Focus on rules the agent must follow for this specific situation.>"
}}
```

Be concise and specific. Output only the JSON."""

    return prompt


def generate_fields(
    entry: Dict[str, Any],
    model: str = "gpt-4o-mini",
    provider: str = "openai"
) -> Dict[str, str]:
    """Generate scratchpad and relevant_policies for a single entry."""
    
    prompt = build_prompt(entry)
    
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
        
        result = json.loads(response_text.strip())
        return {
            "scratchpad": result.get("scratchpad", ""),
            "relevant_policies": result.get("relevant_policies", "")
        }
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return {
            "scratchpad": "",
            "relevant_policies": ""
        }


def get_conversation_preview(conversations: List[Dict[str, Any]], max_len: int = 60) -> str:
    """Get a preview of the last user message."""
    for turn in reversed(conversations):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            return content[:max_len] + "..." if len(content) > max_len else content
    return "No user message"


def load_existing_output(output_path: str) -> tuple[List[Dict[str, Any]], Dict[Any, List[Dict[str, str]]]]:
    """Load existing output file and rebuild task_history from it."""
    existing_entries = []
    task_history: Dict[Any, List[Dict[str, str]]] = {}
    
    if not os.path.exists(output_path):
        return existing_entries, task_history
    
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                existing_entries.append(entry)
                
                # Rebuild task_history from the answer fields
                task_id = entry.get("task_id")
                if entry.get("answer") and len(entry["answer"]) > 0:
                    ans = entry["answer"][0]
                    fields = {
                        "reasoning_content": ans.get("reasoning_content", ""),
                        "scratchpad": ans.get("scratchpad", ""),
                        "relevant_policies": ans.get("relevant_policies", "")
                    }
                    if task_id not in task_history:
                        task_history[task_id] = []
                    task_history[task_id].append(fields)
    
    return existing_entries, task_history


def prompt_resume_choice(existing_count: int, total_count: int) -> int:
    """Prompt user to choose whether to resume or restart."""
    print(f"\n{'='*60}")
    print(f"Found existing output file with {existing_count}/{total_count} entries processed.")
    print(f"{'='*60}")
    print(f"  [1] Continue from entry {existing_count + 1}")
    print(f"  [2] Start fresh (overwrite existing file)")
    print(f"  [3] Start from a specific entry number")
    print(f"  [0] Cancel")
    print()
    
    while True:
        choice = input("Enter your choice [1/2/3/0]: ").strip()
        if choice == "1":
            return existing_count  # Resume from where we left off
        elif choice == "2":
            return 0  # Start fresh
        elif choice == "3":
            while True:
                try:
                    start_from = int(input(f"Enter entry number to start from (1-{total_count}): ").strip())
                    if 1 <= start_from <= total_count:
                        return start_from - 1  # Convert to 0-indexed
                    print(f"Please enter a number between 1 and {total_count}")
                except ValueError:
                    print("Please enter a valid number")
        elif choice == "0":
            print("Cancelled.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")


def augment_jsonl(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    dry_run: bool = False,
    limit: int = None
):
    """Augment all entries in a JSONL file."""
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Model: {model} ({provider})")
    print()
    
    entries = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries")
    
    if limit:
        entries = entries[:limit]
        print(f"Limited to {limit} entries")
    
    # Check for existing output and prompt user
    start_from = 0
    task_history: Dict[Any, List[Dict[str, str]]] = {}
    
    if os.path.exists(output_path) and not dry_run:
        existing_entries, existing_history = load_existing_output(output_path)
        if existing_entries:
            start_from = prompt_resume_choice(len(existing_entries), len(entries))
            
            if start_from > 0:
                # Rebuild task_history from entries we're keeping
                task_history = {}
                for entry in existing_entries[:start_from]:
                    task_id = entry.get("task_id")
                    if entry.get("answer") and len(entry["answer"]) > 0:
                        ans = entry["answer"][0]
                        fields = {
                            "reasoning_content": ans.get("reasoning_content", ""),
                            "scratchpad": ans.get("scratchpad", ""),
                            "relevant_policies": ans.get("relevant_policies", "")
                        }
                        if task_id not in task_history:
                            task_history[task_id] = []
                        task_history[task_id].append(fields)
                
                print(f"\nResuming from entry {start_from + 1}, keeping {start_from} existing entries...")
    
    # Determine file mode and prepare existing content
    if start_from > 0:
        # Read existing entries to preserve
        existing_entries, _ = load_existing_output(output_path)
        entries_to_write = existing_entries[:start_from]
        file_mode = "w"  # Rewrite with preserved entries + new ones
    else:
        entries_to_write = []
        file_mode = "w"
    
    with open(output_path, file_mode) as out_f:
        # Write preserved entries first
        for entry in entries_to_write:
            out_f.write(json.dumps(entry) + "\n")
        out_f.flush()
        
        # Process remaining entries
        for i, entry in enumerate(entries):
            if i < start_from:
                continue  # Skip already processed entries
                
            task_id = entry.get("task_id", i)
            conv_len = len(entry.get("conversations", []))
            preview = get_conversation_preview(entry.get("conversations", []))
            
            print(f"[{i+1}/{len(entries)}] Task {task_id}, {conv_len} turns: {preview}")
            
            # Inject ALL previous turns' fields into conversation history if available
            if task_id in task_history:
                history = task_history[task_id]
                # Find all assistant turns in conversations and add the corresponding fields
                assistant_idx = 0
                for turn in entry.get("conversations", []):
                    if turn.get("role") == "assistant" and assistant_idx < len(history):
                        turn["reasoning_content"] = history[assistant_idx].get("reasoning_content", "")
                        turn["scratchpad"] = history[assistant_idx].get("scratchpad", "")
                        turn["relevant_policies"] = history[assistant_idx].get("relevant_policies", "")
                        assistant_idx += 1
            
            if dry_run:
                prompt = build_prompt(entry)
                print(f"  Prompt length: {len(prompt)} chars")
            else:
                # Generate the two fields
                fields = generate_fields(entry, model, provider)
                
                # Add to answer object
                if entry.get("answer") and len(entry["answer"]) > 0:
                    entry["answer"][0]["scratchpad"] = fields["scratchpad"]
                    entry["answer"][0]["relevant_policies"] = fields["relevant_policies"]
                    
                    # Copy chosen_answer_raw into answer[0] as reasoning_content
                    if "chosen_answer_raw" in entry:
                        entry["answer"][0]["reasoning_content"] = entry["chosen_answer_raw"]
                
                # Append current turn's fields to history for this task
                current_fields = {
                    "reasoning_content": entry.get("chosen_answer_raw", ""),
                    "scratchpad": fields["scratchpad"],
                    "relevant_policies": fields["relevant_policies"]
                }
                if task_id not in task_history:
                    task_history[task_id] = []
                task_history[task_id].append(current_fields)
                
                # Write immediately
                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                
                print(f"  ✓ scratchpad: {fields['scratchpad'][:50]}...")
    
    if not dry_run:
        print(f"\n✓ Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment SFT data with scratchpad and relevant_policies")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("--model", default="gpt-5.1", help="LLM model (default: gpt-5.1)")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without API calls")
    parser.add_argument("--limit", type=int, help="Limit number of entries")
    
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
        limit=args.limit
    )


if __name__ == "__main__":
    main()
