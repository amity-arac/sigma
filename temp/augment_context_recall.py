#!/usr/bin/env python3
"""
Augment GRPO training data with context_recall field.

This script reads a JSONL file with training data and augments each entry
with a "context_recall" field that includes:
1. A structured recap of all customer requests
2. Relevant policy excerpts for those requests

Usage:
    python augment_context_recall.py input.jsonl output.jsonl [--model MODEL] [--dry-run]
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

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


def build_context_recall_prompt(entry: Dict[str, Any]) -> str:
    """Build prompt for generating context_recall."""
    
    task_instructions = entry.get("user_scenario", {}).get("instructions", {}).get("task_instructions", "")
    known_info = entry.get("user_scenario", {}).get("instructions", {}).get("known_info", "")
    
    # Get expected actions for understanding the task complexity
    expected_actions = entry.get("evaluation_criteria", {}).get("actions", [])
    action_names = [a.get("name", "") for a in expected_actions]
    
    # Get DB context for understanding what data is available
    db = entry.get("db", {})
    user_info = ""
    if db.get("users"):
        user_id = list(db["users"].keys())[0]
        user = db["users"][user_id]
        orders = user.get("orders", [])
        user_info = f"User: {user_id}, Orders: {orders}"
    
    prompt = f"""You are helping create training data for a customer service AI agent. 

Given a customer scenario, generate a "context_recall" that the agent should mentally construct before responding.
This context_recall helps the agent track complex multi-request conversations.

## Customer Scenario
{task_instructions}

## Known Info
{known_info}

## User/Order Context
{user_info}

## Expected Actions (for reference)
{', '.join(action_names)}

## Full Policy (extract only relevant sections)
{POLICY_TEXT}

---

Generate a JSON object with this structure:
```json
{{
    "request_summary": {{
        "total_requests": <number>,
        "requests": [
            {{
                "request_id": 1,
                "description": "<concise description of what customer wants>",
                "type": "<category: exchange|return|modify_address|modify_items|cancel|info_lookup|update_profile>",
                "order_id": "<if applicable>",
                "status": "pending|in_progress|completed|blocked",
                "dependencies": ["<any request_ids this depends on>"],
                "key_details": {{
                    "<relevant field>": "<value>"
                }}
            }}
        ]
    }},
    "authentication_status": {{
        "authenticated": false,
        "method_needed": "<email|name_zip>",
        "info_available": "<what customer provided>"
    }},
    "relevant_policies": [
        {{
            "policy_name": "<short name>",
            "summary": "<1-2 sentence summary of the policy rule>",
            "applies_to_request": [<request_ids>]
        }}
    ],
    "constraints": [
        "<key constraint or rule that applies, e.g., 'orders can only be modified once'>"
    ],
    "next_logical_step": "<what the agent should do next based on current state>"
}}
```

Important:
- Be specific and actionable
- Extract ONLY policies relevant to this specific scenario
- Track dependencies between requests (e.g., must authenticate before looking up orders)
- Include any "gotchas" or tricky aspects of this scenario

Output only the JSON, no additional text."""

    return prompt


def generate_context_recall(
    entry: Dict[str, Any],
    model: str = "gpt-4o-mini",
    provider: str = "openai"
) -> Dict[str, Any]:
    """Generate context_recall for a single entry using LLM."""
    
    prompt = build_context_recall_prompt(entry)
    
    try:
        response = completion(
            model=model,
            custom_llm_provider=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
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
        print(f"  Response: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
        return {"error": str(e), "raw_response": response_text[:1000] if 'response_text' in dir() else None}
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return {"error": str(e)}


def augment_jsonl(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    dry_run: bool = False,
    limit: int = None
):
    """Augment all entries in a JSONL file with context_recall."""
    
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
    
    if limit:
        entries = entries[:limit]
        print(f"Limited to {limit} entries")
    
    augmented_entries = []
    
    for i, entry in enumerate(entries):
        entry_id = entry.get("id", i)
        task_preview = entry.get("user_scenario", {}).get("instructions", {}).get("task_instructions", "")[:80]
        print(f"\n[{i+1}/{len(entries)}] Processing entry {entry_id}...")
        print(f"  Task: {task_preview}...")
        
        if dry_run:
            # In dry run, just show the prompt
            prompt = build_context_recall_prompt(entry)
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  --- Prompt preview ---")
            print(prompt[:500])
            print("  ...")
            augmented_entries.append(entry)
        else:
            # Generate context_recall
            context_recall = generate_context_recall(entry, model, provider)
            
            # Add to entry
            entry["context_recall"] = context_recall
            augmented_entries.append(entry)
            
            # Show preview
            if "error" not in context_recall:
                num_requests = context_recall.get("request_summary", {}).get("total_requests", "?")
                num_policies = len(context_recall.get("relevant_policies", []))
                print(f"  ✓ Generated: {num_requests} requests, {num_policies} relevant policies")
            else:
                print(f"  ✗ Error: {context_recall.get('error')}")
    
    # Write output
    if not dry_run:
        with open(output_path, "w") as f:
            for entry in augmented_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"\n✓ Wrote {len(augmented_entries)} augmented entries to {output_path}")
    else:
        print(f"\n[DRY RUN] Would write {len(augmented_entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment GRPO training data with context_recall")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without making API calls")
    parser.add_argument("--limit", type=int, help="Limit number of entries to process")
    
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
