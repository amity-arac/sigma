# Copyright Amity
"""
Optimized Policy generator for trajectories.

This module extracts rejection reasons from trajectories and uses AI to
generate an optimized policy document based on the patterns identified.
It intelligently edits the existing policy rather than just appending.
The UI uses react-diff-viewer to visualize changes.
"""
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


def extract_rejections_from_trajectories(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all rejection reasons and context from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of rejection records with context
    """
    rejections = []
    
    for trajectory in trajectories:
        messages = trajectory.get('messages', [])
        wiki = trajectory.get('wiki', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', '')
        
        # Find rejected messages
        for i, msg in enumerate(messages):
            if msg.get('role') == 'rejected':
                rejected_data = msg.get('rejected', {})
                
                # Get conversation context (last few messages before rejection)
                context_messages = []
                for prev_msg in messages[max(0, i-5):i]:
                    if prev_msg.get('role') != 'rejected':
                        context_messages.append({
                            'role': prev_msg.get('role'),
                            'content': prev_msg.get('content'),
                            'tool_name': prev_msg.get('tool_name'),
                        })
                
                # Get what came after (the correct action)
                correct_action = None
                for j in range(i + 1, min(i + 3, len(messages))):
                    next_msg = messages[j]
                    if next_msg.get('role') != 'rejected':
                        correct_action = {
                            'role': next_msg.get('role'),
                            'content': next_msg.get('content'),
                            'tool_name': next_msg.get('tool_name'),
                            'tool_arguments': next_msg.get('tool_arguments'),
                            'reasoning': next_msg.get('reasoning'),
                        }
                        break
                
                rejections.append({
                    'env_name': env_name,
                    'session_id': session_id,
                    'rejected_action': {
                        'content': rejected_data.get('content'),
                        'reasoning': rejected_data.get('reasoning'),
                        'tool_name': rejected_data.get('tool_name'),
                        'tool_arguments': rejected_data.get('tool_arguments'),
                    },
                    'correct_action': correct_action,
                    'context': context_messages,
                })
    
    return rejections


def generate_optimized_policy(
    current_policy: str,
    rejections: List[Dict[str, Any]],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Use AI to generate an optimized policy based on rejection patterns.
    
    Args:
        current_policy: The current policy document content
        rejections: List of rejection records with context
        model: The OpenAI model to use
        
    Returns:
        Dictionary with optimized policy and analysis
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare rejection summary for the prompt
    rejection_summaries = []
    for i, r in enumerate(rejections[:50], 1):  # Limit to 50 rejections to avoid token limits
        summary = f"""
### Rejection {i}
**Environment:** {r.get('env_name', 'unknown')}
**Rejected Action:**
- Content: {r['rejected_action'].get('content', 'N/A')}
- Tool: {r['rejected_action'].get('tool_name', 'N/A')}
- Reasoning: {r['rejected_action'].get('reasoning', 'N/A')}

**Correct Action (what should have been done):**
- Content: {r['correct_action'].get('content', 'N/A') if r.get('correct_action') else 'N/A'}
- Tool: {r['correct_action'].get('tool_name', 'N/A') if r.get('correct_action') else 'N/A'}
- Reasoning: {r['correct_action'].get('reasoning', 'N/A') if r.get('correct_action') else 'N/A'}

**Conversation Context:**
{json.dumps(r.get('context', []), indent=2)}
"""
        rejection_summaries.append(summary)
    
    prompt = f"""You are a policy optimization expert. Your task is to analyze rejection patterns from customer service agent trajectories and INTELLIGENTLY EDIT the existing policy document.

## Current Policy
```markdown
{current_policy}
```

## Rejection Data
The following are cases where agent actions were rejected during human review. Each rejection shows:
- What the agent tried to do (rejected action)
- What the correct action should have been
- The conversation context

{''.join(rejection_summaries)}

## Analysis Task
Based on these rejections, please:

1. **Identify Patterns**: What common mistakes or policy gaps led to these rejections?

2. **Root Cause Analysis**: For each pattern, explain why agents might be making this mistake.

3. **Policy Improvements**: Suggest specific changes. You MUST intelligently edit the existing policy by:
   - **ENHANCING** existing sections that are unclear or incomplete
   - **ADDING** new rules or sections only where truly needed
   - **REMOVING or REWRITING** confusing or contradictory sections
   - **REORGANIZING** content if the structure causes confusion
   
   DO NOT just append recommendations to the end. Instead, integrate changes into the appropriate sections of the existing policy.

4. **Revised Policy**: Provide a complete revised policy document that maintains the original structure where appropriate but incorporates all necessary changes inline.

5. **Detailed Edits**: For each significant change, provide a structured edit entry that explains exactly what was changed and where.

Please provide your response in the following JSON format:
{{
    "patterns_identified": [
        {{
            "pattern_name": "string",
            "description": "string",
            "frequency": "number of occurrences",
            "example_rejections": ["list of rejection indices"]
        }}
    ],
    "root_causes": [
        {{
            "cause": "string",
            "affected_patterns": ["pattern names"]
        }}
    ],
    "recommended_changes": [
        {{
            "section": "policy section being modified (use 'New Section' for additions)",
            "change_type": "add|modify|remove|clarify|reorganize",
            "original_text": "brief quote of original text being changed (empty string for additions)",
            "new_text": "brief description or quote of new/modified text",
            "description": "what is being changed",
            "rationale": "why this change helps prevent the identified rejection patterns"
        }}
    ],
    "revised_policy": "complete markdown policy document with all changes integrated"
}}

IMPORTANT GUIDELINES:
- Preserve the overall structure and formatting style of the original policy
- Make surgical, targeted edits rather than wholesale rewrites
- Ensure new content flows naturally with existing content
- Remove redundant or contradictory guidance
- Make implicit rules explicit with specific examples where helpful
- Use the same terminology and tone as the original document
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing customer service interactions and optimizing policy documents. You make intelligent, surgical edits to policies rather than simply appending new content. You respond only in valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        result['total_rejections_analyzed'] = len(rejections)
        result['model_used'] = model
        
        # Store original policy for diff comparison in UI
        result['original_policy'] = current_policy
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "total_rejections_analyzed": len(rejections),
            "patterns_identified": [],
            "root_causes": [],
            "recommended_changes": [],
            "revised_policy": current_policy,
            "original_policy": current_policy,
        }


def convert_to_optimized_policy_format(
    trajectories: List[Dict[str, Any]],
    current_policy: str = "",
    model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """
    Convert trajectories to optimized policy format.
    
    This extracts rejections and generates an optimized policy document.
    
    Args:
        trajectories: List of trajectory dictionaries
        current_policy: The current policy document (optional, can be fetched from env)
        model: The AI model to use for optimization
        
    Returns:
        List containing a single record with the optimized policy analysis
    """
    print(f"[OptimizedPolicy] Processing {len(trajectories)} trajectories")
    
    # Extract rejections from trajectories
    rejections = extract_rejections_from_trajectories(trajectories)
    print(f"[OptimizedPolicy] Found {len(rejections)} rejections")
    
    # Get env_name from first trajectory
    env_name = None
    for traj in trajectories:
        if traj.get('env_name'):
            env_name = traj.get('env_name')
            break
    
    if len(rejections) == 0:
        return [{
            "success": False,
            "error": "No rejections found in the selected trajectories. Optimized policy requires trajectories with rejected suggestions.",
            "total_rejections_analyzed": 0,
            "env_name": env_name,
        }]
    
    # If no policy provided, try to get from first trajectory's wiki
    if not current_policy:
        for traj in trajectories:
            if traj.get('wiki'):
                current_policy = traj.get('wiki')
                break
    
    if not current_policy:
        current_policy = "No existing policy found."
    
    # Generate optimized policy
    result = generate_optimized_policy(current_policy, rejections, model)
    result['success'] = 'error' not in result or result.get('error') is None
    result['env_name'] = env_name
    
    return [result]
