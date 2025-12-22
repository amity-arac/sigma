# Copyright Amity
"""
Base utilities shared across export format converters.
"""
import json
import re
from typing import Any, Dict, Optional, Tuple


def parse_tool_call_from_content(content: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse tool name and arguments from a content string that contains embedded tool call info.
    
    The content format is typically:
    "🔧 Calling tool_name\n{json_arguments}"
    
    Args:
        content: The content string to parse
        
    Returns:
        Tuple of (tool_name, tool_arguments) or (None, None) if parsing fails
    """
    if not content:
        return None, None
    
    # Pattern: "🔧 Calling <tool_name>\n<json>"
    match = re.match(r'^🔧 Calling (\w+)\n(.+)$', content, re.DOTALL)
    if match:
        tool_name = match.group(1)
        try:
            tool_arguments = json.loads(match.group(2))
            return tool_name, tool_arguments
        except json.JSONDecodeError:
            return tool_name, {}
    
    return None, None


def serialize_tool_arguments(tool_arguments: Any) -> str:
    """
    Serialize tool arguments to a JSON string.
    
    Args:
        tool_arguments: The tool arguments (can be None, str, or dict)
        
    Returns:
        JSON string representation of the arguments
    """
    if tool_arguments is None:
        return "{}"
    elif isinstance(tool_arguments, str):
        return tool_arguments
    else:
        return json.dumps(tool_arguments, ensure_ascii=False)


def parse_tool_arguments(tool_arguments: Any) -> Dict[str, Any]:
    """
    Parse tool arguments to a dictionary.
    
    Args:
        tool_arguments: The tool arguments (can be None, str, or dict)
        
    Returns:
        Dictionary of arguments
    """
    if tool_arguments is None:
        return {}
    elif isinstance(tool_arguments, str):
        try:
            return json.loads(tool_arguments)
        except json.JSONDecodeError:
            return {}
    else:
        return tool_arguments
