# Copyright Amity
"""
FastAPI Web Server for tau-bench Simulator

This module provides a REST API and WebSocket interface for the simulator,
allowing it to be used through a web browser.

Usage:
    # Start the server
    python -m sigma.api_server --host 0.0.0.0 --port 8000
    
    # Or with uvicorn directly
    uvicorn sigma.api_server:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sigma.simulator_core import (
    SimulatorCore,
    SimulatorSessionManager,
    get_available_environments,
    load_persona_file,
)
from sigma.env_registry import get_environment_config, list_environments, DATA_ENVS_PATH
from sigma.trajectory_storage import (
    TrajectoryStorage,
    TrajectoryData,
    TrajectoryMessage,
    RejectedSuggestion,
    get_trajectory_storage,
    get_configured_backend,
    check_storage_configuration,
)
from sigma.trajectory import Trajectory, TrajectoryError


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new simulation session."""
    env_name: str = "retail"
    user_model: str = "gpt-4o"
    user_provider: str = "openai"
    agent_model: Optional[str] = None
    agent_provider: Optional[str] = None
    persona: Optional[str] = None
    persona_file: Optional[str] = None
    task_index: Optional[int] = None
    task_split: str = "test"
    generate_scenario: bool = False  # If True, auto-generate a new scenario inspired by existing tasks
    task_ids: Optional[List[int]] = None  # Optional list of task IDs to sample from when generating scenarios


class SessionResponse(BaseModel):
    """Response containing session information."""
    session_id: str
    env_name: str
    is_active: bool
    is_done: bool


class StartSessionResponse(BaseModel):
    """Response when starting a session."""
    session_id: str
    initial_message: str
    persona: str
    tools: List[Dict[str, Any]]
    wiki: str
    generated_scenario: Optional[Dict[str, Any]] = None  # Present if scenario was auto-generated


class RespondRequest(BaseModel):
    """Request to send a response to the user."""
    message: str


class ToolCallRequest(BaseModel):
    """Request to call a tool."""
    tool_name: str
    arguments: Dict[str, Any]


class GenerateResponseRequest(BaseModel):
    """Request to generate a response using LLM."""
    prompt: str


class ParseActionRequest(BaseModel):
    """Request to parse natural language into an action."""
    user_input: str


class RollbackRequest(BaseModel):
    """Request to rollback conversation to a specific point."""
    target_index: int  # The index in conversation history to rollback to (exclusive)


class RegenerateUserRequest(BaseModel):
    """Request to regenerate user (simulated user agent) response with additional note."""
    additional_note: Optional[str] = None  # Additional guidance for the user agent


class ActionResultResponse(BaseModel):
    """Response from an action."""
    success: bool
    observation: str
    done: bool
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ConversationEntry(BaseModel):
    """A conversation entry."""
    role: str
    content: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None


class ConversationHistoryResponse(BaseModel):
    """Response containing conversation history."""
    history: List[ConversationEntry]


class SaveTrajectoryMessageRequest(BaseModel):
    """A message in the trajectory save request."""
    id: str
    role: str  # 'user', 'agent', 'tool', 'tool-result', 'system', 'rejected'
    content: Optional[str] = None
    reasoning: Optional[str] = None
    timestamp: Optional[str] = None
    
    # For tool calls
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    
    # For rejected suggestions (role='rejected') - same format as normal message
    rejected: Optional[Dict[str, Any]] = None  # {content, reasoning, tool_name, tool_arguments}


class SaveTrajectoryRequest(BaseModel):
    """Request to save a trajectory."""
    messages: List[SaveTrajectoryMessageRequest]
    # Optional trajectory_id to update existing trajectory (for continued sessions)
    trajectory_id: Optional[str] = None
    # Final result info (optional, will be fetched from session if not provided)
    is_done: Optional[bool] = None
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, Any]] = None


class SaveTrajectoryResponse(BaseModel):
    """Response from saving a trajectory."""
    success: bool
    trajectory_id: Optional[str] = None
    backend: Optional[str] = None
    error: Optional[str] = None


class ContinueTrajectoryRequest(BaseModel):
    """Request to continue from an existing trajectory."""
    trajectory_id: str
    env_name: str
    user_model: str = "gpt-4o"
    user_provider: str = "openai"
    agent_model: Optional[str] = None
    agent_provider: Optional[str] = None


class ContinueTrajectoryResponse(BaseModel):
    """Response when continuing from a trajectory."""
    session_id: str
    trajectory_id: str
    env_name: str
    initial_message: str
    persona: str
    tools: List[Dict[str, Any]]
    wiki: str
    messages: List[Dict[str, Any]]  # Previous messages to restore
    is_done: bool


class UpdateTrajectoryRequest(BaseModel):
    """Request to update a trajectory."""
    is_done: Optional[bool] = None
    reward: Optional[float] = None


class ExportTrajectoryRequest(BaseModel):
    """Request to export trajectories as training data."""
    format: str  # 'dpo', 'grpo', or 'sft'
    env_name: Optional[str] = None  # Filter by environment
    trajectory_ids: Optional[List[str]] = None  # Optional: specific trajectories to export
    date_filter: Optional[str] = None  # Optional: filter by date (YYYY-MM-DD)


class ExportTrajectoryResponse(BaseModel):
    """Response from exporting trajectories."""
    success: bool
    format: str
    count: int  # Number of records exported
    data: str  # JSONL content as string
    error: Optional[str] = None


class FinalResultResponse(BaseModel):
    """Response containing final simulation result."""
    session_id: str
    env_name: str
    is_done: bool
    reward: Optional[float]
    reward_info: Optional[Dict[str, Any]]
    expected_actions: Optional[List[Dict[str, Any]]]
    conversation_history: List[Dict[str, Any]]


class EnvironmentInfo(BaseModel):
    """Information about an available environment."""
    name: str
    display_name: str
    description: str


class EnvironmentFileInfo(BaseModel):
    """Information about a file in an environment."""
    name: str
    type: str  # 'json', 'markdown', 'text'
    size: int
    description: str


class EnvironmentFilesResponse(BaseModel):
    """Response containing list of environment files."""
    env_name: str
    files: List[EnvironmentFileInfo]


class EnvironmentFileContentResponse(BaseModel):
    """Response containing file content."""
    env_name: str
    filename: str
    content: str
    type: str


class UpdateEnvironmentFileRequest(BaseModel):
    """Request to update an environment file."""
    content: str


class ToolInfoResponse(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]


# =============================================================================
# Application Setup
# =============================================================================

# Session manager (global state) - caches simulators by trajectory_id
session_manager = SimulatorSessionManager()


def get_simulator_for_trajectory(trajectory_id: str) -> SimulatorCore:
    """
    Get or create a simulator for a trajectory.
    
    This is the core helper for the trajectory-centric API:
    1. Check if simulator is already cached (for performance)
    2. If not, load trajectory from storage and create simulator
    3. Cache and return the simulator
    
    Args:
        trajectory_id: The trajectory ID
        
    Returns:
        SimulatorCore instance ready for operations
        
    Raises:
        HTTPException: If trajectory not found
    """
    # Check cache first
    simulator = session_manager.get_session(trajectory_id)
    if simulator:
        return simulator
    
    # Load trajectory from storage
    storage = get_trajectory_storage()
    
    # Try to find the trajectory in any environment
    trajectory_dict = None
    for env_name in list_environments():
        trajectory_dict = storage.get(trajectory_id, env_name)
        if trajectory_dict:
            break
    
    if not trajectory_dict:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    
    # Convert to TrajectoryData
    messages_data = trajectory_dict.get('messages', [])
    trajectory_messages = []
    for msg in messages_data:
        trajectory_messages.append(TrajectoryMessage(
            id=msg.get('id', ''),
            role=msg.get('role', 'user'),
            content=msg.get('content'),
            reasoning=msg.get('reasoning'),
            timestamp=msg.get('timestamp'),
            tool_name=msg.get('tool_name'),
            tool_arguments=msg.get('tool_arguments'),
        ))
    
    trajectory = TrajectoryData(
        id=trajectory_dict.get('id'),
        session_id=trajectory_dict.get('session_id', trajectory_id),
        created_at=trajectory_dict.get('created_at'),
        env_name=trajectory_dict.get('env_name'),
        task_index=trajectory_dict.get('task_index'),
        task_split=trajectory_dict.get('task_split', 'test'),
        task_instruction=trajectory_dict.get('task_instruction'),
        user_id=trajectory_dict.get('user_id'),
        user_model=trajectory_dict.get('user_model', 'gpt-4o'),
        user_provider=trajectory_dict.get('user_provider', 'openai'),
        agent_model=trajectory_dict.get('agent_model'),
        agent_provider=trajectory_dict.get('agent_provider'),
        persona=trajectory_dict.get('persona', ''),
        wiki=trajectory_dict.get('wiki', ''),
        persona_data=trajectory_dict.get('persona_data'),
        messages=trajectory_messages,
        is_done=trajectory_dict.get('is_done', False),
        reward=trajectory_dict.get('reward'),
        reward_info=trajectory_dict.get('reward_info'),
        expected_actions=trajectory_dict.get('expected_actions'),
    )
    
    # Create simulator from trajectory
    simulator = session_manager.create_from_trajectory(trajectory)
    
    return simulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("🚀 Starting tau-bench Simulator API Server")
    print(f"📂 Available environments: {', '.join(list_environments())}")
    yield
    # Shutdown
    print("👋 Shutting down...")
    # Cleanup all sessions
    for session_id in session_manager.list_sessions():
        session_manager.remove_session(session_id)


app = FastAPI(
    title="tau-bench Simulator API",
    description="REST API for the tau-bench CLI Simulator",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler to log 500 errors server-side
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions server-side before returning 500."""
    print(f"\n❌ UNHANDLED EXCEPTION on {request.method} {request.url.path}")
    print(f"   Error: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
REACT_DIST_DIR = STATIC_DIR / "react-app" / "dist"

# Mount React build assets if available
if REACT_DIST_DIR.exists():
    assets_dir = REACT_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Mount legacy static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Routes - General
# =============================================================================

@app.get("/")
async def root():
    """Serve the main HTML page."""
    # First try React build
    react_index = REACT_DIST_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index))
    # Try assets folder (alternate build location)
    assets_index = STATIC_DIR / "assets" / "index.html"
    if assets_index.exists():
        return FileResponse(str(assets_index))
    # Fallback to legacy static index
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>tau-bench Simulator API</h1><p>See /docs for API documentation</p>")


@app.get("/admin")
async def admin_page():
    """Serve the admin page (SPA routing - same index.html)."""
    react_index = REACT_DIST_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Admin page not available</h1>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(session_manager.list_sessions())}


@app.get("/environments", response_model=List[EnvironmentInfo])
async def get_environments():
    """Get list of available environments."""
    return get_available_environments()


@app.get("/environments/{env_name}/tasks")
async def get_environment_tasks(env_name: str, split: str = "test"):
    """Get list of tasks for an environment."""
    try:
        env_config = get_environment_config(env_name)
        if env_config.tasks_loader:
            tasks = env_config.tasks_loader(split)
            return [
                {
                    "index": i,
                    "user_id": getattr(task, "user_id", task.get("user_id", "N/A")) if isinstance(task, dict) else task.user_id,
                    "instruction": (getattr(task, "instruction", task.get("instruction", "N/A")) if isinstance(task, dict) else task.instruction)[:100] + "...",
                }
                for i, task in enumerate(tasks)
            ]
        return []
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/environments/{env_name}/wiki")
async def get_environment_wiki(env_name: str):
    """Get policy content for an environment (HTML if available, otherwise Markdown).
    
    Looks for files in the data/envs/{env_name}/ folder:
    1. policy.html (preferred for rich formatting)
    2. policy.md (standard format)
    """
    try:
        # Look within data/envs folder
        env_path = os.path.join(DATA_ENVS_PATH, env_name)
        
        if not os.path.exists(env_path):
            raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
        
        # Check for policy.html first (preferred for rich formatting)
        policy_html_path = os.path.join(env_path, "policy.html")
        if os.path.exists(policy_html_path):
            with open(policy_html_path, "r") as f:
                return {
                    "content": f.read(),
                    "content_type": "html"
                }
        
        # Check for policy.md (standard format)
        policy_md_path = os.path.join(env_path, "policy.md")
        if os.path.exists(policy_md_path):
            with open(policy_md_path, "r") as f:
                return {
                    "content": f.read(),
                    "content_type": "markdown"
                }
        
        # No policy found
        return {
            "content": None,
            "content_type": None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Routes - Environment File Management
# =============================================================================

# Define which files are editable and their descriptions
EDITABLE_ENV_FILES = {
    "db.json": {
        "type": "json",
        "description": "Database containing users, products, and orders data"
    },
    "tasks.json": {
        "type": "json",
        "description": "Tasks with user scenarios and evaluation criteria"
    },
    "policy.md": {
        "type": "markdown",
        "description": "Agent policy and behavioral rules"
    },
    "user_guidelines.md": {
        "type": "markdown",
        "description": "User simulation guidelines"
    },
    "agent_guidelines.md": {
        "type": "markdown",
        "description": "Agent-specific guidelines"
    },
}


@app.get("/environments/{env_name}/files", response_model=EnvironmentFilesResponse)
async def get_environment_files(env_name: str):
    """Get list of editable files in an environment."""
    env_path = os.path.join(DATA_ENVS_PATH, env_name)
    
    if not os.path.exists(env_path):
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    
    files = []
    for filename, info in EDITABLE_ENV_FILES.items():
        file_path = os.path.join(env_path, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            files.append(EnvironmentFileInfo(
                name=filename,
                type=info["type"],
                size=size,
                description=info["description"]
            ))
    
    return EnvironmentFilesResponse(env_name=env_name, files=files)


@app.get("/environments/{env_name}/files/{filename}", response_model=EnvironmentFileContentResponse)
async def get_environment_file(env_name: str, filename: str):
    """Get content of a specific environment file."""
    if filename not in EDITABLE_ENV_FILES:
        raise HTTPException(status_code=400, detail=f"File '{filename}' is not editable")
    
    env_path = os.path.join(DATA_ENVS_PATH, env_name)
    file_path = os.path.join(env_path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in environment '{env_name}'")
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        return EnvironmentFileContentResponse(
            env_name=env_name,
            filename=filename,
            content=content,
            type=EDITABLE_ENV_FILES[filename]["type"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/environments/{env_name}/files/{filename}")
async def update_environment_file(env_name: str, filename: str, request: UpdateEnvironmentFileRequest):
    """Update content of an environment file."""
    if filename not in EDITABLE_ENV_FILES:
        raise HTTPException(status_code=400, detail=f"File '{filename}' is not editable")
    
    env_path = os.path.join(DATA_ENVS_PATH, env_name)
    file_path = os.path.join(env_path, filename)
    
    if not os.path.exists(env_path):
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    
    try:
        # Validate JSON files
        if EDITABLE_ENV_FILES[filename]["type"] == "json":
            try:
                json.loads(request.content)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(request.content)
        
        return {"success": True, "message": f"File '{filename}' updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Routes - Sessions
# =============================================================================

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new simulation session."""
    # Load persona file if provided
    persona_data = None
    if request.persona_file:
        if not os.path.exists(request.persona_file):
            raise HTTPException(status_code=400, detail=f"Persona file not found: {request.persona_file}")
        persona_data = load_persona_file(request.persona_file)
    
    try:
        simulator = session_manager.create_session(
            env_name=request.env_name,
            user_model=request.user_model,
            user_provider=request.user_provider,
            agent_model=request.agent_model,
            agent_provider=request.agent_provider,
            persona=request.persona,
            persona_data=persona_data,
            task_index=request.task_index,
            task_split=request.task_split,
            generate_scenario=request.generate_scenario,
            task_ids=request.task_ids,
        )
        
        return SessionResponse(
            session_id=simulator.session_id,
            env_name=simulator.env_name,
            is_active=True,
            is_done=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sessions", response_model=List[str])
async def list_sessions():
    """List all active session IDs."""
    return session_manager.list_sessions()


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=simulator.session_id,
        env_name=simulator.env_name,
        is_active=simulator.state.is_active,
        is_done=simulator.is_done,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_manager.remove_session(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions/{session_id}/start", response_model=StartSessionResponse)
async def start_session(session_id: str):
    """Start the simulation and get the initial user message."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        initial_message = simulator.start()
        
        # Convert tools to dict format
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "required_params": t.required_params,
            }
            for t in simulator.tools
        ]
        
        # Include generated scenario info if available
        generated_scenario = None
        if simulator.generated_scenario:
            generated_scenario = {
                "instruction": simulator.generated_scenario.instruction,
                "user": simulator.generated_scenario.user,
                "data": simulator.generated_scenario.data,
                "data_key": simulator.generated_scenario.data_key,
                "seed_task_instruction": simulator.generated_scenario.seed_task_instruction,
                "generation_timestamp": simulator.generated_scenario.generation_timestamp,
                "env_name": simulator.generated_scenario.env_name,
            }
        
        return StartSessionResponse(
            session_id=simulator.session_id,
            initial_message=initial_message,
            persona=simulator.current_persona,
            tools=tools,
            wiki=simulator.wiki,
            generated_scenario=generated_scenario,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Routes - Actions
# =============================================================================

@app.post("/sessions/{session_id}/respond", response_model=ActionResultResponse)
async def respond_to_user(session_id: str, request: RespondRequest):
    """Send a text response to the user."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.respond_to_user(request.message)
    
    return ActionResultResponse(
        success=result.success,
        observation=result.observation,
        done=result.done,
        reward=result.reward,
        reward_info=result.reward_info,
        error=result.error,
    )


@app.post("/sessions/{session_id}/tool", response_model=ActionResultResponse)
async def call_tool(session_id: str, request: ToolCallRequest):
    """Call a tool with the given arguments."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.call_tool(request.tool_name, request.arguments)
    
    return ActionResultResponse(
        success=result.success,
        observation=result.observation,
        done=result.done,
        reward=result.reward,
        reward_info=result.reward_info,
        error=result.error,
    )


@app.post("/sessions/{session_id}/generate-response")
async def generate_response(session_id: str, request: GenerateResponseRequest):
    """Generate a response using LLM based on a prompt."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    response = simulator.generate_response(request.prompt)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return {"response": response}


@app.post("/sessions/{session_id}/parse-action")
async def parse_action(session_id: str, request: ParseActionRequest):
    """Parse natural language into a structured action."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    parsed = simulator.parse_natural_language_action(request.user_input)
    if parsed is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not parse action. Check server logs for details. Agent model: {simulator.agent_model}"
        )
    
    return parsed


@app.post("/sessions/{session_id}/undo")
async def undo_last_action(session_id: str):
    """Undo the last action in the conversation."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.undo_last_action()
    return result


@app.post("/sessions/{session_id}/rollback")
async def rollback_to_point(session_id: str, request: RollbackRequest):
    """Rollback conversation to a specific point (removing all messages after target_index)."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.rollback_to_index(request.target_index)
    return result


@app.post("/sessions/{session_id}/regenerate-user")
async def regenerate_user_response(session_id: str, request: RegenerateUserRequest):
    """Regenerate the simulated user's response with optional additional guidance."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.regenerate_user_response(request.additional_note)
    return result


# =============================================================================
# Routes - Information
# =============================================================================

@app.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str):
    """Get the conversation history for a session."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = [
        ConversationEntry(
            role=e.role,
            content=e.content,
            tool_call=e.tool_call,
        )
        for e in simulator.conversation_history
    ]
    
    return ConversationHistoryResponse(history=history)


@app.get("/sessions/{session_id}/tools", response_model=List[ToolInfoResponse])
async def get_session_tools(session_id: str):
    """Get available tools for a session."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return [
        ToolInfoResponse(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
            required_params=t.required_params,
        )
        for t in simulator.tools
    ]


@app.get("/sessions/{session_id}/wiki")
async def get_session_wiki(session_id: str):
    """Get the wiki/policy for a session."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"wiki": simulator.wiki}


@app.get("/sessions/{session_id}/debug-context")
async def get_debug_context(session_id: str):
    """Get the current agent context for debugging (what the LLM sees)."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    context = simulator._build_agent_context()
    return {
        "context": context,
        "history_count": len(simulator.conversation_history),
        "history_entries": [
            {
                "role": e.role,
                "content": e.content[:200] if e.content else None,
                "tool_call": e.tool_call,
            }
            for e in simulator.conversation_history
        ]
    }


@app.get("/sessions/{session_id}/result", response_model=FinalResultResponse)
async def get_final_result(session_id: str):
    """Get the final result of a completed session."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = simulator.get_final_result()
    
    return FinalResultResponse(**result)


# =============================================================================
# Routes - Trajectory Storage
# =============================================================================

@app.get("/trajectory/status")
async def get_trajectory_storage_status():
    """Check trajectory storage configuration and backend."""
    config = check_storage_configuration()
    backend = get_configured_backend()
    return {
        "configured": True,  # Always configured (local is always available)
        "backend": backend,
        "message": f"Using {backend} storage backend",
        "details": config
    }


@app.post("/sessions/{session_id}/save-trajectory", response_model=SaveTrajectoryResponse)
async def save_trajectory(session_id: str, request: SaveTrajectoryRequest):
    """
    Save the current session trajectory.
    
    This saves the complete session data including:
    - Task instruction and environment info
    - All messages (user, agent, tool calls, tool results)
    - Reasoning content for each agent action
    - Rejected suggestions (inline in messages with role='rejected')
    - Final reward and result info
    
    Storage backend is auto-detected from environment:
    - Azure Blob Storage (recommended for analytics) - set AZURE_STORAGE_CONNECTION_STRING
    - Local filesystem (default fallback)
    """
    # Debug: Log incoming request
    print(f"[save_trajectory] Session: {session_id}, Messages received: {len(request.messages)}")
    for i, m in enumerate(request.messages):
        print(f"[save_trajectory] Message {i}: role={m.role}, tool_name={m.tool_name}, has_content={bool(m.content)}")
    
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        storage = get_trajectory_storage()
        
        # Build trajectory messages from request
        messages = []
        for m in request.messages:
            rejected_data = None
            if m.rejected:
                rejected_data = RejectedSuggestion(
                    content=m.rejected.get('content'),
                    reasoning=m.rejected.get('reasoning'),
                    tool_name=m.rejected.get('tool_name'),
                    tool_arguments=m.rejected.get('tool_arguments'),
                )
            messages.append(TrajectoryMessage(
                id=str(m.id),
                role=m.role,
                content=m.content,
                reasoning=m.reasoning,
                timestamp=m.timestamp,
                tool_name=m.tool_name,
                tool_arguments=m.tool_arguments,
                rejected=rejected_data,
            ))
        
        # Get task instruction from the simulator's environment
        task_instruction = None
        user_id = None
        if hasattr(simulator, 'env') and hasattr(simulator.env, 'task'):
            task_instruction = getattr(simulator.env.task, 'instruction', None)
            user_id = getattr(simulator.env.task, 'user_id', None)
        
        # If scenario was generated, try to get user_id from there
        if not user_id and simulator.generated_scenario:
            user_id = getattr(simulator.generated_scenario, 'user_id', None)
        
        # Get persona_data for session resumption
        # This includes user profile, orders/reservations, and augmented_data
        persona_data = simulator.persona_data if hasattr(simulator, 'persona_data') else None
        
        # If updating existing trajectory, preserve original session_id and created_at
        original_session_id = session_id
        original_created_at = None
        if request.trajectory_id:
            existing = storage.get(request.trajectory_id, simulator.env_name)
            if existing:
                original_session_id = existing.get('session_id', session_id)
                original_created_at = existing.get('created_at')
        
        # Build trajectory data
        # If trajectory_id is provided (e.g., for continued sessions), use it to update existing
        trajectory = TrajectoryData(
            id=request.trajectory_id,  # Use provided ID or None (will generate new)
            session_id=original_session_id,  # Preserve original session_id when updating
            created_at=original_created_at,  # Preserve original created_at when updating
            env_name=simulator.env_name,
            task_index=simulator.task_index,
            task_split=simulator.task_split,
            task_instruction=task_instruction,
            user_id=user_id,
            user_model=simulator.user_model,
            user_provider=simulator.user_provider,
            agent_model=simulator.agent_model,
            agent_provider=simulator.agent_provider,
            persona=simulator.current_persona or "",
            wiki=simulator.wiki or "",
            persona_data=persona_data,  # Full persona data for session resumption
            messages=messages,
            is_done=request.is_done if request.is_done is not None else simulator.state.is_done,
            reward=request.reward if request.reward is not None else simulator.state.last_reward,
            reward_info=request.reward_info or simulator.state.reward_info,
            expected_actions=simulator.state.expected_actions,
        )
        
        # Save to storage (will update existing if trajectory.id is set)
        trajectory_id = storage.save(trajectory)
        
        print(f"[save_trajectory] Saved trajectory: {trajectory_id} (requested_id: {request.trajectory_id})")
        
        return SaveTrajectoryResponse(
            success=True,
            trajectory_id=trajectory_id,
            backend=storage.backend_type,
        )
        
    except Exception as e:
        return SaveTrajectoryResponse(
            success=False,
            error=str(e),
        )


@app.get("/trajectories")
async def list_trajectories(env_name: Optional[str] = None, limit: int = 100):
    """List saved trajectories."""
    try:
        storage = get_trajectory_storage()
        trajectories = storage.list(env_name=env_name, limit=limit)
        return {
            "trajectories": trajectories,
            "backend": storage.backend_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# New Request/Response Models for Trajectory-centric API
# =============================================================================

class CreateTrajectoryRequest(BaseModel):
    """Request to create a new trajectory."""
    env_name: str = "retail"
    user_model: str = "gpt-4o"
    user_provider: str = "openai"
    agent_model: Optional[str] = None
    agent_provider: Optional[str] = None
    persona: Optional[str] = None
    task_index: Optional[int] = None
    task_split: str = "test"
    generate_scenario: bool = True  # Default to generating scenario
    task_ids: Optional[List[int]] = None


class CreateTrajectoryResponse(BaseModel):
    """Response when creating a new trajectory."""
    trajectory_id: str
    env_name: str
    initial_message: str
    persona: str
    tools: List[Dict[str, Any]]
    wiki: str
    generated_scenario: Optional[Dict[str, Any]] = None


@app.post("/trajectories", response_model=CreateTrajectoryResponse)
async def create_trajectory(request: CreateTrajectoryRequest):
    """
    Create a new trajectory with scenario generation.
    
    This is the main entry point for starting a new simulation:
    1. Creates a simulator with optional scenario generation
    2. Starts the simulation to get the initial user message
    3. Saves the initial state as a trajectory
    4. Returns the trajectory_id for subsequent operations
    
    All subsequent operations use the trajectory_id:
    - POST /trajectories/{id}/respond - Send agent response
    - POST /trajectories/{id}/tool - Execute tool call
    - PUT /trajectories/{id} - Save/update trajectory
    """
    try:
        # Create simulator
        simulator = session_manager.create_session(
            env_name=request.env_name,
            user_model=request.user_model,
            user_provider=request.user_provider,
            agent_model=request.agent_model,
            agent_provider=request.agent_provider,
            persona=request.persona,
            task_index=request.task_index,
            task_split=request.task_split,
            generate_scenario=request.generate_scenario,
            task_ids=request.task_ids,
        )
        
        # Start simulation to get initial message
        initial_message = simulator.start()
        
        # Convert tools to dict format
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "required_params": t.required_params,
            }
            for t in simulator.tools
        ]
        
        # Get generated scenario info if available
        generated_scenario = None
        if simulator.generated_scenario:
            generated_scenario = {
                "instruction": simulator.generated_scenario.instruction,
                "user_id": simulator.generated_scenario.user_id,
                "initial_message": initial_message,  # Use the initial_message from simulator.start()
                "expected_actions": simulator.state.expected_actions or [],  # Use expected_actions from state
            }
        
        # Save initial trajectory
        storage = get_trajectory_storage()
        
        # Get task instruction
        task_instruction = None
        user_id = None
        if hasattr(simulator, 'env') and hasattr(simulator.env, 'task'):
            task_instruction = getattr(simulator.env.task, 'instruction', None)
            user_id = getattr(simulator.env.task, 'user_id', None)
        if not user_id and simulator.generated_scenario:
            user_id = getattr(simulator.generated_scenario, 'user_id', None)
        
        # Create initial message for trajectory
        initial_messages = [
            TrajectoryMessage(
                id=str(hash(initial_message)),
                role="user",
                content=initial_message,
                timestamp=None,
            )
        ]
        
        trajectory = TrajectoryData(
            id=simulator.session_id,  # Use session_id as trajectory_id
            session_id=simulator.session_id,
            env_name=simulator.env_name,
            task_index=simulator.task_index,
            task_split=simulator.task_split,
            task_instruction=task_instruction,
            user_id=user_id,
            user_model=simulator.user_model,
            user_provider=simulator.user_provider,
            agent_model=simulator.agent_model,
            agent_provider=simulator.agent_provider,
            persona=simulator.current_persona or "",
            wiki=simulator.wiki or "",
            persona_data=simulator.persona_data,
            messages=initial_messages,
            is_done=False,
            expected_actions=simulator.state.expected_actions,
        )
        
        trajectory_id = storage.save(trajectory)
        
        return CreateTrajectoryResponse(
            trajectory_id=trajectory_id,
            env_name=simulator.env_name,
            initial_message=initial_message,
            persona=simulator.current_persona or "",
            tools=tools,
            wiki=simulator.wiki or "",
            generated_scenario=generated_scenario,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"\n❌ ERROR in create_trajectory: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trajectories/{trajectory_id}")
async def get_trajectory_by_id(trajectory_id: str, env_name: str):
    """Get a specific trajectory."""
    try:
        storage = get_trajectory_storage()
        trajectory = storage.get(trajectory_id, env_name)
        if not trajectory:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        return trajectory
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/trajectories/{trajectory_id}")
async def delete_trajectory(trajectory_id: str, env_name: str):
    """Delete a specific trajectory."""
    try:
        storage = get_trajectory_storage()
        success = storage.delete(trajectory_id, env_name)
        if not success:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        return {"message": "Trajectory deleted successfully", "id": trajectory_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/trajectories/{trajectory_id}")
async def update_trajectory(trajectory_id: str, env_name: str, request: UpdateTrajectoryRequest):
    """Update a trajectory (e.g., mark as complete)."""
    try:
        storage = get_trajectory_storage()
        
        # Build updates dict from non-None fields
        updates = {}
        if request.is_done is not None:
            updates["is_done"] = request.is_done
        if request.reward is not None:
            updates["reward"] = request.reward
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        success = storage.update(trajectory_id, env_name, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        
        return {"message": "Trajectory updated successfully", "id": trajectory_id, "updates": updates}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trajectory Action Endpoints (new trajectory-centric API)
# =============================================================================

@app.post("/trajectories/{trajectory_id}/respond", response_model=ActionResultResponse)
async def trajectory_respond(trajectory_id: str, request: RespondRequest):
    """
    Send an agent response in the trajectory simulation.
    
    This loads the trajectory, sends the response, and the frontend should
    save the updated messages via PUT /trajectories/{id}.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        result = simulator.send_response(request.message)
        
        return ActionResultResponse(
            success=True,
            observation=result.observation,
            done=result.done,
            reward=result.reward,
            reward_info=result.info,
        )
    except HTTPException:
        raise
    except Exception as e:
        return ActionResultResponse(
            success=False,
            observation="",
            done=False,
            error=str(e),
        )


@app.post("/trajectories/{trajectory_id}/tool", response_model=ActionResultResponse)
async def trajectory_tool_call(trajectory_id: str, request: ToolCallRequest):
    """
    Execute a tool call in the trajectory simulation.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        result = simulator.call_tool(request.tool_name, request.arguments)
        
        return ActionResultResponse(
            success=True,
            observation=result.observation,
            done=result.done,
            reward=result.reward,
            reward_info=result.info,
        )
    except HTTPException:
        raise
    except Exception as e:
        return ActionResultResponse(
            success=False,
            observation="",
            done=False,
            error=str(e),
        )


@app.post("/trajectories/{trajectory_id}/parse-action")
async def trajectory_parse_action(trajectory_id: str, request: ParseActionRequest):
    """
    Parse natural language input into a structured action.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        action = simulator.parse_action(request.user_input)
        
        return {
            "action_type": action.action_type.value if hasattr(action.action_type, 'value') else action.action_type,
            "content": action.content,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "reasoning": action.reasoning,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trajectories/{trajectory_id}/rollback")
async def trajectory_rollback(trajectory_id: str, request: RollbackRequest):
    """
    Rollback trajectory to a specific point.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        removed_count = simulator.rollback(request.target_index)
        
        # Re-enable simulation if it was done
        if simulator.state.is_done:
            simulator.state.is_done = False
            simulator.state.last_reward = None
            simulator.state.reward_info = None
        
        return {
            "success": True,
            "removed_count": removed_count,
            "new_length": len(simulator.state.conversation_history),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/trajectories/{trajectory_id}/regenerate-user")
async def trajectory_regenerate_user(trajectory_id: str, request: RegenerateUserRequest):
    """
    Regenerate the last user response with optional additional guidance.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        new_response = simulator.regenerate_user_response(request.additional_note)
        
        return {
            "success": True,
            "observation": new_response,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/trajectories/{trajectory_id}/tools")
async def trajectory_get_tools(trajectory_id: str):
    """Get available tools for a trajectory."""
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "required_params": t.required_params,
            }
            for t in simulator.tools
        ]
        return {"tools": tools}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trajectories/{trajectory_id}/wiki")
async def trajectory_get_wiki(trajectory_id: str):
    """Get wiki content for a trajectory."""
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        return {"wiki": simulator.wiki or ""}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/trajectories/{trajectory_id}/messages")
async def trajectory_save_messages(trajectory_id: str, request: SaveTrajectoryRequest):
    """
    Save/update trajectory messages.
    
    This is the new way to save trajectory state - just update the messages.
    The trajectory_id in the URL is used; request.trajectory_id is ignored.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        storage = get_trajectory_storage()
        
        # Load existing trajectory to preserve metadata
        existing = storage.get(trajectory_id, simulator.env_name)
        if not existing:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        
        # Build messages from request
        messages = []
        for m in request.messages:
            rejected_data = None
            if m.rejected:
                rejected_data = RejectedSuggestion(
                    content=m.rejected.get('content'),
                    reasoning=m.rejected.get('reasoning'),
                    tool_name=m.rejected.get('tool_name'),
                    tool_arguments=m.rejected.get('tool_arguments'),
                )
            messages.append(TrajectoryMessage(
                id=str(m.id),
                role=m.role,
                content=m.content,
                reasoning=m.reasoning,
                timestamp=m.timestamp,
                tool_name=m.tool_name,
                tool_arguments=m.tool_arguments,
                rejected=rejected_data,
            ))
        
        # Build updated trajectory
        trajectory = TrajectoryData(
            id=trajectory_id,
            session_id=existing.get('session_id', trajectory_id),
            created_at=existing.get('created_at'),
            env_name=simulator.env_name,
            task_index=simulator.task_index,
            task_split=simulator.task_split,
            task_instruction=existing.get('task_instruction'),
            user_id=existing.get('user_id'),
            user_model=simulator.user_model,
            user_provider=simulator.user_provider,
            agent_model=simulator.agent_model,
            agent_provider=simulator.agent_provider,
            persona=simulator.current_persona or "",
            wiki=simulator.wiki or "",
            persona_data=existing.get('persona_data'),
            messages=messages,
            is_done=request.is_done if request.is_done is not None else simulator.state.is_done,
            reward=request.reward if request.reward is not None else simulator.state.last_reward,
            reward_info=request.reward_info or simulator.state.reward_info,
            expected_actions=simulator.state.expected_actions,
        )
        
        # Save
        saved_id = storage.save(trajectory)
        
        return SaveTrajectoryResponse(
            success=True,
            trajectory_id=saved_id,
            backend=storage.backend_type,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return SaveTrajectoryResponse(
            success=False,
            error=str(e),
        )


@app.post("/trajectories/{trajectory_id}/continue", response_model=ContinueTrajectoryResponse)
async def continue_trajectory(trajectory_id: str, request: ContinueTrajectoryRequest):
    """
    Continue a simulation from an existing trajectory.
    
    This creates a new session initialized with the conversation state from
    the specified trajectory, allowing the user to continue from where they left off.
    
    Uses SimulatorCore.from_trajectory() for clean, decoupled session restoration.
    The trajectory's persona_data (including augmented_data) is properly restored
    so that all tools work correctly with the injected data.
    
    The trajectory's messages will be returned so the frontend can restore the UI state.
    A new session is created with the same environment configuration.
    """
    try:
        storage = get_trajectory_storage()
        trajectory_dict = storage.get(trajectory_id, request.env_name)
        
        if not trajectory_dict:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        
        # Convert dict to TrajectoryData for type safety
        # Handle messages conversion
        messages_data = trajectory_dict.get('messages', [])
        trajectory_messages = []
        for msg in messages_data:
            trajectory_messages.append(TrajectoryMessage(
                id=msg.get('id', ''),
                role=msg.get('role', 'user'),
                content=msg.get('content'),
                reasoning=msg.get('reasoning'),
                timestamp=msg.get('timestamp'),
                tool_name=msg.get('tool_name'),
                tool_arguments=msg.get('tool_arguments'),
            ))
        
        trajectory = TrajectoryData(
            id=trajectory_dict.get('id'),
            session_id=trajectory_dict.get('session_id', trajectory_id),
            created_at=trajectory_dict.get('created_at'),
            env_name=trajectory_dict.get('env_name', request.env_name),
            task_index=trajectory_dict.get('task_index'),
            task_split=trajectory_dict.get('task_split', 'test'),
            task_instruction=trajectory_dict.get('task_instruction'),
            user_id=trajectory_dict.get('user_id'),
            user_model=trajectory_dict.get('user_model', request.user_model),
            user_provider=trajectory_dict.get('user_provider', request.user_provider),
            agent_model=trajectory_dict.get('agent_model'),
            agent_provider=trajectory_dict.get('agent_provider'),
            persona=trajectory_dict.get('persona', ''),
            wiki=trajectory_dict.get('wiki', ''),
            persona_data=trajectory_dict.get('persona_data'),  # Full persona data for restoration
            messages=trajectory_messages,
            is_done=trajectory_dict.get('is_done', False),
            reward=trajectory_dict.get('reward'),
            reward_info=trajectory_dict.get('reward_info'),
            expected_actions=trajectory_dict.get('expected_actions'),
        )
        
        # Use the clean from_trajectory approach
        # This properly handles persona_data injection including augmented_data
        simulator = session_manager.create_from_trajectory(
            trajectory=trajectory,
            user_model=request.user_model,
            user_provider=request.user_provider,
            agent_model=request.agent_model,
            agent_provider=request.agent_provider,
        )
        
        # Start the session (generates initial user message for fresh sessions,
        # but for restored sessions we already have the conversation)
        # Only start if conversation is empty (fresh session)
        if not simulator.state.conversation_history:
            simulator.start()
        
        # Convert tools to dict format
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "required_params": t.required_params,
            }
            for t in simulator.tools
        ]
        
        # Get the first user message for initial_message
        initial_message = ''
        for msg in messages_data:
            if msg.get('role') == 'user':
                initial_message = msg.get('content', '')
                break
        
        return ContinueTrajectoryResponse(
            session_id=simulator.session_id,
            trajectory_id=trajectory_id,
            env_name=trajectory.env_name,
            initial_message=initial_message,
            persona=trajectory.persona,
            tools=tools,
            wiki=trajectory.wiki,
            messages=messages_data,
            is_done=trajectory.is_done,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trajectories/export", response_model=ExportTrajectoryResponse)
async def export_trajectories(request: ExportTrajectoryRequest):
    """
    Export trajectories as training data in DPO, GRPO, or SFT format.
    
    DPO Format:
    - Creates chosen/rejected pairs from trajectories with rejected suggestions
    - Each row contains: chosen (message list), rejected (message list), task_id, mistake_reason
    
    GRPO Format:
    - Extracts task definition with ground truth tool call sequences
    - Output matches tau_bench tasks.py format: id, user_id, instruction, actions
    
    SFT Format (Supervised Fine-Tuning):
    - Exports conversations with the next correct action as the answer
    - Each row contains: task_id, conversations, answer, rubric, reject_rubric
    - If rejected suggestions exist, includes reject_answer_raw and chosen_answer_raw
    """
    try:
        storage = get_trajectory_storage()
        
        # Get trajectories based on filters
        trajectories_list = storage.list(env_name=request.env_name, limit=10000)
        print(f"[Export] Initial trajectories from storage: {len(trajectories_list)}")
        print(f"[Export] Request filters - env_name: {request.env_name}, trajectory_ids: {request.trajectory_ids}, date_filter: {request.date_filter}")
        
        # Filter by trajectory IDs if provided
        if request.trajectory_ids:
            before_count = len(trajectories_list)
            trajectories_list = [t for t in trajectories_list if t.get('id') in request.trajectory_ids]
            print(f"[Export] After trajectory_ids filter: {len(trajectories_list)} (was {before_count})")
        
        # Filter by date if provided
        if request.date_filter:
            before_count = len(trajectories_list)
            trajectories_list = [
                t for t in trajectories_list 
                if t.get('created_at', '').startswith(request.date_filter)
            ]
            print(f"[Export] After date_filter: {len(trajectories_list)} (was {before_count})")
        
        # Load full trajectory data for each
        full_trajectories = []
        for t in trajectories_list:
            trajectory = storage.get(t['id'], t['env_name'])
            if trajectory:
                full_trajectories.append(trajectory)
            else:
                print(f"[Export] Warning: Could not load trajectory {t['id']} from {t['env_name']}")
        
        print(f"[Export] Full trajectories loaded: {len(full_trajectories)}")
        
        # Convert based on format
        if request.format.lower() == 'dpo':
            result = _convert_to_dpo_format(full_trajectories)
            print(f"[Export] DPO conversion result: {len(result)} records")
        elif request.format.lower() == 'grpo':
            result = _convert_to_grpo_format(full_trajectories)
            print(f"[Export] GRPO conversion result: {len(result)} records")
        elif request.format.lower() == 'sft':
            result = _convert_to_sft_format(full_trajectories)
            print(f"[Export] SFT conversion result: {len(result)} records")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}. Use 'dpo', 'grpo', or 'sft'")
        
        return ExportTrajectoryResponse(
            success=True,
            format=request.format,
            count=len(result),
            data='\n'.join(json.dumps(r, ensure_ascii=False) for r in result)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Export] Error: {e}")
        traceback.print_exc()
        return ExportTrajectoryResponse(
            success=False,
            format=request.format,
            count=0,
            data="",
            error=str(e)
        )


def _convert_to_dpo_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to DPO training format.
    
    DPO (Direct Preference Optimization) format:
    - prompt: list of messages representing the conversation history up to the decision point
    - chosen: list containing the single correct action (what was actually done)
    - rejected: list containing the single rejected action (what was proposed but rejected)
    
    This function looks for trajectories with rejected suggestions (role='rejected')
    and creates training pairs where the model learns to prefer the chosen action over rejected.
    """
    dpo_records = []
    
    print(f"[DPO] Processing {len(trajectories)} trajectories")
    
    for traj_idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        task_instruction = trajectory.get('task_instruction', '')
        wiki = trajectory.get('wiki', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', '')
        
        # Count rejected messages in this trajectory
        rejected_count = sum(1 for m in messages if m.get('role') == 'rejected')
        print(f"[DPO] Trajectory {traj_idx} ({session_id[:8] if session_id else 'N/A'}...): {len(messages)} messages, {rejected_count} rejected")
        
        # Build system message
        system_content = f"""
<instructions>
You are a customer service agent that helps the user according to the <policy>
provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.
Try to be helpful and always follow the policy. Always make sure you generate
valid JSON only.
</instructions>
<policy>
{wiki}
</policy>
"""
        
        # Find rejected suggestions and create DPO pairs
        for i, msg in enumerate(messages):
            if msg.get('role') == 'rejected':
                print(f"[DPO]   Found rejection at message {i}")
                # We found a rejection point - create a DPO pair
                rejected_data = msg.get('rejected', {})
                
                # Build conversation history (prompt) up to this point
                prompt = []
                prompt.append({
                    "role": "system",
                    "content": system_content.strip()
                })
                
                for prev_msg in messages[:i]:
                    if prev_msg.get('role') == 'rejected':
                        continue  # Skip previous rejections
                    
                    converted = _convert_message_for_dpo(prev_msg)
                    if converted:
                        prompt.append(converted)
                
                # The chosen action is the one that came after (find next non-rejected message)
                chosen_action = None
                for j in range(i + 1, len(messages)):
                    next_msg = messages[j]
                    if next_msg.get('role') != 'rejected':
                        chosen_action = _convert_message_for_dpo(next_msg)
                        break
                
                # Build rejected action from the rejected suggestion
                rejected_action = None
                if rejected_data:
                    tool_name = rejected_data.get('tool_name')
                    tool_arguments = rejected_data.get('tool_arguments')
                    content = rejected_data.get('content')
                    reasoning = rejected_data.get('reasoning', '')
                    
                    # If tool_name is null, try to parse from content field
                    if not tool_name and content:
                        parsed_tool = _parse_tool_call_from_content(content)
                        if parsed_tool:
                            tool_name = parsed_tool['name']
                            tool_arguments = parsed_tool['arguments']
                    
                    # Only create tool_calls if we have a valid tool_name
                    if tool_name:
                        # Ensure tool_arguments is properly serialized
                        if tool_arguments is None:
                            args_str = "{}"
                        elif isinstance(tool_arguments, str):
                            args_str = tool_arguments
                        else:
                            args_str = json.dumps(tool_arguments, ensure_ascii=False)
                        
                        rejected_action = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "function": {
                                    "name": tool_name,
                                    "arguments": args_str
                                },
                                "type": "function"
                            }],
                            "reasoning_content": reasoning
                        }
                    elif content:
                        rejected_action = {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning
                        }
                
                # Only create DPO record if we have valid chosen and rejected actions
                if chosen_action and rejected_action:
                    dpo_records.append({
                        "prompt": prompt,
                        "chosen": [chosen_action],
                        "rejected": [rejected_action]
                    })
    
    return dpo_records


def _parse_tool_call_from_content(content: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse tool name and arguments from a content string that contains embedded tool call info.
    
    The content format is typically:
    "🔧 Calling tool_name\n{json_arguments}"
    
    Returns:
        (tool_name, tool_arguments) or (None, None) if parsing fails
    """
    if not content:
        return None, None
    
    # Check for the tool call pattern
    import re
    
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


def _convert_message_for_dpo(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a trajectory message to DPO format."""
    role = msg.get('role', '')
    
    if role == 'user':
        return {
            "role": "user",
            "content": msg.get('content', '')
        }
    elif role == 'agent':
        result = {
            "role": "assistant",
            "content": msg.get('content')
        }
        if msg.get('reasoning'):
            result["reasoning_content"] = msg.get('reasoning')
        # Check if this agent message has tool_calls
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        if tool_name:
            if tool_arguments is None:
                args_str = "{}"
            elif isinstance(tool_arguments, str):
                args_str = tool_arguments
            else:
                args_str = json.dumps(tool_arguments, ensure_ascii=False)
            result["tool_calls"] = [{
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                },
                "type": "function"
            }]
        return result
    elif role == 'tool':
        # Tool call from agent - this represents an assistant making a tool call
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        
        # If tool_name is not set, try to parse it from the content field
        # (some trajectories store tool calls as "🔧 Calling tool_name\n{args}")
        if not tool_name:
            content = msg.get('content', '')
            tool_name, tool_arguments = _parse_tool_call_from_content(content)
        
        # Skip if we still have no tool_name - this would create corrupted data
        if not tool_name:
            return None
        
        # Properly serialize tool_arguments
        if tool_arguments is None:
            args_str = "{}"
        elif isinstance(tool_arguments, str):
            args_str = tool_arguments
        else:
            args_str = json.dumps(tool_arguments, ensure_ascii=False)
        
        result = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                },
                "type": "function"
            }]
        }
        if msg.get('reasoning'):
            result["reasoning_content"] = msg.get('reasoning')
        return result
    elif role == 'tool-result':
        # Tool response - this goes back to the model as role "tool"
        tool_call_id = msg.get('id') or msg.get('tool_call_id') or str(msg.get('timestamp', ''))
        tool_name = msg.get('tool_name', '')
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": msg.get('content', '')
        }
    elif role == 'system':
        return {
            "role": "system",
            "content": msg.get('content', '')
        }
    
    return None


def _convert_to_grpo_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to GRPO training format matching tasks.json structure.
    
    The exported format matches the tasks.json schema so it can be used directly
    as new tasks for training or evaluation:
    
    {
        "id": "string",
        "description": { "purpose": null, "relevant_policies": null, "notes": null },
        "user_scenario": {
            "persona": null,
            "instructions": {
                "task_instructions": "...",
                "domain": "retail",
                "reason_for_call": "...",
                "known_info": "...",
                "unknown_info": "..."
            }
        },
        "initial_state": null,
        "evaluation_criteria": {
            "actions": [...],  # Ground truth: actual tool calls from trajectory
            "communicate_info": [...],
            "nl_assertions": null
        }
    }
    
    The 'actions' in evaluation_criteria is the ground truth sequence of tool calls
    that actually happened during the trajectory.
    """
    grpo_records = []
    
    print(f"[GRPO] Processing {len(trajectories)} trajectories")
    
    for idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        task_instruction = trajectory.get('task_instruction', '')
        user_id = trajectory.get('user_id', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', trajectory.get('id', ''))
        reward = trajectory.get('reward', 0)
        persona = trajectory.get('persona', '')
        persona_data = trajectory.get('persona_data', {})
        
        # Extract tool call sequence from the actual trajectory (ground truth)
        actions = []
        action_counter = 0
        for msg in messages:
            if msg.get('role') == 'tool':
                tool_name = msg.get('tool_name')
                tool_args = msg.get('tool_arguments', {})
                
                # If tool_name is not set, try to parse from content
                # Format: "🔧 Calling tool_name\n{...json args...}"
                if not tool_name and msg.get('content'):
                    content = msg.get('content', '')
                    if content.startswith('🔧 Calling '):
                        try:
                            # Extract tool name from first line
                            first_line = content.split('\n')[0]
                            tool_name = first_line.replace('🔧 Calling ', '').strip()
                            
                            # Extract JSON arguments from the rest
                            json_start = content.find('{')
                            if json_start != -1:
                                json_str = content[json_start:]
                                tool_args = json.loads(json_str)
                        except (json.JSONDecodeError, IndexError, ValueError):
                            # If parsing fails, skip this message
                            pass
                
                if tool_name:
                    # Format action matching tasks.json structure
                    action = {
                        "action_id": f"{len(grpo_records)}_{action_counter}",
                        "name": tool_name,
                        "arguments": tool_args if tool_args else {},
                        "info": None
                    }
                    actions.append(action)
                    action_counter += 1
        
        print(f"[GRPO] Trajectory {idx} ({session_id[:8] if session_id else 'N/A'}...): reward={reward}, {len(actions)} tool calls")
        
        # Skip trajectories without any actions (no function calls)
        if not actions:
            print(f"[GRPO] Skipping trajectory {idx} - no actions ground truth (no function calls)")
            continue
        
        # Extract user info from persona_data if available
        user_info = persona_data.get('user', {}) if persona_data else {}
        user_name = user_info.get('name', {})
        first_name = user_name.get('first_name', '')
        last_name = user_name.get('last_name', '')
        user_address = user_info.get('address', {})
        zip_code = user_address.get('zip', '')
        email = user_info.get('email', '')
        
        # Build known_info string
        known_info_parts = []
        if first_name and last_name:
            known_info_parts.append(f"You are {first_name} {last_name}")
            if zip_code:
                known_info_parts.append(f"in zip code {zip_code}")
        known_info = " ".join(known_info_parts) + "." if known_info_parts else ""
        
        # Build unknown_info (email is commonly "forgotten")
        unknown_info = "You do not remember your email address." if email else ""
        
        # Extract communicate_info from reward_info if available
        communicate_info = []
        reward_info = trajectory.get('reward_info', {})
        if reward_info and 'outputs' in reward_info:
            outputs = reward_info.get('outputs', {})
            communicate_info = list(outputs.keys()) if isinstance(outputs, dict) else []
        
        # Create GRPO record matching tasks.json format
        record_id = str(len(grpo_records))
        grpo_record = {
            "id": record_id,
            "description": {
                "purpose": f"Generated from trajectory {session_id[:8] if session_id else 'unknown'}",
                "relevant_policies": None,
                "notes": f"Reward: {reward}" if reward is not None else None
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": persona or ".",
                    "domain": env_name or "retail",
                    "reason_for_call": task_instruction or "",
                    "known_info": known_info,
                    "unknown_info": unknown_info
                }
            },
            "initial_state": None,
            "evaluation_criteria": {
                "actions": actions,  # Ground truth: actual tool calls from trajectory
                "communicate_info": communicate_info,
                "nl_assertions": None
            }
        }
        
        grpo_records.append(grpo_record)
    
    skipped_count = len(trajectories) - len(grpo_records)
    print(f"[GRPO] Summary: {len(grpo_records)} records created, {skipped_count} skipped (no actions ground truth)")
    
    return grpo_records


def _convert_to_sft_format(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trajectories to SFT (Supervised Fine-Tuning) training format.
    
    SFT format:
    - task_id: identifier for the task
    - conversations: list of messages up to the decision point (with role, content, tool_calls)
    - answer: list containing the single correct next action
    - rubric: evaluation criteria (optional)
    - reject_rubric: criteria for rejected actions (optional)
    - reject_answer_raw: raw thinking/reasoning for rejected action (if available)
    - chosen_answer_raw: raw thinking/reasoning for chosen action (if available)
    
    One-to-Many Expansion:
    Each trajectory generates multiple SFT records - one for each assistant turn.
    For example, a trajectory with 4 assistant turns generates 4 SFT records,
    each with progressively longer conversation history.
    
    If the trajectory has rejected suggestions, the record at that point will include
    both reject_answer_raw and chosen_answer_raw for preference learning.
    """
    sft_records = []
    
    print(f"[SFT] Processing {len(trajectories)} trajectories")
    
    for traj_idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        task_instruction = trajectory.get('task_instruction', '')
        wiki = trajectory.get('wiki', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', '')
        task_id = trajectory.get('task_id', traj_idx)
        
        # Build rejection map: index of rejection -> rejected data
        rejection_map = {}
        for i, msg in enumerate(messages):
            if msg.get('role') == 'rejected':
                rejection_map[i] = msg.get('rejected', {})
        
        print(f"[SFT] Trajectory {traj_idx} ({session_id[:8] if session_id else 'N/A'}...): {len(messages)} messages, {len(rejection_map)} rejected")
        
        # Always do one-to-many expansion: create one record per assistant turn
        sft_records_for_traj = _create_sft_records_for_trajectory(
            messages=messages,
            task_id=task_id,
            wiki=wiki,
            rejection_map=rejection_map
        )
        sft_records.extend(sft_records_for_traj)
    
    print(f"[SFT] Summary: {len(sft_records)} records created")
    
    return sft_records


def _create_sft_records_for_trajectory(
    messages: List[Dict[str, Any]],
    task_id: Any,
    wiki: str,
    rejection_map: Optional[Dict[int, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create SFT records for a trajectory with one-to-many expansion.
    
    Creates a record at each assistant action point (tool call or response).
    If rejection_map is provided, includes rejected reasoning at relevant points.
    
    Args:
        messages: List of trajectory messages
        task_id: Task identifier
        wiki: Policy/wiki content
        rejection_map: Optional dict mapping rejection indices to rejected data
    """
    sft_records = []
    rejection_map = rejection_map or {}
    
    # Start with initial greeting
    base_conversations = [{
        "role": "assistant",
        "content": "Hi! How can I help you today?",
        "tool_calls": None
    }]
    
    conversations = list(base_conversations)
    
    # Track if the previous message was a rejection (the current assistant turn is the "chosen" action)
    pending_rejection = None
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        
        # Check if this is a rejection marker
        if role == 'rejected':
            # Store the rejection data - the next assistant action will be the "chosen" action
            pending_rejection = msg.get('rejected', {})
            continue
        
        if role in ['agent', 'tool']:
            # Check if this is an action point (agent response or tool call)
            converted = _convert_message_for_sft(msg)
            if converted:
                # Create SFT record with conversation up to this point as context
                # and the current action as the answer
                sft_record = {
                    "task_id": task_id,
                    "conversations": list(conversations),
                    "answer": [converted],
                    "rubric": "",
                    "reject_rubric": ""
                }
                
                # Add chosen reasoning if available
                reasoning = msg.get('reasoning', '')
                if reasoning:
                    sft_record["chosen_answer_raw"] = reasoning
                
                # If there was a pending rejection, add the rejected reasoning
                if pending_rejection:
                    rejected_reasoning = pending_rejection.get('reasoning', '')
                    if rejected_reasoning:
                        sft_record["reject_answer_raw"] = rejected_reasoning
                    pending_rejection = None  # Clear the pending rejection
                
                sft_records.append(sft_record)
                
                # Add this message to the conversation for subsequent records
                conversations.append(converted)
        elif role in ['user', 'tool-result']:
            # Add to conversation history
            converted = _convert_message_for_sft(msg)
            if converted:
                conversations.append(converted)
    
    return sft_records


def _convert_message_for_sft(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a trajectory message to SFT format."""
    role = msg.get('role', '')
    
    if role == 'user':
        return {
            "role": "user",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    elif role == 'agent':
        result = {
            "role": "assistant",
            "content": msg.get('content'),
            "tool_calls": None
        }
        
        # Check if this agent message has tool_calls
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        if tool_name:
            # Format arguments properly
            if tool_arguments is None:
                args = {}
            elif isinstance(tool_arguments, str):
                try:
                    args = json.loads(tool_arguments)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = tool_arguments
            
            result["tool_calls"] = [{
                "id": f"chatcmpl-tool-{msg.get('id', '')}",
                "name": tool_name,
                "arguments": args,
                "requestor": "assistant"
            }]
        return result
    elif role == 'tool':
        # Tool call from agent - represents assistant making a tool call
        tool_name = msg.get('tool_name')
        tool_arguments = msg.get('tool_arguments')
        
        # If tool_name is not set, try to parse from content
        if not tool_name:
            content = msg.get('content', '')
            parsed = _parse_tool_call_from_content(content)
            if parsed:
                tool_name, tool_arguments = parsed
        
        if not tool_name:
            return None
        
        # Format arguments properly
        if tool_arguments is None:
            args = {}
        elif isinstance(tool_arguments, str):
            try:
                args = json.loads(tool_arguments)
            except json.JSONDecodeError:
                args = {}
        else:
            args = tool_arguments
        
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"chatcmpl-tool-{msg.get('id', '')}",
                "name": tool_name,
                "arguments": args,
                "requestor": "assistant"
            }]
        }
    elif role == 'tool-result':
        # Tool response - goes back to the model
        return {
            "role": "tool",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    elif role == 'system':
        return {
            "role": "system",
            "content": msg.get('content', ''),
            "tool_calls": None
        }
    
    return None


# =============================================================================
# WebSocket for Real-time Communication
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


ws_manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time simulation."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await ws_manager.connect(session_id, websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action_type = data.get("type")
            
            if action_type == "start":
                # Start the simulation
                initial_message = simulator.start()
                await websocket.send_json({
                    "type": "user_message",
                    "content": initial_message,
                    "persona": simulator.current_persona,
                })
            
            elif action_type == "respond":
                # Send response to user
                message = data.get("message", "")
                result = simulator.respond_to_user(message)
                
                await websocket.send_json({
                    "type": "agent_message",
                    "content": message,
                })
                
                if result.done:
                    await websocket.send_json({
                        "type": "simulation_end",
                        "result": simulator.get_final_result(),
                    })
                else:
                    await websocket.send_json({
                        "type": "user_message",
                        "content": result.observation,
                    })
            
            elif action_type == "tool_call":
                # Call a tool
                tool_name = data.get("tool_name")
                arguments = data.get("arguments", {})
                
                await websocket.send_json({
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                })
                
                result = simulator.call_tool(tool_name, arguments)
                
                await websocket.send_json({
                    "type": "tool_result",
                    "content": result.observation,
                })
                
                if result.done:
                    await websocket.send_json({
                        "type": "simulation_end",
                        "result": simulator.get_final_result(),
                    })
            
            elif action_type == "generate_response":
                # Generate response using LLM
                prompt = data.get("prompt", "")
                response = simulator.generate_response(prompt)
                
                await websocket.send_json({
                    "type": "generated_response",
                    "content": response,
                })
            
            elif action_type == "parse_action":
                # Parse natural language action
                user_input = data.get("user_input", "")
                parsed = simulator.parse_natural_language_action(user_input)
                
                await websocket.send_json({
                    "type": "parsed_action",
                    "action": parsed,
                })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
        ws_manager.disconnect(session_id)


# =============================================================================
# SPA Catch-All Route (must be last to not interfere with API routes)
# =============================================================================

@app.get("/trajectories/{trajectory_id}/simulation")
async def trajectory_simulation_page(trajectory_id: str):
    """Serve the SPA for trajectory simulation pages."""
    react_index = REACT_DIST_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Page not available</h1>")


@app.get("/trajectory")
async def trajectory_page():
    """Serve the SPA for trajectory list page."""
    react_index = REACT_DIST_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Page not available</h1>")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="tau-bench Simulator API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create static directory if it doesn't exist
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        "sigma.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
