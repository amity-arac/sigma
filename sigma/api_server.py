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

import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from sigma.env_registry import get_environment_config, list_environments, refresh_environment_registry
from sigma.envs import (
    DATA_ENVS_PATH,
    EnvironmentInfo,
    EnvironmentFileInfo,
    EnvironmentFilesResponse,
    EnvironmentFileContentResponse,
    UpdateEnvironmentFileRequest,
    EDITABLE_ENV_FILES,
    list_env_files,
    get_env_file,
    update_env_file,
    duplicate_env,
    rename_env,
    delete_env,
)
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
from sigma.exports import (
    ExportTrajectoryRequest,
    ExportTrajectoryResponse,
    convert_trajectories,
)


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
    target_index: Optional[int] = None  # DEPRECATED: The index in conversation history to rollback to
    message_id: Optional[str] = None  # The message ID to rollback to (removes this message and all after)


class RegenerateUserRequest(BaseModel):
    """Request to regenerate user (simulated user agent) response with feedback."""
    rejected_message: Optional[str] = None  # The rejected user message content
    feedback: Optional[str] = None  # User feedback on what should be different


class RegenerateActionRequest(BaseModel):
    """Request to regenerate agent action with feedback on rejected action."""
    rejected_action: Dict[str, Any]  # The rejected action (action_type, content, tool_name, arguments, reasoning)
    feedback: Optional[str] = None  # User feedback on why the action was rejected


class CheckPolicyComplianceRequest(BaseModel):
    """Request to check if an action complies with policy."""
    action: Dict[str, Any]  # The action to check (action_type, content, tool_name, arguments, reasoning)


class PolicyComplianceResponse(BaseModel):
    """Response from policy compliance check."""
    approved: bool  # True if Policy AI confidently approves
    confidence: str  # "high", "medium", or "low"
    reason: str  # Explanation for the decision
    policy_concerns: List[str]  # Any specific policy concerns identified


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
    id: Optional[str] = None  # Unique message ID for idempotent operations


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
    persona_data: Optional[Dict[str, Any]] = None  # Full persona data including augmented_data (injected scenario data)


class UpdateTrajectoryRequest(BaseModel):
    """Request to update a trajectory."""
    is_done: Optional[bool] = None
    reward: Optional[float] = None


class EditMessageRequest(BaseModel):
    """Request to edit a single message content in a trajectory."""
    message_id: Union[str, float, int]  # The ID of the message to edit (accepts string or number)
    content: str  # The new content for the message


# ExportTrajectoryRequest and ExportTrajectoryResponse are imported from sigma.exports


class FinalResultResponse(BaseModel):
    """Response containing final simulation result."""
    session_id: str
    env_name: str
    is_done: bool
    reward: Optional[float]
    reward_info: Optional[Dict[str, Any]]
    expected_actions: Optional[List[Dict[str, Any]]]
    conversation_history: List[Dict[str, Any]]


# EnvironmentFileInfo, EnvironmentFilesResponse, EnvironmentFileContentResponse,
# UpdateEnvironmentFileRequest are imported from sigma.envs


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
ASSETS_DIR = STATIC_DIR / "assets"

# Mount React build assets directly (JS/CSS files in assets folder)
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount legacy static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Routes - General
# =============================================================================

@app.get("/")
async def root():
    """Serve the main HTML page."""
    # Serve from assets folder (React build output)
    assets_index = ASSETS_DIR / "index.html"
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
    assets_index = ASSETS_DIR / "index.html"
    if assets_index.exists():
        return FileResponse(str(assets_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Admin page not available</h1>")


@app.get("/env-config")
async def env_config_page():
    """Serve the environment configuration page (SPA routing - same index.html)."""
    assets_index = ASSETS_DIR / "index.html"
    if assets_index.exists():
        return FileResponse(str(assets_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Environment config page not available</h1>")


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

# EDITABLE_ENV_FILES is imported from sigma.envs


@app.get("/environments/{env_name}/files", response_model=EnvironmentFilesResponse)
async def get_environment_files(env_name: str):
    """Get list of editable files in an environment."""
    files, error = list_env_files(env_name)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return EnvironmentFilesResponse(env_name=env_name, files=files)


@app.get("/environments/{env_name}/files/{filename}", response_model=EnvironmentFileContentResponse)
async def get_environment_file_route(env_name: str, filename: str):
    """Get content of a specific environment file."""
    content, file_type, error = get_env_file(env_name, filename)
    if error:
        if "not editable" in error:
            raise HTTPException(status_code=400, detail=error)
        elif "not found" in error:
            raise HTTPException(status_code=404, detail=error)
        else:
            raise HTTPException(status_code=500, detail=error)
    
    return EnvironmentFileContentResponse(
        env_name=env_name,
        filename=filename,
        content=content,
        type=file_type
    )


@app.put("/environments/{env_name}/files/{filename}")
async def update_environment_file_route(env_name: str, filename: str, request: UpdateEnvironmentFileRequest):
    """Update content of an environment file."""
    success, error = update_env_file(env_name, filename, request.content)
    if not success:
        if "not editable" in error:
            raise HTTPException(status_code=400, detail=error)
        elif "not found" in error:
            raise HTTPException(status_code=404, detail=error)
        elif "Invalid JSON" in error:
            raise HTTPException(status_code=400, detail=error)
        else:
            raise HTTPException(status_code=500, detail=error)
    
    return {"success": True, "message": f"File '{filename}' updated successfully"}


class DuplicateEnvironmentRequest(BaseModel):
    """Request to duplicate an environment."""
    new_name: str


class RenameEnvironmentRequest(BaseModel):
    """Request to rename an environment."""
    new_name: str


@app.post("/environments/{env_name}/duplicate")
async def duplicate_environment_route(env_name: str, request: DuplicateEnvironmentRequest):
    """Duplicate an environment with a new name."""
    success, error = duplicate_env(env_name, request.new_name)
    if not success:
        if "not found" in error:
            raise HTTPException(status_code=404, detail=error)
        elif "already exists" in error:
            raise HTTPException(status_code=409, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)
    
    # Refresh environment registry to detect the new environment
    refresh_environment_registry()
    
    return {"success": True, "message": f"Environment '{env_name}' duplicated to '{request.new_name}'"}


@app.put("/environments/{env_name}/rename")
async def rename_environment_route(env_name: str, request: RenameEnvironmentRequest):
    """Rename an environment."""
    success, error = rename_env(env_name, request.new_name)
    if not success:
        if "not found" in error:
            raise HTTPException(status_code=404, detail=error)
        elif "already exists" in error:
            raise HTTPException(status_code=409, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)
    
    # Refresh environment registry to update the renamed environment
    refresh_environment_registry()
    
    return {"success": True, "message": f"Environment '{env_name}' renamed to '{request.new_name}'"}


@app.delete("/environments/{env_name}")
async def delete_environment_route(env_name: str):
    """Delete an environment."""
    success, error = delete_env(env_name)
    if not success:
        if "not found" in error:
            raise HTTPException(status_code=404, detail=error)
        elif "last remaining" in error:
            raise HTTPException(status_code=400, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)
    
    # Refresh environment registry to remove the deleted environment
    refresh_environment_registry()
    
    return {"success": True, "message": f"Environment '{env_name}' deleted"}


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
        # Run blocking LLM operation in thread pool
        initial_message = await asyncio.to_thread(simulator.start)
        
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
                "augmented_data": simulator.generated_scenario.augmented_data,
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
    
    # Run blocking LLM operation in thread pool
    result = await asyncio.to_thread(simulator.respond_to_user, request.message)
    
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
    
    # Run blocking operation in thread pool
    result = await asyncio.to_thread(simulator.call_tool, request.tool_name, request.arguments)
    
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
    
    # Run blocking LLM operation in thread pool
    response = await asyncio.to_thread(simulator.generate_response, request.prompt)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return {"response": response}


@app.post("/sessions/{session_id}/parse-action")
async def parse_action(session_id: str, request: ParseActionRequest):
    """Parse natural language into a structured action."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Run blocking LLM operation in thread pool
    parsed = await asyncio.to_thread(simulator.parse_natural_language_action, request.user_input)
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
    """Rollback conversation to a specific point (removing the target message and all after)."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Prefer message_id for idempotent operations, fall back to target_index for backwards compatibility
    if request.message_id:
        result = simulator.rollback_to_message_id(request.message_id)
    elif request.target_index is not None:
        result = simulator.rollback_to_index(request.target_index)
    else:
        return {
            "success": False,
            "error": "Either message_id or target_index must be provided",
            "removed_count": 0,
            "remaining_count": len(simulator.state.conversation_history)
        }
    return result


@app.post("/sessions/{session_id}/regenerate-user")
async def regenerate_user_response(session_id: str, request: RegenerateUserRequest):
    """Regenerate the simulated user's response with feedback on rejected message."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Run blocking LLM operation in thread pool
    result = await asyncio.to_thread(
        simulator.regenerate_user_response,
        request.rejected_message,
        request.feedback
    )
    return result


@app.post("/sessions/{session_id}/regenerate-action")
async def regenerate_action(session_id: str, request: RegenerateActionRequest):
    """Regenerate agent action with feedback on a rejected action."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Run blocking LLM operation in thread pool
    result = await asyncio.to_thread(
        simulator.regenerate_action_with_feedback,
        request.rejected_action,
        request.feedback
    )
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Could not regenerate action. Check server logs for details."
        )
    return result


@app.post("/sessions/{session_id}/check-policy", response_model=PolicyComplianceResponse)
async def check_policy_compliance(session_id: str, request: CheckPolicyComplianceRequest):
    """
    Check if a proposed agent action complies with the policy.
    
    Uses Policy AI to evaluate whether the action confidently follows policy.
    Returns approval if confident, or flags for human review if uncertain.
    
    This endpoint is used by the auto-approve feature to automatically approve
    actions that clearly comply with policy, while flagging uncertain cases
    for human review.
    """
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Run blocking LLM operation in thread pool
    result = await asyncio.to_thread(simulator.check_policy_compliance, request.action)
    return PolicyComplianceResponse(**result)


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
            id=e.id,
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


def _create_and_start_simulator(
    session_manager: SimulatorSessionManager,
    env_name: str,
    user_model: str,
    user_provider: str,
    agent_model: str = None,
    agent_provider: str = None,
    persona: str = None,
    task_index: int = None,
    task_split: str = "test",
    generate_scenario: bool = True,
    task_ids: List[int] = None,
):
    """Synchronous helper to create and start simulator (runs in thread pool)."""
    simulator = session_manager.create_session(
        env_name=env_name,
        user_model=user_model,
        user_provider=user_provider,
        agent_model=agent_model,
        agent_provider=agent_provider,
        persona=persona,
        task_index=task_index,
        task_split=task_split,
        generate_scenario=generate_scenario,
        task_ids=task_ids,
    )
    initial_message = simulator.start()
    return simulator, initial_message


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
        # Run blocking LLM operations in thread pool to avoid blocking event loop
        simulator, initial_message = await asyncio.to_thread(
            _create_and_start_simulator,
            session_manager,
            request.env_name,
            request.user_model,
            request.user_provider,
            request.agent_model,
            request.agent_provider,
            request.persona,
            request.task_index,
            request.task_split,
            request.generate_scenario,
            request.task_ids,
        )
        
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
        # Run blocking LLM operation in thread pool
        result = await asyncio.to_thread(simulator.send_response, request.message)
        
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
        # Run blocking operation in thread pool
        result = await asyncio.to_thread(simulator.call_tool, request.tool_name, request.arguments)
        
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
        # Run blocking LLM operation in thread pool
        action = await asyncio.to_thread(simulator.parse_action, request.user_input)
        
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
        
        # Prefer message_id for idempotent operations, fall back to target_index for backwards compatibility
        if request.message_id:
            result = simulator.rollback_to_message_id(request.message_id)
        elif request.target_index is not None:
            result = simulator.rollback_to_index(request.target_index)
        else:
            return {
                "success": False,
                "error": "Either message_id or target_index must be provided",
                "removed_count": 0,
                "new_length": len(simulator.state.conversation_history),
            }
        
        # Re-enable simulation if it was done
        if result.get("success") and simulator.state.is_done:
            simulator.state.is_done = False
            simulator.state.last_reward = None
            simulator.state.reward_info = None
        
        # Add new_length to response for backwards compatibility
        result["new_length"] = result.get("remaining_count", len(simulator.state.conversation_history))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/trajectories/{trajectory_id}/regenerate-user")
async def trajectory_regenerate_user(trajectory_id: str, request: RegenerateUserRequest):
    """
    Regenerate the last user response with feedback on rejected message.
    """
    try:
        simulator = get_simulator_for_trajectory(trajectory_id)
        # Run blocking LLM operation in thread pool
        new_response = await asyncio.to_thread(
            simulator.regenerate_user_response,
            request.rejected_message,
            request.feedback
        )
        
        return {
            "success": True,
            "observation": new_response,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.patch("/trajectories/{trajectory_id}/messages/{message_id}")
async def edit_trajectory_message(trajectory_id: str, message_id: str, request: EditMessageRequest):
    """
    Edit a single message's content in a trajectory.
    
    This allows editing any user or agent message content without rollback
    or any other side effects. Works even on completed trajectories.
    """
    try:
        storage = get_trajectory_storage()
        
        # Find the trajectory in any environment
        trajectory_dict = None
        env_name = None
        for env in list_environments():
            trajectory_dict = storage.get(trajectory_id, env)
            if trajectory_dict:
                env_name = env
                break
        
        if not trajectory_dict:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        
        # Convert message_id for comparison - handle both string and numeric IDs
        # Use the path parameter message_id as primary, fall back to request body
        target_message_id = message_id  # from path parameter (already a string)
        request_message_id = str(request.message_id) if request.message_id else target_message_id
        
        # Find and update the message
        messages = trajectory_dict.get('messages', [])
        message_found = False
        for msg in messages:
            msg_id = msg.get('id')
            msg_id_str = str(msg_id) if msg_id is not None else None
            
            # Try multiple matching approaches to handle float precision issues
            if (msg_id_str == target_message_id or 
                msg_id_str == request_message_id or
                msg_id == target_message_id or
                (isinstance(msg_id, (int, float)) and abs(float(msg_id) - float(target_message_id)) < 0.0001)):
                # Only allow editing user and agent messages
                if msg.get('role') not in ['user', 'agent']:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Cannot edit message with role '{msg.get('role')}'. Only user and agent messages can be edited."
                    )
                msg['content'] = request.content
                message_found = True
                break
        
        if not message_found:
            # Provide more helpful error message with available message IDs
            available_ids = [str(msg.get('id')) for msg in messages[:10]]  # First 10 IDs
            raise HTTPException(
                status_code=404, 
                detail=f"Message not found. Looking for '{target_message_id}'. Available IDs (first 10): {available_ids}"
            )
        
        # Save the updated messages using the update method
        success = storage.update(trajectory_id, env_name, {"messages": messages})
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update trajectory")
        
        return {
            "success": True,
            "message": "Message updated successfully",
            "trajectory_id": trajectory_id,
            "message_id": message_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def _create_from_trajectory_sync(
    session_manager: SimulatorSessionManager,
    trajectory: TrajectoryData,
    user_model: str,
    user_provider: str,
    agent_model: str = None,
    agent_provider: str = None,
):
    """Synchronous helper to create simulator from trajectory (runs in thread pool)."""
    simulator = session_manager.create_from_trajectory(
        trajectory=trajectory,
        user_model=user_model,
        user_provider=user_provider,
        agent_model=agent_model,
        agent_provider=agent_provider,
    )
    # Start the session if conversation is empty
    if not simulator.state.conversation_history:
        simulator.start()
    return simulator


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
        
        # Use the clean from_trajectory approach in thread pool to avoid blocking
        # This properly handles persona_data injection including augmented_data
        simulator = await asyncio.to_thread(
            _create_from_trajectory_sync,
            session_manager,
            trajectory,
            request.user_model,
            request.user_provider,
            request.agent_model,
            request.agent_provider,
        )
        
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
            persona_data=trajectory.persona_data,
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
        
        # Convert using the exports module
        try:
            result = convert_trajectories(request.format, full_trajectories)
            print(f"[Export] {request.format.upper()} conversion result: {len(result)} records")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
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
    assets_index = ASSETS_DIR / "index.html"
    if assets_index.exists():
        return FileResponse(str(assets_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Page not available</h1>")


@app.get("/trajectory")
async def trajectory_page():
    """Serve the SPA for trajectory list page."""
    assets_index = ASSETS_DIR / "index.html"
    if assets_index.exists():
        return FileResponse(str(assets_index))
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Page not available</h1>")


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
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (use 1 with --reload)")
    
    args = parser.parse_args()
    
    # Create static directory if it doesn't exist
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Note: workers > 1 is incompatible with --reload
    # For development, use --reload with 1 worker
    # For production, use --workers 4 (or more) without --reload
    uvicorn.run(
        "sigma.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
    )


if __name__ == "__main__":
    main()
