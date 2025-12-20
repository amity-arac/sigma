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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sigma.simulator_core import (
    SimulatorCore,
    SimulatorSessionManager,
    get_available_environments,
    load_persona_file,
)
from sigma.env_registry import get_environment_config, list_environments
from sigma.trajectory_storage import (
    TrajectoryStorage,
    TrajectoryData,
    TrajectoryMessage,
    get_trajectory_storage,
    get_configured_backend,
    check_storage_configuration,
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


class ToolInfoResponse(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]


# =============================================================================
# Application Setup
# =============================================================================

# Session manager (global state)
session_manager = SimulatorSessionManager()


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
    
    Looks for files in the sigma/envs/{env_name}/ folder only:
    1. policy.html (preferred for rich formatting)
    2. policy.md (standard format)
    """
    try:
        import os
        
        # Only look within sigma/envs folder
        env_path = os.path.join(os.path.dirname(__file__), "envs", env_name)
        
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
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        storage = get_trajectory_storage()
        
        # Build trajectory messages from request
        from sigma.trajectory_storage import RejectedSuggestion
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
        
        # Build trajectory data
        trajectory = TrajectoryData(
            session_id=session_id,
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
            messages=messages,
            is_done=request.is_done if request.is_done is not None else simulator.state.is_done,
            reward=request.reward if request.reward is not None else simulator.state.last_reward,
            reward_info=request.reward_info or simulator.state.reward_info,
            expected_actions=simulator.state.expected_actions,
        )
        
        # Save to storage
        trajectory_id = storage.save(trajectory)
        
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
    Convert trajectories to GRPO training format.
    
    GRPO (Group Relative Policy Optimization) format matches tau_bench tasks.py:
    - id: task identifier
    - user_id: the user involved in the task
    - instruction: user persona and task instruction
    - actions: sequence of tool calls (ground truth)
    - outputs: expected output values (optional)
    
    This extracts the sequence of tool calls made during the trajectory
    to serve as ground truth for verifiable rewards.
    """
    grpo_records = []
    
    print(f"[GRPO] Processing {len(trajectories)} trajectories")
    
    for idx, trajectory in enumerate(trajectories):
        messages = trajectory.get('messages', [])
        task_instruction = trajectory.get('task_instruction', '')
        user_id = trajectory.get('user_id', '')
        env_name = trajectory.get('env_name', '')
        session_id = trajectory.get('session_id', '')
        reward = trajectory.get('reward', 0)
        
        # Extract tool call sequence from the actual trajectory
        actions = []
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
                    # Format action like tasks.py
                    action = {
                        "name": tool_name,
                        "arguments": tool_args if tool_args else {}
                    }
                    actions.append(action)
        
        print(f"[GRPO] Trajectory {idx} ({session_id[:8] if session_id else 'N/A'}...): reward={reward}, {len(actions)} tool calls")
        
        # Skip trajectories without any actions (no function calls)
        if not actions:
            print(f"[GRPO] Skipping trajectory {idx} - no actions ground truth (no function calls)")
            continue
        
        # Create GRPO record
        grpo_record = {
            "id": len(grpo_records),  # Use sequential id based on actual records created
            "user_id": user_id or f"user_{session_id[:8] if session_id else 'unknown'}",
            "instruction": task_instruction or f"Task from session {session_id}",
            "actions": actions,
        }
        
        # Add expected_actions if available (from original task)
        expected_actions = trajectory.get('expected_actions')
        if expected_actions:
            grpo_record["expected_actions"] = expected_actions
        
        # Add reward info if available
        reward_info = trajectory.get('reward_info')
        if reward_info and 'outputs' in reward_info:
            grpo_record["outputs"] = list(reward_info['outputs'].keys())
        
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
