const API_BASE = window.location.origin

export async function fetchEnvironments() {
  const response = await fetch(`${API_BASE}/environments`)
  if (!response.ok) {
    throw new Error('Failed to load environments')
  }
  return response.json()
}

export async function fetchEnvironmentWiki(envName) {
  const response = await fetch(`${API_BASE}/environments/${envName}/wiki`)
  if (!response.ok) {
    throw new Error('Failed to load environment wiki')
  }
  return response.json()
}

export async function createSession(payload) {
  // Use the new trajectory-centric API
  const response = await fetch(`${API_BASE}/trajectories`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      env_name: payload.env_name || payload.envName || 'retail',
      user_model: payload.user_model || payload.userModel || 'gpt-4o',
      user_provider: payload.user_provider || payload.userProvider || 'openai',
      agent_model: payload.agent_model || payload.agentModel || null,
      agent_provider: payload.agent_provider || payload.agentProvider || null,
      persona: payload.persona || null,
      task_index: payload.task_index || payload.taskIndex || null,
      task_split: payload.task_split || payload.taskSplit || 'test',
      generate_scenario: payload.generate_scenario ?? payload.generateScenario ?? true,
      task_ids: payload.task_ids || payload.taskIds || null,
    })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to create trajectory')
  }
  const result = await response.json()
  // Map the response to match the old session API format
  return {
    session_id: result.trajectory_id,  // Use trajectory_id as session_id
    ...result
  }
}

// createTrajectory is the same as createSession
export const createTrajectory = createSession

// startSession is no longer needed - trajectories are started on creation
// Keep for backward compatibility but return the same data
export async function startSession(trajectoryId) {
  // The trajectory is already started when created
  // Just load and return the trajectory data
  try {
    const trajectories = await listTrajectories(null, 500)
    const traj = (trajectories.trajectories || []).find(t => t.id === trajectoryId)
    if (!traj) {
      throw new Error('Trajectory not found')
    }
    const fullTrajectory = await getTrajectory(trajectoryId, traj.env_name)
    
    // Get first user message
    let initialMessage = ''
    for (const msg of (fullTrajectory.messages || [])) {
      if (msg.role === 'user') {
        initialMessage = msg.content || ''
        break
      }
    }
    
    return {
      session_id: trajectoryId,
      initial_message: initialMessage,
      persona: fullTrajectory.persona || '',
      tools: [],  // Would need to load from simulator
      wiki: fullTrajectory.wiki || '',
      generated_scenario: null,
    }
  } catch (e) {
    // Fallback to old endpoint for truly old sessions
    const response = await fetch(`${API_BASE}/sessions/${trajectoryId}/start`, {
      method: 'POST'
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to start simulation')
    }
    return response.json()
  }
}

// Alias for consistency
export const startTrajectory = startSession

export async function continueTrajectory(trajectoryId, envName, options = {}) {
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}/continue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trajectory_id: trajectoryId,
      env_name: envName,
      user_model: options.userModel || 'gpt-4o',
      user_provider: options.userProvider || 'openai',
      agent_model: options.agentModel || null,
      agent_provider: options.agentProvider || null
    })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to continue trajectory')
  }
  return response.json()
}

export async function parseAction(sessionId, userInput) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/parse-action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_input: userInput })
  })
  if (!response.ok) {
    throw new Error('Could not understand your request. Please try again.')
  }
  return response.json()
}

export async function sendResponse(sessionId, message) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  })
  if (!response.ok) {
    throw new Error('Failed to send response')
  }
  return response.json()
}

export async function callTool(sessionId, toolName, args) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/tool`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool_name: toolName, arguments: args })
  })
  if (!response.ok) {
    throw new Error('Failed to call tool')
  }
  return response.json()
}

export async function undoAction(sessionId) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/undo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  })
  if (!response.ok) {
    throw new Error('Failed to undo action')
  }
  return response.json()
}

export async function rollbackToPoint(sessionId, messageId) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message_id: String(messageId) })
  })
  if (!response.ok) {
    throw new Error('Failed to rollback')
  }
  return response.json()
}

export async function regenerateUserResponse(sessionId, rejectedMessage = null, feedback = null) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/regenerate-user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      rejected_message: rejectedMessage,
      feedback: feedback 
    })
  })
  if (!response.ok) {
    throw new Error('Failed to regenerate user response')
  }
  return response.json()
}

export async function generateResponse(sessionId, prompt) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/generate-response`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  })
  if (!response.ok) {
    throw new Error('Failed to generate response')
  }
  return response.json()
}

export async function regenerateAction(sessionId, rejectedAction, feedback = null) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/regenerate-action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      rejected_action: rejectedAction,
      feedback: feedback 
    })
  })
  if (!response.ok) {
    throw new Error('Failed to regenerate action')
  }
  return response.json()
}

/**
 * Check if a proposed agent action complies with the policy.
 * Used by auto-approve feature to automatically approve compliant actions.
 * 
 * @param {string} sessionId - The session ID
 * @param {object} action - The action to check (action_type, content, tool_name, arguments, reasoning)
 * @returns {Promise<{approved: boolean, confidence: string, reason: string, policy_concerns: string[]}>}
 */
export async function checkPolicyCompliance(sessionId, action) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/check-policy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action })
  })
  if (!response.ok) {
    throw new Error('Failed to check policy compliance')
  }
  return response.json()
}

export async function checkTrajectoryStorageStatus() {
  const response = await fetch(`${API_BASE}/trajectory/status`)
  if (!response.ok) {
    throw new Error('Failed to check trajectory storage status')
  }
  return response.json()
}

export async function saveTrajectory(trajectoryId, messages, resultInfo = {}) {
  // Use the new trajectory-centric endpoint: PUT /trajectories/{id}/messages
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}/messages`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: messages.map(m => ({
        id: String(m.id),
        role: m.role,
        content: m.content,
        reasoning: m.reasoning || null,
        timestamp: m.timestamp || null,
        tool_name: m.tool_name || null,
        tool_arguments: m.tool_arguments || null,
        rejected: m.rejected || null,
      })),
      is_done: resultInfo.is_done,
      reward: resultInfo.reward,
      reward_info: resultInfo.reward_info
    })
  })
  
  if (!response.ok) {
    // Fallback to old session endpoint for compatibility
    const fallbackResponse = await fetch(`${API_BASE}/sessions/${trajectoryId}/save-trajectory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trajectory_id: trajectoryId,
        messages: messages.map(m => ({
          id: String(m.id),
          role: m.role,
          content: m.content,
          reasoning: m.reasoning || null,
          timestamp: m.timestamp || null,
          tool_name: m.tool_name || null,
          tool_arguments: m.tool_arguments || null,
          rejected: m.rejected || null,
        })),
        is_done: resultInfo.is_done,
        reward: resultInfo.reward,
        reward_info: resultInfo.reward_info
      })
    })
    
    if (!fallbackResponse.ok) {
      const error = await fallbackResponse.json()
      throw new Error(error.detail || 'Failed to save trajectory')
    }
    return fallbackResponse.json()
  }
  
  return response.json()
}

export async function listTrajectories(envName = null, limit = 100) {
  const params = new URLSearchParams()
  if (envName) params.append('env_name', envName)
  if (limit) params.append('limit', limit)
  
  const response = await fetch(`${API_BASE}/trajectories?${params}`)
  if (!response.ok) {
    throw new Error('Failed to list trajectories')
  }
  return response.json()
}

export async function getTrajectory(trajectoryId, envName) {
  const params = new URLSearchParams({ env_name: envName })
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}?${params}`)
  if (!response.ok) {
    throw new Error('Failed to get trajectory')
  }
  return response.json()
}

export async function deleteTrajectory(trajectoryId, envName) {
  const params = new URLSearchParams({ env_name: envName })
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}?${params}`, {
    method: 'DELETE'
  })
  if (!response.ok) {
    throw new Error('Failed to delete trajectory')
  }
  return response.json()
}

export async function updateTrajectory(trajectoryId, envName, updates) {
  const params = new URLSearchParams({ env_name: envName })
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}?${params}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates)
  })
  if (!response.ok) {
    throw new Error('Failed to update trajectory')
  }
  return response.json()
}

export async function editTrajectoryMessage(trajectoryId, messageId, newContent) {
  const response = await fetch(`${API_BASE}/trajectories/${trajectoryId}/messages/${messageId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message_id: messageId,
      content: newContent
    })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to edit message')
  }
  return response.json()
}

export async function exportTrajectories(format, envName = null, trajectoryIds = null, dateFilter = null) {
  const response = await fetch(`${API_BASE}/trajectories/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      format: format,
      env_name: envName || null,
      trajectory_ids: trajectoryIds || null,
      date_filter: dateFilter || null
    })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to export trajectories')
  }
  return response.json()
}

// Get trajectory by ID, searching across all environments
export async function getTrajectoryByIdAnyEnv(trajectoryId) {
  // First try to get trajectory list to find the env
  const response = await fetch(`${API_BASE}/trajectories?limit=500`)
  if (!response.ok) {
    throw new Error('Failed to fetch trajectories')
  }
  const result = await response.json()
  
  // Find the trajectory in the list
  const trajectory = (result.trajectories || []).find(t => t.id === trajectoryId)
  if (!trajectory) {
    throw new Error('Trajectory not found')
  }
  
  // Now get the full trajectory with messages
  return getTrajectory(trajectoryId, trajectory.env_name)
}

// =============================================================================
// Environment Files API
// =============================================================================

export async function fetchEnvironmentFiles(envName) {
  const response = await fetch(`${API_BASE}/environments/${envName}/files`)
  if (!response.ok) {
    throw new Error('Failed to load environment files')
  }
  return response.json()
}

export async function fetchEnvironmentFile(envName, filename) {
  const response = await fetch(`${API_BASE}/environments/${envName}/files/${filename}`)
  if (!response.ok) {
    throw new Error('Failed to load file content')
  }
  return response.json()
}

export async function updateEnvironmentFile(envName, filename, content) {
  const response = await fetch(`${API_BASE}/environments/${envName}/files/${filename}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to save file')
  }
  return response.json()
}

export async function duplicateEnvironment(envName, newName) {
  const response = await fetch(`${API_BASE}/environments/${envName}/duplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_name: newName })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to duplicate environment')
  }
  return response.json()
}

export async function renameEnvironment(envName, newName) {
  const response = await fetch(`${API_BASE}/environments/${envName}/rename`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_name: newName })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to rename environment')
  }
  return response.json()
}

export async function deleteEnvironment(envName) {
  const response = await fetch(`${API_BASE}/environments/${envName}`, {
    method: 'DELETE'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to delete environment')
  }
  return response.json()
}
