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
  const response = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to create session')
  }
  return response.json()
}

export async function startSession(sessionId) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/start`, {
    method: 'POST'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to start simulation')
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

export async function rollbackToPoint(sessionId, targetIndex) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_index: targetIndex })
  })
  if (!response.ok) {
    throw new Error('Failed to rollback')
  }
  return response.json()
}

export async function regenerateUserResponse(sessionId, additionalNote = null) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/regenerate-user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ additional_note: additionalNote })
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

export async function checkTrajectoryStorageStatus() {
  const response = await fetch(`${API_BASE}/trajectory/status`)
  if (!response.ok) {
    throw new Error('Failed to check trajectory storage status')
  }
  return response.json()
}

export async function saveTrajectory(sessionId, messages, resultInfo = {}) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/save-trajectory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      // Messages now include rejected suggestions inline (role='rejected')
      messages: messages.map(m => ({
        id: String(m.id),
        role: m.role,
        content: m.content,
        reasoning: m.reasoning || null,
        timestamp: m.timestamp || null,
        // For tool calls
        tool_name: m.tool_name || null,
        tool_arguments: m.tool_arguments || null,
        // For rejected suggestions (role='rejected') - same format as normal message
        rejected: m.rejected || null,
      })),
      is_done: resultInfo.is_done,
      reward: resultInfo.reward,
      reward_info: resultInfo.reward_info
    })
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to save trajectory')
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
