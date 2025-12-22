import { useState } from 'react'
import './TrajectoryViewer.css'

const ICONS = {
  user: '👤',
  agent: '🤖',
  tool: '🔧',
  'tool-result': '📋',
  system: '⚙️',
  rejected: '❌'
}

const ROLE_NAMES = {
  user: 'User',
  agent: 'Agent',
  tool: 'Tool Call',
  'tool-result': 'Tool Result',
  system: 'System',
  rejected: 'Rejected Suggestion'
}

function TrajectoryViewer({ trajectory, onClose, onDelete, onMarkComplete, onSimulate }) {
  const [activeTab, setActiveTab] = useState('conversation')
  const [expandedMessages, setExpandedMessages] = useState(new Set())

  const toggleMessage = (id) => {
    setExpandedMessages(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    try {
      // If the timestamp doesn't have timezone info, assume UTC
      let normalizedStr = dateStr
      if (!dateStr.endsWith('Z') && !dateStr.includes('+') && !dateStr.includes('-', 10)) {
        normalizedStr = dateStr + 'Z'
      }
      const date = new Date(normalizedStr)
      // Check for Invalid Date
      if (isNaN(date.getTime())) {
        return dateStr
      }
      return date.toLocaleString()
    } catch {
      return dateStr
    }
  }

  const formatJSON = (obj) => {
    if (!obj) return 'null'
    try {
      return JSON.stringify(obj, null, 2)
    } catch {
      return String(obj)
    }
  }

  const renderMessage = (message, index) => {
    const isExpanded = expandedMessages.has(message.id)
    const hasDetails = message.reasoning || message.tool_arguments || 
                       (message.rejected && (message.rejected.reasoning || message.rejected.tool_arguments))

    return (
      <div 
        key={message.id || index} 
        className={`viewer-message ${message.role}`}
      >
        <div 
          className="viewer-message-header"
          onClick={() => hasDetails && toggleMessage(message.id)}
          style={{ cursor: hasDetails ? 'pointer' : 'default' }}
        >
          <span className="message-icon">{ICONS[message.role] || '💬'}</span>
          <span className="message-role">{ROLE_NAMES[message.role] || message.role}</span>
          {message.tool_name && (
            <span className="tool-name-badge">{message.tool_name}</span>
          )}
          {message.timestamp && (
            <span className="message-timestamp">{formatDate(message.timestamp)}</span>
          )}
          {hasDetails && (
            <span className="expand-indicator">{isExpanded ? '▼' : '▶'}</span>
          )}
        </div>

        {/* Main content */}
        {message.content && (
          <div className="viewer-message-content">
            {message.content}
          </div>
        )}

        {/* Tool arguments */}
        {message.tool_arguments && (
          <div className="viewer-message-details">
            <div className="details-label">Arguments:</div>
            <pre className="details-code">{formatJSON(message.tool_arguments)}</pre>
          </div>
        )}

        {/* Expanded details */}
        {isExpanded && (
          <div className="viewer-message-expanded">
            {message.reasoning && (
              <div className="reasoning-section">
                <div className="reasoning-label">💭 Reasoning</div>
                <div className="reasoning-content">{message.reasoning}</div>
              </div>
            )}

            {/* Rejected suggestion details - using nested rejected object */}
            {message.role === 'rejected' && message.rejected && (
              <div className="rejected-details">
                <div className="rejected-type">
                  Type: {message.rejected.tool_name ? 'tool_call' : 'respond'}
                </div>
                {message.rejected.content && (
                  <div className="rejected-content">
                    <div className="details-label">Rejected Content:</div>
                    <div className="details-value">{message.rejected.content}</div>
                  </div>
                )}
                {message.rejected.tool_name && (
                  <div className="rejected-tool">
                    <div className="details-label">Rejected Tool: {message.rejected.tool_name}</div>
                    {message.rejected.tool_arguments && (
                      <pre className="details-code">{formatJSON(message.rejected.tool_arguments)}</pre>
                    )}
                  </div>
                )}
                {message.rejected.reasoning && (
                  <div className="rejected-reasoning">
                    <div className="details-label">Rejected Reasoning:</div>
                    <div className="details-value">{message.rejected.reasoning}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="trajectory-viewer-overlay" onClick={onClose}>
      <div className="trajectory-viewer" onClick={(e) => e.stopPropagation()}>
        <div className="viewer-header">
          <h2>📜 Trajectory Details</h2>
          <button className="close-button" onClick={onClose}>✕</button>
        </div>

        <div className="viewer-tabs">
          <button 
            className={`tab ${activeTab === 'conversation' ? 'active' : ''}`}
            onClick={() => setActiveTab('conversation')}
          >
            💬 Conversation
          </button>
          <button 
            className={`tab ${activeTab === 'metadata' ? 'active' : ''}`}
            onClick={() => setActiveTab('metadata')}
          >
            📋 Metadata
          </button>
          <button 
            className={`tab ${activeTab === 'task' ? 'active' : ''}`}
            onClick={() => setActiveTab('task')}
          >
            📝 Task Info
          </button>
        </div>

        <div className="viewer-content">
          {activeTab === 'conversation' && (
            <div className="conversation-tab">
              <div className="conversation-stats">
                <span>{trajectory.messages?.length || 0} messages</span>
                <span className="separator">•</span>
                <span className={`reward ${trajectory.reward >= 1 ? 'success' : 'partial'}`}>
                  Reward: {trajectory.reward?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="messages-container">
                {trajectory.messages?.map((msg, idx) => renderMessage(msg, idx))}
              </div>
            </div>
          )}

          {activeTab === 'metadata' && (
            <div className="metadata-tab">
              <div className="metadata-grid">
                <div className="metadata-item">
                  <label>Trajectory ID</label>
                  <span className="mono">{trajectory.id}</span>
                </div>
                <div className="metadata-item">
                  <label>Session ID</label>
                  <span className="mono">{trajectory.session_id}</span>
                </div>
                <div className="metadata-item">
                  <label>Environment</label>
                  <span>{trajectory.env_name}</span>
                </div>
                <div className="metadata-item">
                  <label>Created At</label>
                  <span>{formatDate(trajectory.created_at)}</span>
                </div>
                <div className="metadata-item">
                  <label>User Model</label>
                  <span>{trajectory.user_model} ({trajectory.user_provider})</span>
                </div>
                <div className="metadata-item">
                  <label>Agent Model</label>
                  <span>{trajectory.agent_model || 'N/A'} ({trajectory.agent_provider || 'N/A'})</span>
                </div>
                <div className="metadata-item">
                  <label>Status</label>
                  <span className={trajectory.is_done ? 'status-done' : 'status-incomplete'}>
                    {trajectory.is_done ? '✓ Complete' : '○ Incomplete'}
                  </span>
                </div>
                <div className="metadata-item">
                  <label>Reward</label>
                  <span className={`reward ${trajectory.reward >= 1 ? 'success' : 'partial'}`}>
                    {trajectory.reward?.toFixed(2) || 'N/A'}
                  </span>
                </div>
              </div>
              
              {trajectory.reward_info && (
                <div className="metadata-section">
                  <h4>Reward Info</h4>
                  <pre className="metadata-json">{formatJSON(trajectory.reward_info)}</pre>
                </div>
              )}
            </div>
          )}

          {activeTab === 'task' && (
            <div className="task-tab">
              <div className="metadata-grid">
                <div className="metadata-item">
                  <label>Task Index</label>
                  <span>{trajectory.task_index ?? 'N/A'}</span>
                </div>
                <div className="metadata-item">
                  <label>Task Split</label>
                  <span>{trajectory.task_split || 'N/A'}</span>
                </div>
              </div>
              
              {trajectory.task_instruction && (
                <div className="task-instruction-section">
                  <h4>Task Instruction</h4>
                  <div className="task-instruction">{trajectory.task_instruction}</div>
                </div>
              )}
              
              {trajectory.persona && (
                <div className="persona-section">
                  <h4>Persona</h4>
                  <div className="persona-content">{trajectory.persona}</div>
                </div>
              )}
              
              {trajectory.expected_actions && trajectory.expected_actions.length > 0 && (
                <div className="expected-actions-section">
                  <h4>Expected Actions</h4>
                  <pre className="metadata-json">{formatJSON(trajectory.expected_actions)}</pre>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="viewer-footer">
          {onSimulate && (
            <button className="simulate-button" onClick={onSimulate}>
              ▶️ Continue Simulation
            </button>
          )}
          <button className="delete-button" onClick={onDelete}>
            🗑️ Delete Trajectory
          </button>
          {!trajectory.is_done && onMarkComplete && (
            <button className="complete-button" onClick={onMarkComplete}>
              ✓ Mark as Complete
            </button>
          )}
          <button className="close-footer-button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

export default TrajectoryViewer
