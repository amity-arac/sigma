import { useState } from 'react'
import ToolResultContent from './ToolResultContent'
import './Message.css'

const ICONS = {
  user: '👤',
  agent: '🤖',
  tool: '🔧',
  'tool-result': '📋',
  reasoning: '💭'
}

const ROLE_NAMES = {
  user: 'User',
  agent: 'Agent (You)',
  tool: 'Tool Call',
  'tool-result': 'Tool Result',
  reasoning: 'Agent Reasoning'
}

function Message({ 
  role, 
  content, 
  reasoning, 
  isTemporary, 
  messageIndex,
  onRollback,
  onRegenerateUser,
  isSimulationActive
}) {
  const [showReasoning, setShowReasoning] = useState(false)

  const toggleReasoning = () => {
    setShowReasoning(prev => !prev)
  }

  // Determine if this message should show rollback button (only agent response and tool call)
  const showRollbackButton = isSimulationActive && 
    (role === 'agent' || role === 'tool') && 
    onRollback && 
    messageIndex !== undefined

  // Determine if this message should show refresh button (user messages)
  const showRefreshButton = isSimulationActive && 
    role === 'user' && 
    onRegenerateUser && 
    messageIndex !== undefined

  return (
    <div className={`message ${role} ${isTemporary ? 'temporary' : ''}`}>
      <div className="message-header">
        <span className="message-icon">{ICONS[role]}</span>
        <span className="message-role">{ROLE_NAMES[role]}</span>
        {reasoning && (
          <button 
            className={`reasoning-toggle ${showReasoning ? 'active' : ''}`}
            onClick={toggleReasoning}
            title={showReasoning ? 'Hide reasoning' : 'Show reasoning'}
          >
            💭 {showReasoning ? 'Hide' : 'Show'} Reasoning
          </button>
        )}
        <div className="message-actions">
          {showRefreshButton && (
            <button 
              className="message-action-btn refresh-btn"
              onClick={() => onRegenerateUser(messageIndex)}
              title="Regenerate this user response with additional guidance"
            >
              🔄
            </button>
          )}
          {showRollbackButton && (
            <button 
              className="message-action-btn rollback-btn"
              onClick={() => onRollback(messageIndex)}
              title="Rollback conversation to before this message"
            >
              ✕
            </button>
          )}
        </div>
      </div>
      
      {reasoning && showReasoning && (
        <div className="message-reasoning">
          <div className="reasoning-label">💭 Agent Reasoning</div>
          <div className="reasoning-content">{reasoning}</div>
        </div>
      )}
      
      <div className="message-content">
        {role === 'tool-result' ? (
          <ToolResultContent content={content} />
        ) : (
          content
        )}
      </div>
    </div>
  )
}

export default Message
