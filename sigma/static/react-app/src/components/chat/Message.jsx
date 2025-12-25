import { useState } from 'react'
import ToolResultContent from './ToolResultContent'
import './Message.css'

const ICONS = {
  user: '👤',
  agent: '🤖',
  tool: '🔧',
  'tool-result': '📋',
  reasoning: '💭',
  rejected: '❌'
}

const ROLE_NAMES = {
  user: 'User',
  agent: 'Agent (You)',
  tool: 'Tool Call',
  'tool-result': 'Tool Result',
  reasoning: 'Agent Reasoning',
  rejected: 'Rejected'
}

function Message({ 
  role, 
  content, 
  reasoning,
  rejected,
  isTemporary, 
  messageIndex,
  messageId,
  onRollback,
  onRegenerateUser,
  onEditMessage,
  onRemoveRejected,
  isSimulationActive,
  wasOriginallyCompleted
}) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [showRejectedDetails, setShowRejectedDetails] = useState(false)

  const toggleReasoning = () => {
    setShowReasoning(prev => !prev)
  }

  const toggleRejectedDetails = () => {
    setShowRejectedDetails(prev => !prev)
  }

  // Determine if this message should show rollback/regenerate button (agent response and tool call)
  // Allow regeneration on completed trajectories too - the handler will show a warning
  const showRollbackButton = (isSimulationActive || wasOriginallyCompleted) && 
    (role === 'agent' || role === 'tool') && 
    onRollback && 
    messageIndex !== undefined

  // Determine if this message should show refresh button (user messages)
  // Allow regeneration on completed trajectories too - the handler will show a warning
  const showRefreshButton = (isSimulationActive || wasOriginallyCompleted) && 
    role === 'user' && 
    onRegenerateUser && 
    messageIndex !== undefined

  // Determine if this message should show edit button (user and agent messages only, not tool/tool-result)
  // Edit is allowed even on completed conversations
  const showEditButton = (role === 'user' || role === 'agent') && 
    onEditMessage && 
    messageIndex !== undefined

  // Determine if this is a rejected message that can be removed
  const isRejectedMessage = role === 'rejected'
  const showRemoveButton = isRejectedMessage && onRemoveRejected && messageId

  // For rejected messages, get the type
  const rejectedType = rejected?.tool_name ? 'Tool Call' : 'Response'

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
        {isRejectedMessage && rejected && (
          <button 
            className={`reasoning-toggle ${showRejectedDetails ? 'active' : ''}`}
            onClick={toggleRejectedDetails}
            title={showRejectedDetails ? 'Hide details' : 'Show details'}
          >
            📋 {showRejectedDetails ? 'Hide' : 'Show'} Details
          </button>
        )}
        <div className="message-actions">
          {showEditButton && (
            <button 
              className="message-action-btn edit-btn"
              onClick={() => onEditMessage(messageIndex, role, content)}
              title="Edit this message"
            >
              ✏️
            </button>
          )}
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
              title="Regenerate from this point"
            >
              🔄
            </button>
          )}
          {showRemoveButton && (
            <button 
              className="message-action-btn remove-btn"
              onClick={() => onRemoveRejected(messageId)}
              title="Remove this rejected action from trajectory"
            >
              🗑️
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

      {/* Rejected message details (collapsible) */}
      {isRejectedMessage && rejected && showRejectedDetails && (
        <div className="rejected-details-panel">
          <div className="rejected-type-label">Type: {rejectedType}</div>
          {rejected.reasoning && (
            <div className="rejected-detail-section">
              <div className="rejected-detail-label">💭 Reasoning:</div>
              <div className="rejected-detail-content reasoning">{rejected.reasoning}</div>
            </div>
          )}
          {rejected.content && (
            <div className="rejected-detail-section">
              <div className="rejected-detail-label">📝 Response:</div>
              <div className="rejected-detail-content">{rejected.content}</div>
            </div>
          )}
          {rejected.tool_name && (
            <div className="rejected-detail-section">
              <div className="rejected-detail-label">🔧 Tool:</div>
              <div className="rejected-detail-content code">{rejected.tool_name}</div>
            </div>
          )}
          {rejected.tool_arguments && (
            <div className="rejected-detail-section">
              <div className="rejected-detail-label">Arguments:</div>
              <div className="rejected-detail-content code">
                {JSON.stringify(rejected.tool_arguments, null, 2)}
              </div>
            </div>
          )}
        </div>
      )}
      
      <div className="message-content">
        {role === 'tool-result' ? (
          <ToolResultContent content={content} />
        ) : isRejectedMessage ? (
          <span className="rejected-summary">
            Rejected {rejectedType}
            {rejected && !showRejectedDetails && <span className="hint"> (click "Show Details" to see more)</span>}
          </span>
        ) : (
          content
        )}
      </div>
    </div>
  )
}

export default Message
