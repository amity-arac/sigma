import './ActionSuggestion.css'

function ActionSuggestion({ action, onApprove, onReject }) {
  const getHeaderTitle = () => {
    if (action.action_type === 'respond') {
      return '📝 Suggested Response'
    }
    return `🔧 Suggested Tool Call: ${action.tool_name}`
  }

  return (
    <div className="action-suggestion pending">
      <div className="action-suggestion-header">
        <h4>{getHeaderTitle()}</h4>
      </div>
      
      <div className="action-suggestion-body">
        {action.reasoning && (
          <div className="action-suggestion-section">
            <div className="action-suggestion-label reasoning">💭 Reasoning</div>
            <div className="action-suggestion-content reasoning">
              {action.reasoning}
            </div>
          </div>
        )}
        
        {action.action_type === 'respond' ? (
          <div className="action-suggestion-section">
            <div className="action-suggestion-label response">Response to User</div>
            <div className="action-suggestion-content">
              {action.content}
            </div>
          </div>
        ) : (
          <div className="action-suggestion-section">
            <div className="action-suggestion-label tool">Arguments</div>
            <div className="action-suggestion-content code">
              {JSON.stringify(action.arguments, null, 2)}
            </div>
          </div>
        )}
      </div>
      
      <div className="action-suggestion-buttons">
        <button className="btn btn-reject" onClick={onReject}>
          ✕ Reject
        </button>
        <button className="btn btn-approve" onClick={onApprove}>
          ✓ Approve & Execute
        </button>
      </div>
    </div>
  )
}

export default ActionSuggestion
