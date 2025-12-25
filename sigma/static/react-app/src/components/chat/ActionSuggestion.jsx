import { useState } from 'react'
import './ActionSuggestion.css'

function ActionSuggestion({ action, onApprove, onReject, policyResult, isCheckingPolicy = false, mode = 'pending' }) {
  const [showApprovalDetails, setShowApprovalDetails] = useState(false)
  
  const getHeaderTitle = () => {
    if (isCheckingPolicy) {
      return '🔍 Checking Policy Compliance...'
    }
    if (policyResult && !policyResult.approved) {
      return '🛡️ Policy Review Required'
    }
    if (action.action_type === 'respond') {
      return '📝 Suggested Response'
    }
    return '🔧 Suggested Tool Call'
  }

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'high':
        return '#4caf50'
      case 'medium':
        return '#ff9800'
      case 'low':
        return '#f44336'
      default:
        return '#9e9e9e'
    }
  }

  const getConfidenceEmoji = (confidence) => {
    switch (confidence) {
      case 'high':
        return '✅'
      case 'medium':
        return '⚠️'
      case 'low':
        return '❌'
      default:
        return '❓'
    }
  }

  const hasPolicyConcerns = policyResult && !policyResult.approved

  return (
    <div className={`action-suggestion ${mode} ${hasPolicyConcerns ? 'needs-review' : ''} ${isCheckingPolicy ? 'checking-policy' : ''}`}>
      <div className={`action-suggestion-header ${hasPolicyConcerns ? 'review-header' : ''} ${isCheckingPolicy ? 'checking-header' : ''}`}>
        <h4>{getHeaderTitle()}</h4>
        {policyResult && policyResult.approved && (
          <button 
            className="approval-details-btn"
            onClick={() => setShowApprovalDetails(!showApprovalDetails)}
            title="View approval details"
          >
            {showApprovalDetails ? '▼' : '▶'} AI Approved
          </button>
        )}
      </div>
      
      {/* Show approval details popup when clicked */}
      {showApprovalDetails && policyResult && policyResult.approved && (
        <div className="approval-details-panel">
          <div className="approval-details-row">
            <span className="approval-label">Confidence:</span>
            <span className="approval-value" style={{ color: getConfidenceColor(policyResult.confidence) }}>
              {getConfidenceEmoji(policyResult.confidence)} {policyResult.confidence?.toUpperCase()}
            </span>
          </div>
          <div className="approval-details-row">
            <span className="approval-label">Reason:</span>
            <span className="approval-value">{policyResult.reason}</span>
          </div>
          {policyResult.analysis && (
            <div className="approval-details-row analysis">
              <span className="approval-label">Analysis:</span>
              <span className="approval-value">{policyResult.analysis}</span>
            </div>
          )}
        </div>
      )}
      
      {/* Policy concerns section for flagged actions */}
      {hasPolicyConcerns && (
        <div className="policy-concerns-section">
          <div className="policy-assessment">
            <div className="confidence-row">
              <span className="confidence-label">AI Confidence:</span>
              <span className="confidence-value" style={{ color: getConfidenceColor(policyResult.confidence) }}>
                {getConfidenceEmoji(policyResult.confidence)} {policyResult.confidence?.toUpperCase()}
              </span>
            </div>
            <div className="policy-reason">
              <strong>Reason:</strong> {policyResult.reason}
            </div>
          </div>
          
          {policyResult.policy_concerns && policyResult.policy_concerns.length > 0 && (
            <div className="concerns-list">
              <strong>Concerns:</strong>
              <ul>
                {policyResult.policy_concerns.map((concern, index) => (
                  <li key={index}>{concern}</li>
                ))}
              </ul>
            </div>
          )}
          
          {policyResult.analysis && (
            <details className="analysis-details">
              <summary>View detailed analysis</summary>
              <p>{policyResult.analysis}</p>
            </details>
          )}
        </div>
      )}
      
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
          <>
            <div className="action-suggestion-section">
              <div className="action-suggestion-label tool">🔧 Tool Name</div>
              <div className="action-suggestion-content tool-name">
                {action.tool_name}
              </div>
            </div>
            <div className="action-suggestion-section">
              <div className="action-suggestion-label tool">Arguments</div>
              <div className="action-suggestion-content code">
                {JSON.stringify(action.arguments, null, 2)}
              </div>
            </div>
          </>
        )}
      </div>
      
      <div className={`action-suggestion-buttons ${isCheckingPolicy ? 'checking' : ''}`}>
        {isCheckingPolicy ? (
          <div className="policy-check-status">
            <div className="policy-spinner"></div>
            <span>Policy AI is reviewing this action...</span>
          </div>
        ) : (
          <>
            <button className="btn btn-reject" onClick={onReject}>
              ✕ {hasPolicyConcerns ? 'Reject & Regenerate' : 'Reject'}
            </button>
            <button className="btn btn-approve" onClick={onApprove}>
              ✓ {hasPolicyConcerns ? 'Approve Anyway' : 'Approve & Execute'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export default ActionSuggestion
