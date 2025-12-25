import React from 'react'
import './PolicyReviewDialog.css'

const PolicyReviewDialog = ({ policyResult, action, onApprove, onReject }) => {
  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'high':
        return '#4caf50'  // green
      case 'medium':
        return '#ff9800'  // orange
      case 'low':
        return '#f44336'  // red
      default:
        return '#9e9e9e'  // gray
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

  const formatAction = (action) => {
    if (action.action_type === 'text_response') {
      return `Text Response: "${action.content?.substring(0, 100)}${action.content?.length > 100 ? '...' : ''}"`
    } else if (action.action_type === 'tool_call') {
      return `Tool Call: ${action.tool_name}(${JSON.stringify(action.arguments || {})})`
    }
    return JSON.stringify(action)
  }

  return (
    <div className="policy-review-overlay">
      <div className="policy-review-dialog">
        <h3>🛡️ Policy Review Required</h3>
        
        <div className="policy-review-section">
          <h4>Action to Review</h4>
          <div className="action-preview">
            {formatAction(action)}
          </div>
          {action.reasoning && (
            <div className="action-reasoning">
              <strong>Reasoning:</strong> {action.reasoning}
            </div>
          )}
        </div>

        <div className="policy-review-section">
          <h4>Policy AI Assessment</h4>
          <div className="confidence-indicator" style={{ color: getConfidenceColor(policyResult.confidence) }}>
            {getConfidenceEmoji(policyResult.confidence)} Confidence: <strong>{policyResult.confidence?.toUpperCase()}</strong>
          </div>
          <div className={`approval-status ${policyResult.approved ? 'approved' : 'not-approved'}`}>
            {policyResult.approved ? '✓ Would Approve' : '✗ Would Not Approve'}
          </div>
        </div>

        <div className="policy-review-section">
          <h4>Reason</h4>
          <p className="policy-reason">{policyResult.reason || 'No reason provided'}</p>
        </div>

        {policyResult.policy_concerns && policyResult.policy_concerns.length > 0 && (
          <div className="policy-review-section">
            <h4>Policy Concerns</h4>
            <ul className="policy-concerns-list">
              {policyResult.policy_concerns.map((concern, index) => (
                <li key={index}>{concern}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="policy-review-note">
          <strong>Note:</strong> Auto-approve only triggers on HIGH confidence approvals. 
          This action requires human review because the Policy AI's confidence is not high enough.
        </div>

        <div className="policy-review-actions">
          <button className="policy-btn policy-btn-reject" onClick={onReject}>
            Reject & Regenerate
          </button>
          <button className="policy-btn policy-btn-approve" onClick={onApprove}>
            Approve Anyway
          </button>
        </div>
      </div>
    </div>
  )
}

export default PolicyReviewDialog
