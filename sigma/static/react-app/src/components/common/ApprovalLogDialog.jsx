import React from 'react'
import './ApprovalLogDialog.css'

const ApprovalLogDialog = ({ logs, onClose }) => {
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

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A'
    const date = new Date(timestamp)
    return date.toLocaleTimeString()
  }

  const formatAction = (action) => {
    if (!action) return 'Unknown action'
    if (action.action_type === 'respond') {
      const content = action.content || ''
      return `Text: "${content.substring(0, 80)}${content.length > 80 ? '...' : ''}"`
    } else if (action.action_type === 'tool_call') {
      return `Tool: ${action.tool_name}(${JSON.stringify(action.arguments || {}).substring(0, 50)}...)`
    }
    return JSON.stringify(action).substring(0, 80)
  }

  return (
    <div className="approval-log-overlay" onClick={onClose}>
      <div className="approval-log-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="approval-log-header">
          <h3>🛡️ Approval AI Log</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="approval-log-info">
          <p>
            The Approval AI (powered by <strong>gpt-5.2</strong>) evaluates each agent action 
            against the environment policy and approval guidelines.
          </p>
        </div>
        
        {logs.length === 0 ? (
          <div className="approval-log-empty">
            <p>No approval checks yet.</p>
            <p className="hint">Enable Auto-Approve with Autopilot to see approval logs here.</p>
          </div>
        ) : (
          <div className="approval-log-list">
            {logs.map((log, index) => (
              <div 
                key={index} 
                className={`approval-log-entry ${log.approved ? 'approved' : 'flagged'}`}
              >
                <div className="log-entry-header">
                  <span className="log-number">#{logs.length - index}</span>
                  <span className="log-time">{formatTimestamp(log.timestamp)}</span>
                  <span 
                    className="log-confidence"
                    style={{ color: getConfidenceColor(log.confidence) }}
                  >
                    {getConfidenceEmoji(log.confidence)} {log.confidence?.toUpperCase()}
                  </span>
                  <span className={`log-status ${log.approved ? 'approved' : 'flagged'}`}>
                    {log.approved ? '✓ Auto-Approved' : '⏸ Flagged for Review'}
                  </span>
                </div>
                
                <div className="log-entry-action">
                  <strong>Action:</strong> {formatAction(log.action_checked)}
                </div>
                
                <div className="log-entry-reason">
                  <strong>Decision:</strong> {log.reason}
                </div>
                
                {log.policy_concerns && log.policy_concerns.length > 0 && (
                  <div className="log-entry-concerns">
                    <strong>Policy Concerns:</strong>
                    <ul>
                      {log.policy_concerns.map((concern, i) => (
                        <li key={i}>{concern}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {log.analysis && (
                  <details className="log-entry-analysis">
                    <summary>View Detailed Analysis</summary>
                    <div className="analysis-content">
                      {log.analysis}
                    </div>
                  </details>
                )}
                
                <div className="log-entry-model">
                  Model: {log.model_used || 'gpt-5.2'}
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="approval-log-footer">
          <button className="btn-close" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}

export default ApprovalLogDialog
