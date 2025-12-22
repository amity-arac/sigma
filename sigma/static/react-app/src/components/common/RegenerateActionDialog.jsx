import { useState } from 'react'
import './RegenerateActionDialog.css'

function RegenerateActionDialog({ rejectedAction, onConfirm, onCancel }) {
  const [feedback, setFeedback] = useState('')

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onCancel()
    }
  }

  const handleConfirm = () => {
    onConfirm(feedback.trim() || null)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleConfirm()
    } else if (e.key === 'Escape') {
      onCancel()
    }
  }

  const isToolCall = rejectedAction?.action_type === 'tool_call'

  return (
    <div className="regenerate-overlay" onClick={handleOverlayClick}>
      <div className="regenerate-dialog">
        <h3>🔄 Regenerate Response</h3>
        
        <div className="rejected-action-preview">
          <div className="rejected-label">Rejected {isToolCall ? 'Tool Call' : 'Response'}:</div>
          {rejectedAction?.reasoning && (
            <div className="rejected-section">
              <span className="section-label">💭 Reasoning:</span>
              <div className="section-content reasoning">{rejectedAction.reasoning}</div>
            </div>
          )}
          {isToolCall ? (
            <>
              <div className="rejected-section">
                <span className="section-label">🔧 Tool:</span>
                <div className="section-content tool-name">{rejectedAction.tool_name}</div>
              </div>
              <div className="rejected-section">
                <span className="section-label">Arguments:</span>
                <div className="section-content code">
                  {JSON.stringify(rejectedAction.arguments, null, 2)}
                </div>
              </div>
            </>
          ) : (
            <div className="rejected-section">
              <span className="section-label">📝 Response:</span>
              <div className="section-content">{rejectedAction?.content}</div>
            </div>
          )}
        </div>

        <div className="feedback-section">
          <label htmlFor="feedback-input">What should be different? (optional)</label>
          <textarea
            id="feedback-input"
            className="feedback-input"
            placeholder="e.g., 'Be more formal', 'Don't use that tool', 'Ask for confirmation first', etc."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            autoFocus
          />
          <span className="hint">Press Ctrl+Enter to regenerate</span>
        </div>

        <div className="regenerate-dialog-buttons">
          <button className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={handleConfirm}>
            🔄 Regenerate
          </button>
        </div>
      </div>
    </div>
  )
}

export default RegenerateActionDialog
