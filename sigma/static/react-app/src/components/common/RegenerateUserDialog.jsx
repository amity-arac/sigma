import { useState } from 'react'
import './RegenerateActionDialog.css'  // Reuse the same styles

function RegenerateUserDialog({ rejectedMessage, onConfirm, onCancel }) {
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

  return (
    <div className="regenerate-overlay" onClick={handleOverlayClick}>
      <div className="regenerate-dialog">
        <h3>🔄 Regenerate User Response</h3>
        
        <div className="rejected-action-preview">
          <div className="rejected-label">Current User Message:</div>
          <div className="rejected-section">
            <div className="section-content">{rejectedMessage}</div>
          </div>
        </div>

        <div className="feedback-section">
          <label htmlFor="feedback-input">What should be different? (optional)</label>
          <textarea
            id="feedback-input"
            className="feedback-input"
            placeholder="e.g., 'be more frustrated', 'ask about a different issue', 'be more polite', etc."
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

export default RegenerateUserDialog
