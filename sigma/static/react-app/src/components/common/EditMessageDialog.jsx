import { useState, useEffect } from 'react'
import './RegenerateActionDialog.css'  // Reuse the same styles

function EditMessageDialog({ messageContent, messageRole, onConfirm, onCancel }) {
  const [editedContent, setEditedContent] = useState(messageContent || '')

  // Reset content when dialog opens with new message
  useEffect(() => {
    setEditedContent(messageContent || '')
  }, [messageContent])

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onCancel()
    }
  }

  const handleConfirm = () => {
    const trimmed = editedContent.trim()
    if (trimmed && trimmed !== messageContent) {
      onConfirm(trimmed)
    } else if (trimmed === messageContent) {
      onCancel() // No changes made
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleConfirm()
    } else if (e.key === 'Escape') {
      onCancel()
    }
  }

  const roleLabel = messageRole === 'user' ? 'User' : 'Agent'
  const roleIcon = messageRole === 'user' ? '👤' : '🤖'

  return (
    <div className="regenerate-overlay" onClick={handleOverlayClick}>
      <div className="regenerate-dialog edit-message-dialog">
        <h3>✏️ Edit {roleLabel} Message</h3>
        
        <div className="feedback-section">
          <label htmlFor="edit-input">{roleIcon} {roleLabel} Message:</label>
          <textarea
            id="edit-input"
            className="feedback-input edit-content-input"
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={6}
            autoFocus
          />
          <span className="hint">Press Ctrl+Enter to save, Escape to cancel</span>
        </div>

        <div className="regenerate-dialog-buttons">
          <button className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button 
            className="btn btn-primary" 
            onClick={handleConfirm}
            disabled={!editedContent.trim() || editedContent.trim() === messageContent}
          >
            ✓ Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}

export default EditMessageDialog
