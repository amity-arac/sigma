import { useState } from 'react'
import './PromptDialog.css'

function PromptDialog({ title, message, placeholder, onConfirm, onCancel, allowEmpty = false }) {
  const [inputValue, setInputValue] = useState('')

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onCancel()
    }
  }

  const handleConfirm = () => {
    onConfirm(inputValue.trim() || null)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (allowEmpty || inputValue.trim())) {
      handleConfirm()
    } else if (e.key === 'Escape') {
      onCancel()
    }
  }

  return (
    <div className="prompt-overlay" onClick={handleOverlayClick}>
      <div className="prompt-dialog">
        <h3>{title}</h3>
        <p>{message}</p>
        <input
          type="text"
          className="prompt-input"
          placeholder={placeholder}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          autoFocus
        />
        <div className="prompt-dialog-buttons">
          <button className="btn btn-danger" onClick={onCancel}>
            Cancel
          </button>
          <button 
            className="btn btn-success" 
            onClick={handleConfirm}
            disabled={!allowEmpty && !inputValue.trim()}
          >
            {allowEmpty ? 'Regenerate' : 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default PromptDialog
