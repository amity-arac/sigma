import { useState } from 'react'
import './ChatInput.css'

function ChatInput({ onSend, onAutoGenerate, disabled }) {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (disabled) return
    
    if (message.trim()) {
      // If there's text, send the custom message
      onSend(message.trim())
      setMessage('')
    } else {
      // If no text, trigger auto-generate
      onAutoGenerate()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-input">
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe action or leave empty for auto..."
          disabled={disabled}
        />
        <button 
          type="submit" 
          className="btn btn-auto"
          disabled={disabled}
          title={message.trim() ? 'Generate action from your description' : 'Auto-generate action based on wiki/policy'}
        >
          🤖 <span className="btn-text">{message.trim() ? 'Go' : 'Auto'}</span>
        </button>
      </form>
    </div>
  )
}

export default ChatInput
