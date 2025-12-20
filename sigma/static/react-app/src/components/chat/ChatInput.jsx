import { useState } from 'react'
import './ChatInput.css'

function ChatInput({ onSend, onAutoGenerate, disabled }) {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !disabled) {
      onSend(message.trim())
      setMessage('')
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
          placeholder="Describe any action (respond, call tool, etc.)..."
          disabled={disabled}
        />
        <button 
          type="submit" 
          className="btn btn-success"
          disabled={disabled || !message.trim()}
        >
          Generate Action
        </button>
        <button 
          type="button"
          className="btn btn-auto"
          onClick={onAutoGenerate}
          disabled={disabled}
          title="Auto-generate action based on wiki/policy"
        >
          🤖 Auto
        </button>
      </form>
    </div>
  )
}

export default ChatInput
