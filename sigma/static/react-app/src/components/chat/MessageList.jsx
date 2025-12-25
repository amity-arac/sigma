import Message from './Message'
import './MessageList.css'

function MessageList({ messages, onRollback, onRegenerateUser, onEditMessage, onRemoveRejected, isSimulationActive, isConversationEnded, onNewSession, wasOriginallyCompleted }) {
  // Check if the last user message contains ###STOP### to determine if conversation has ended
  const lastUserMessage = [...messages].reverse().find(m => m.role === 'user')
  const hasStopSignal = lastUserMessage && lastUserMessage.content?.includes('###STOP###')
  
  // Show conversation ended only when:
  // 1. There are messages in the conversation, AND
  // 2. Either isConversationEnded is true OR the last user message contains ###STOP###
  const showConversationEnded = messages.length > 0 && (isConversationEnded || hasStopSignal)

  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <Message 
          key={message.id}
          messageId={message.id}
          role={message.role}
          content={message.content}
          reasoning={message.reasoning}
          rejected={message.rejected}
          isTemporary={message.isTemporary}
          messageIndex={index}
          onRollback={onRollback}
          onRegenerateUser={onRegenerateUser}
          onEditMessage={onEditMessage}
          onRemoveRejected={onRemoveRejected}
          isSimulationActive={isSimulationActive}
          wasOriginallyCompleted={wasOriginallyCompleted}
        />
      ))}
      
      {showConversationEnded && (
        <div className="conversation-ended">
          <div className="conversation-ended-message">
            <span className="ended-icon">✅</span>
            <span className="ended-text">Conversation has ended</span>
          </div>
          {onNewSession && (
            <button className="btn btn-new-trajectory" onClick={onNewSession}>
              ✨ Start New Trajectory
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default MessageList
