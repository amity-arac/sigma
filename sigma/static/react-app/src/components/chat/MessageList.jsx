import Message from './Message'
import './MessageList.css'

function MessageList({ messages, onRollback, onRegenerateUser, isSimulationActive }) {
  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <Message 
          key={message.id}
          role={message.role}
          content={message.content}
          reasoning={message.reasoning}
          isTemporary={message.isTemporary}
          messageIndex={index}
          onRollback={onRollback}
          onRegenerateUser={onRegenerateUser}
          isSimulationActive={isSimulationActive}
        />
      ))}
    </div>
  )
}

export default MessageList
