import './StickyUserMessage.css'

function StickyUserMessage({ content }) {
  return (
    <div className="sticky-user-message visible">
      <div className="sticky-label">👤 User Persona:</div>
      <div className="sticky-content">{content}</div>
    </div>
  )
}

export default StickyUserMessage
