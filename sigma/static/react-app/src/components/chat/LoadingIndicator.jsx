import './LoadingIndicator.css'

function LoadingIndicator({ text = 'Processing', variant = 'default' }) {
  return (
    <div className={`loading-message ${variant}`}>
      <div className="loading-spinner"></div>
      <span className="loading-text">
        {text}<span className="loading-dots"></span>
      </span>
    </div>
  )
}

export default LoadingIndicator
