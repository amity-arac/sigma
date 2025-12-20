import './LoadingIndicator.css'

function LoadingIndicator({ text = 'Processing' }) {
  return (
    <div className="loading-message">
      <div className="loading-spinner"></div>
      <span className="loading-text">
        {text}<span className="loading-dots"></span>
      </span>
    </div>
  )
}

export default LoadingIndicator
