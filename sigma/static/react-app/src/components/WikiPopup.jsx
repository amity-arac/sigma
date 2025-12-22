import { useMemo } from 'react'
import './WikiPopup.css'

/**
 * WikiPopup - Shows wiki content in a popup while waiting for session to start
 * Supports both HTML (rendered in iframe) and Markdown content types
 * Shows a "Begin Session" button that lights up when session is ready
 */
function WikiPopup({ content, contentType, loading, envName, sessionReady, generatedScenario, onBeginSession, onClose }) {
  // Simple markdown to HTML conversion (only for markdown content)
  const renderedMarkdown = useMemo(() => {
    if (!content || contentType !== 'markdown') return null
    
    // Basic markdown rendering
    let html = content
      // Headers
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/_(.+?)_/g, '<em>$1</em>')
      // Code blocks
      .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Unordered lists
      .replace(/^\s*[-*+]\s+(.*)$/gim, '<li>$1</li>')
      // Ordered lists  
      .replace(/^\s*\d+\.\s+(.*)$/gim, '<li>$1</li>')
      // Line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br/>')
    
    // Wrap in paragraph if needed
    if (!html.startsWith('<')) {
      html = '<p>' + html + '</p>'
    }
    
    return html
  }, [content, contentType])

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  // Render content based on type
  const renderContent = () => {
    // Don't show loading for wiki - it's just static files and loads quickly
    // The footer already shows "Initializing session..." for the actual work
    if (!content) {
      return (
        <div className="wiki-popup-placeholder">
          <p>Loading documentation...</p>
        </div>
      )
    }
    
    // For HTML content, use an iframe for proper style isolation
    if (contentType === 'html') {
      return (
        <iframe
          className="wiki-popup-iframe"
          srcDoc={content}
          title={`Wiki - ${envName}`}
          sandbox="allow-scripts allow-same-origin"
        />
      )
    }
    
    // For markdown, render as HTML
    return (
      <div 
        className="wiki-popup-body"
        dangerouslySetInnerHTML={{ __html: renderedMarkdown }}
      />
    )
  }

  return (
    <div className="wiki-popup-overlay" onClick={handleOverlayClick}>
      <div className={`wiki-popup ${contentType === 'html' ? 'wiki-popup-fullsize' : ''}`}>
        <div className="wiki-popup-header">
          <div className="wiki-popup-title-section">
            <span className="wiki-popup-icon">👋</span>
            <span className="wiki-popup-title">Welcome! Please review the agent policy before starting</span>
          </div>
          <button className="wiki-popup-close" onClick={onClose} title="Close">
            ✕
          </button>
        </div>
        
        <div className={`wiki-popup-content ${contentType === 'html' ? 'wiki-popup-content-iframe' : ''}`}>
          {renderContent()}
        </div>
        
        {/* Show generated scenario inspiration if available */}
        {generatedScenario && (
          <div className="wiki-popup-inspiration">
            <div className="inspiration-header">
              <span className="inspiration-icon">💡</span>
              <span className="inspiration-title">Generated Scenario</span>
            </div>
            <div className="inspiration-content">
              <div className="inspiration-label">Inspired by task:</div>
              <div className="inspiration-text">{generatedScenario.seed_task_instruction}</div>
            </div>
          </div>
        )}

        <div className="wiki-popup-footer">
          <div className="wiki-popup-status">
            {sessionReady ? (
              <span className="status-ready">✓ Trajectory Ready</span>
            ) : (
              <span className="status-loading">
                <span className="mini-spinner"></span>
                Initializing trajectory...
              </span>
            )}
          </div>
          <div className="wiki-popup-actions">
            <button 
              className="btn btn-secondary" 
              onClick={onClose}
            >
              Cancel
            </button>
            <button 
              className={`btn btn-begin ${sessionReady ? 'ready pulse' : ''}`}
              onClick={onBeginSession}
              disabled={!sessionReady}
            >
              {sessionReady ? '🚀 Begin Trajectory' : 'Waiting...'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WikiPopup
