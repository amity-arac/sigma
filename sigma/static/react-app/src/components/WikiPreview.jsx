import { useMemo } from 'react'
import './WikiPreview.css'

/**
 * WikiPreview - Shows wiki content while waiting for session to start
 * Supports both HTML and Markdown content types
 */
function WikiPreview({ content, contentType, loading, envName }) {
  // Simple markdown to HTML conversion
  const renderedContent = useMemo(() => {
    if (!content) return null
    
    if (contentType === 'html') {
      return content
    }
    
    if (contentType === 'markdown') {
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
    }
    
    return content
  }, [content, contentType])

  if (loading) {
    return (
      <div className="wiki-preview">
        <div className="wiki-preview-header">
          <span className="wiki-preview-icon">📖</span>
          <span className="wiki-preview-title">Loading Wiki for {envName}...</span>
        </div>
        <div className="wiki-preview-loading">
          <div className="wiki-loading-spinner"></div>
          <span>Fetching documentation...</span>
        </div>
      </div>
    )
  }

  if (!content) {
    return (
      <div className="wiki-preview">
        <div className="wiki-preview-header">
          <span className="wiki-preview-icon">📖</span>
          <span className="wiki-preview-title">Initializing Session...</span>
        </div>
        <div className="wiki-preview-placeholder">
          <p>Setting up the simulation environment. Please wait...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="wiki-preview">
      <div className="wiki-preview-header">
        <span className="wiki-preview-icon">📖</span>
        <span className="wiki-preview-title">Policy / Wiki - {envName}</span>
        <span className="wiki-preview-subtitle">(Review while session starts)</span>
      </div>
      <div 
        className="wiki-preview-content"
        dangerouslySetInnerHTML={{ __html: renderedContent }}
      />
    </div>
  )
}

export default WikiPreview
