import { useState, useMemo } from 'react'
import './ToolResultContent.css'

function ToolResultContent({ content }) {
  const [showRaw, setShowRaw] = useState(false)
  
  const { parsed, isJson } = useMemo(() => {
    try {
      return { parsed: JSON.parse(content), isJson: true }
    } catch {
      return { parsed: null, isJson: false }
    }
  }, [content])

  if (!isJson) {
    return <div className="tool-result-raw">{content}</div>
  }

  const getValueClass = (value) => {
    if (typeof value === 'boolean') {
      return value ? 'success' : 'error'
    }
    if (value === null) {
      return 'error'
    }
    return ''
  }

  const formatValue = (value) => {
    if (typeof value === 'boolean') {
      return value ? '✓ true' : '✗ false'
    }
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2)
    }
    if (value === null) {
      return 'null'
    }
    return String(value)
  }

  return (
    <div className="tool-result-container">
      <div className="tool-result-formatted">
        {Object.entries(parsed).map(([key, value]) => (
          <div key={key} className="tool-result-item">
            <span className="tool-result-key">{key}:</span>
            <span className={`tool-result-value ${getValueClass(value)}`}>
              {formatValue(value)}
            </span>
          </div>
        ))}
      </div>
      
      <div 
        className="tool-result-toggle"
        onClick={() => setShowRaw(!showRaw)}
      >
        {showRaw ? '▼ Hide raw JSON' : '▶ Show raw JSON'}
      </div>
      
      {showRaw && (
        <div className="tool-result-raw">
          {JSON.stringify(parsed, null, 2)}
        </div>
      )}
    </div>
  )
}

export default ToolResultContent
