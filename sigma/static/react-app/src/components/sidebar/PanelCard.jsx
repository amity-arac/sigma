import { useState } from 'react'
import './PanelCard.css'

function PanelCard({ title, children, defaultExpanded = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  return (
    <div className="panel-card">
      <div 
        className="panel-card-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span>{title}</span>
        <span className={`panel-toggle ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      {isExpanded && (
        <div className="panel-card-content">
          {children}
        </div>
      )}
    </div>
  )
}

export default PanelCard
