import { useState } from 'react'
import './PanelCard.css'

function PanelCard({ title, children, defaultExpanded = false, flexGrow = false }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  return (
    <div className={`panel-card ${flexGrow ? 'panel-card-flex-grow' : ''}`}>
      <div 
        className="panel-card-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span>{title}</span>
        <span className={`panel-toggle ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      {isExpanded && (
        <div className={`panel-card-content ${flexGrow ? 'panel-card-content-flex' : ''}`}>
          {children}
        </div>
      )}
    </div>
  )
}

export default PanelCard
