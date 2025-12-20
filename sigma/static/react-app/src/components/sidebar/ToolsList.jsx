import { useState } from 'react'
import ToolForm from './ToolForm'
import './ToolsList.css'

function ToolsList({ tools }) {
  const [selectedToolIndex, setSelectedToolIndex] = useState(null)

  const handleToolClick = (index) => {
    setSelectedToolIndex(selectedToolIndex === index ? null : index)
  }

  return (
    <div className="tools-container">
      <div className="tools-list">
        {tools.map((tool, index) => (
          <div 
            key={tool.name}
            className={`tool-item ${selectedToolIndex === index ? 'selected' : ''}`}
            onClick={() => handleToolClick(index)}
          >
            <div className="tool-name">{tool.name}</div>
            <div className="tool-desc">
              {tool.description?.slice(0, 80)}...
            </div>
          </div>
        ))}
      </div>
      
      {selectedToolIndex !== null && (
        <ToolForm tool={tools[selectedToolIndex]} />
      )}
    </div>
  )
}

export default ToolsList
