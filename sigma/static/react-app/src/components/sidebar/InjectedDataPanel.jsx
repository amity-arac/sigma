import { useState } from 'react'
import { useSession } from '../../context/SessionContext'
import PanelCard from './PanelCard'
import './InjectedDataPanel.css'

/**
 * Renders a collapsible JSON tree view
 */
function JsonTreeNode({ keyName, value, depth = 0 }) {
  const [isExpanded, setIsExpanded] = useState(depth < 2) // Auto-expand first 2 levels
  
  const isObject = value !== null && typeof value === 'object'
  const isArray = Array.isArray(value)
  const isEmpty = isObject && Object.keys(value).length === 0
  
  if (!isObject || isEmpty) {
    // Render primitive value
    let displayValue = value
    let valueClass = 'json-value'
    
    if (value === null) {
      displayValue = 'null'
      valueClass += ' json-null'
    } else if (typeof value === 'boolean') {
      displayValue = value.toString()
      valueClass += ' json-boolean'
    } else if (typeof value === 'number') {
      valueClass += ' json-number'
    } else if (typeof value === 'string') {
      // Truncate long strings
      if (value.length > 100) {
        displayValue = `"${value.substring(0, 100)}..."`
      } else {
        displayValue = `"${value}"`
      }
      valueClass += ' json-string'
    }
    
    return (
      <div className="json-node json-leaf" style={{ paddingLeft: `${depth * 16}px` }}>
        {keyName !== null && <span className="json-key">{keyName}: </span>}
        <span className={valueClass}>{displayValue}</span>
      </div>
    )
  }
  
  // Render object/array
  const entries = Object.entries(value)
  const bracket = isArray ? ['[', ']'] : ['{', '}']
  const itemCount = entries.length
  
  return (
    <div className="json-node">
      <div 
        className="json-node-header"
        style={{ paddingLeft: `${depth * 16}px` }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className={`json-toggle ${isExpanded ? 'expanded' : ''}`}>▶</span>
        {keyName !== null && <span className="json-key">{keyName}: </span>}
        <span className="json-bracket">{bracket[0]}</span>
        {!isExpanded && (
          <span className="json-collapsed-info">
            {itemCount} {isArray ? 'items' : 'properties'}
          </span>
        )}
        {!isExpanded && <span className="json-bracket">{bracket[1]}</span>}
      </div>
      {isExpanded && (
        <>
          <div className="json-children">
            {entries.map(([k, v], index) => (
              <JsonTreeNode 
                key={k} 
                keyName={isArray ? index : k} 
                value={v} 
                depth={depth + 1} 
              />
            ))}
          </div>
          <div style={{ paddingLeft: `${depth * 16}px` }}>
            <span className="json-bracket">{bracket[1]}</span>
          </div>
        </>
      )}
    </div>
  )
}

/**
 * Panel component to display injected scenario data
 * @param {Object} props
 * @param {boolean} props.embedded - If true, renders without PanelCard wrapper
 */
function InjectedDataPanel({ embedded = false }) {
  const { injectedData } = useSession()
  const [viewMode, setViewMode] = useState('tree') // 'tree' or 'raw'
  
  if (!injectedData) {
    if (embedded) {
      return (
        <div className="no-injected-data">
          No scenario data available.
        </div>
      )
    }
    return null // Don't show panel if no injected data
  }
  
  // Extract augmented_data which contains the actual injected data
  const augmentedData = injectedData.augmented_data || {}
  const hasAugmentedData = Object.keys(augmentedData).length > 0
  
  // Get scenario metadata
  const scenarioGoal = injectedData.scenario_goal
  const seedInstruction = injectedData.seed_instruction
  const seedTaskId = injectedData.seed_task_id
  const userData = injectedData.user
  
  const content = (
    <div className="injected-data-panel">
      {/* View mode toggle */}
      <div className="injected-data-controls">
        <button 
          className={`view-mode-btn ${viewMode === 'tree' ? 'active' : ''}`}
          onClick={() => setViewMode('tree')}
        >
          Tree
        </button>
        <button 
          className={`view-mode-btn ${viewMode === 'raw' ? 'active' : ''}`}
          onClick={() => setViewMode('raw')}
        >
          Raw JSON
        </button>
      </div>
      
      {/* Scenario Goal */}
      {scenarioGoal && (
        <div className="injected-section">
          <div className="injected-section-header">🎯 Scenario Goal</div>
          <div className="injected-section-content scenario-goal">
            {scenarioGoal}
          </div>
        </div>
      )}
      
      {/* Seed Task ID */}
      {seedTaskId && (
        <div className="injected-section">
          <div className="injected-section-header">🔗 Seed Task ID</div>
          <div className="injected-section-content seed-task-id">
            <code>{seedTaskId}</code>
          </div>
        </div>
      )}
      
      {/* User Data Summary */}
      {userData && (
        <div className="injected-section">
          <div className="injected-section-header">👤 Injected User</div>
          <div className="injected-section-content">
            {userData.user_id && <div><strong>ID:</strong> {userData.user_id}</div>}
            {userData.name && (
              <div>
                <strong>Name:</strong> {typeof userData.name === 'object' 
                  ? `${userData.name.first_name || ''} ${userData.name.last_name || ''}`.trim()
                  : userData.name
                }
              </div>
            )}
            {userData.email && <div><strong>Email:</strong> {userData.email}</div>}
          </div>
        </div>
      )}
      
      {/* Augmented Data (Orders, Products, etc.) */}
      {hasAugmentedData && (
        <div className="injected-section">
          <div className="injected-section-header">📦 Augmented Database Records</div>
          <div className="injected-section-content">
            {viewMode === 'tree' ? (
              <div className="json-tree-container">
                {Object.entries(augmentedData).map(([collection, data]) => (
                  <div key={collection} className="collection-group">
                    <JsonTreeNode keyName={collection} value={data} depth={0} />
                  </div>
                ))}
              </div>
            ) : (
              <pre className="raw-json">
                {JSON.stringify(augmentedData, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
      
      {/* Seed Instruction (collapsed by default) */}
      {seedInstruction && (
        <details className="injected-section">
          <summary className="injected-section-header">🌱 Seed Task Inspiration</summary>
          <div className="injected-section-content seed-instruction">
            {seedInstruction}
          </div>
        </details>
      )}
      
      {!hasAugmentedData && !scenarioGoal && !userData && (
        <div className="no-injected-data">
          No augmented data available for this scenario.
        </div>
      )}
    </div>
  )
  
  // If embedded mode, return content directly without PanelCard wrapper
  if (embedded) {
    return content
  }
  
  return (
    <PanelCard title="📋 Scenario Data" defaultExpanded={false}>
      {content}
    </PanelCard>
  )
}

export default InjectedDataPanel
