import { useState } from 'react'
import { useSession } from '../../context/SessionContext'
import { useToast } from '../../context/ToastContext'
import { callTool } from '../../services/api'
import './ToolForm.css'

function ToolForm({ tool }) {
  const [paramValues, setParamValues] = useState({})
  const [isExecuting, setIsExecuting] = useState(false)
  
  const { sessionId, isSimulationActive, addMessage } = useSession()
  const { showToast } = useToast()

  const handleParamChange = (paramName, value) => {
    setParamValues(prev => ({
      ...prev,
      [paramName]: value
    }))
  }

  const parseValue = (value, type) => {
    if (!value) return undefined
    
    switch (type) {
      case 'integer':
        return parseInt(value)
      case 'number':
        return parseFloat(value)
      case 'boolean':
        return value.toLowerCase() === 'true'
      case 'array':
      case 'object':
        try {
          return JSON.parse(value)
        } catch {
          if (type === 'array') {
            return value.split(',').map(v => v.trim())
          }
          return value
        }
      default:
        return value
    }
  }

  const handleExecute = async () => {
    if (!isSimulationActive) return
    
    setIsExecuting(true)
    
    const args = {}
    for (const [paramName, paramInfo] of Object.entries(tool.parameters || {})) {
      const value = paramValues[paramName]
      if (value) {
        args[paramName] = parseValue(value, paramInfo.type)
      }
    }

    try {
      addMessage('tool', `🔧 Calling ${tool.name}\n${JSON.stringify(args, null, 2)}`, {
        toolName: tool.name,
        toolArguments: args
      })
      const data = await callTool(sessionId, tool.name, args)
      addMessage('tool-result', data.observation, {
        toolName: tool.name
      })
    } catch (error) {
      showToast(error.message, 'error')
    } finally {
      setIsExecuting(false)
    }
  }

  const params = tool.parameters || {}
  const required = tool.required_params || []

  return (
    <div className="tool-form">
      <h4>{tool.name}</h4>
      
      {Object.entries(params).map(([paramName, paramInfo]) => (
        <div key={paramName} className="tool-param">
          <label>
            {paramName}
            {required.includes(paramName) && (
              <span className="required">*</span>
            )}
            <span className="param-type">({paramInfo.type})</span>
          </label>
          <input
            type="text"
            value={paramValues[paramName] || ''}
            onChange={(e) => handleParamChange(paramName, e.target.value)}
            placeholder={paramInfo.description || ''}
          />
        </div>
      ))}
      
      <button 
        className="btn btn-primary"
        onClick={handleExecute}
        disabled={isExecuting || !isSimulationActive}
      >
        {isExecuting ? 'Executing...' : 'Execute Tool'}
      </button>
    </div>
  )
}

export default ToolForm
