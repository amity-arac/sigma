import { useState, useEffect, useCallback } from 'react'
import { useSession } from '../context/SessionContext'
import { useToast } from '../context/ToastContext'
import { fetchEnvironments, fetchEnvironmentWiki, createSession, startSession } from '../services/api'
import WikiPopup from './WikiPopup'
import '../styles/components.css'
import './SetupPanel.css'

function SetupPanel({ onSimulationStart, isCollapsed }) {
  const [environments, setEnvironments] = useState([])
  const [selectedEnv, setSelectedEnv] = useState('')
  const [envDescription, setEnvDescription] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  
  // Wiki popup state
  const [showWikiPopup, setShowWikiPopup] = useState(false)
  const [wikiContent, setWikiContent] = useState(null)
  const [wikiContentType, setWikiContentType] = useState(null)
  const [loadingWiki, setLoadingWiki] = useState(false)
  
  // Session ready state - when session is created and ready to begin
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionData, setSessionData] = useState(null)
  
  // Advanced options
  const [taskIndex, setTaskIndex] = useState('')
  const [userModel, setUserModel] = useState('gpt-5-mini')
  const [userProvider, setUserProvider] = useState('openai')
  const [agentModel, setAgentModel] = useState('gpt-5.2')
  const [customPersona, setCustomPersona] = useState('')
  const [generateScenario, setGenerateScenario] = useState(true)
  const [taskIds, setTaskIds] = useState('')  // Comma-separated list of task IDs

  const { setSessionId, setIsSimulationActive, setTools, setPersona, setWiki, addMessage } = useSession()
  const { showToast } = useToast()

  useEffect(() => {
    loadEnvironments()
  }, [])

  useEffect(() => {
    setCollapsed(isCollapsed)
  }, [isCollapsed])

  const loadEnvironments = async () => {
    try {
      const envs = await fetchEnvironments()
      setEnvironments(envs)
      if (envs.length > 0) {
        setSelectedEnv(envs[0].name)
        setEnvDescription(envs[0].description || '')
      }
    } catch (error) {
      showToast('Failed to load environments', 'error')
    }
  }

  const handleEnvChange = (e) => {
    const envName = e.target.value
    setSelectedEnv(envName)
    const env = environments.find(env => env.name === envName)
    setEnvDescription(env?.description || '')
  }

  const handleStartSimulation = async () => {
    if (!selectedEnv) {
      showToast('Please select an environment', 'error')
      return
    }

    setIsLoading(true)
    setShowWikiPopup(true)
    setSessionReady(false)
    setSessionData(null)
    
    // Immediately start fetching wiki content to show while session initializes
    setLoadingWiki(true)
    fetchEnvironmentWiki(selectedEnv)
      .then(wikiData => {
        setWikiContent(wikiData.content)
        setWikiContentType(wikiData.content_type)
      })
      .catch(() => {
        // Silently fail - wiki preview is optional
      })
      .finally(() => {
        setLoadingWiki(false)
      })

    try {
      const payload = {
        env_name: selectedEnv,
        user_model: userModel,
        user_provider: userProvider,
        agent_model: agentModel || null,
        task_index: taskIndex ? parseInt(taskIndex) : null,
        task_split: 'test',
        persona: customPersona || null,
        generate_scenario: generateScenario,
        task_ids: taskIds ? taskIds.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id)) : null
      }

      const createData = await createSession(payload)
      setSessionId(createData.session_id)

      const data = await startSession(createData.session_id)

      // Store session data and mark as ready (don't start yet)
      setSessionData(data)
      setSessionReady(true)
      setIsLoading(false)
      
    } catch (error) {
      showToast(error.message, 'error')
      // Close popup on error
      setShowWikiPopup(false)
      setWikiContent(null)
      setWikiContentType(null)
      setIsLoading(false)
    }
  }

  // Called when user clicks "Begin Session" in the popup
  const handleBeginSession = useCallback(() => {
    if (!sessionData) return
    
    setIsSimulationActive(true)
    setTools(sessionData.tools)
    setPersona(sessionData.persona)
    setWiki(sessionData.wiki)
    addMessage('user', sessionData.initial_message)

    // Close popup and clear state
    setShowWikiPopup(false)
    setWikiContent(null)
    setWikiContentType(null)
    setSessionReady(false)
    setSessionData(null)

    showToast('Simulation started!', 'success')
    onSimulationStart()
  }, [sessionData, setIsSimulationActive, setTools, setPersona, setWiki, addMessage, showToast, onSimulationStart])

  // Called when user closes the popup without beginning
  const handleCloseWikiPopup = useCallback(() => {
    setShowWikiPopup(false)
    setWikiContent(null)
    setWikiContentType(null)
    // If session was ready but user closed, that's ok - they can restart
    if (sessionReady) {
      showToast('Session ready - click Start Simulation again to continue', 'info')
    }
    setSessionReady(false)
    setSessionData(null)
    setIsLoading(false)
  }, [sessionReady, showToast])

  const toggleCollapse = () => {
    setCollapsed(!collapsed)
  }

  return (
    <div className={`setup-panel ${collapsed ? 'collapsed' : ''}`}>
      <h2 onClick={toggleCollapse}>
        📋 Session Setup 
        <span className="toggle-icon">▼</span>
      </h2>
      
      {!collapsed && (
        <div className="setup-content">
          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label htmlFor="envSelect">Environment</label>
              <select 
                id="envSelect" 
                value={selectedEnv} 
                onChange={handleEnvChange}
              >
                {environments.length === 0 ? (
                  <option value="">Loading environments...</option>
                ) : (
                  environments.map(env => (
                    <option key={env.name} value={env.name}>
                      {env.display_name}
                    </option>
                  ))
                )}
              </select>
              {envDescription && (
                <div className="env-description visible">
                  {envDescription}
                </div>
              )}
            </div>
          </div>

          <div className="form-row">
            <button 
              className="btn btn-success" 
              onClick={handleStartSimulation}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span> Starting...
                </>
              ) : (
                'Start Simulation'
              )}
            </button>
          </div>

          <div className="advanced-section">
            <div 
              className="advanced-header" 
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span>⚙️ Advanced Options</span>
              <span className={`advanced-toggle-icon ${showAdvanced ? 'expanded' : ''}`}>
                ▶
              </span>
            </div>
            
            {showAdvanced && (
              <div className="advanced-content">
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="taskIndex">Task Index (optional)</label>
                    <input 
                      type="number" 
                      id="taskIndex" 
                      value={taskIndex}
                      onChange={(e) => setTaskIndex(e.target.value)}
                      placeholder="Random if empty"
                      disabled={generateScenario}
                    />
                  </div>
                  <div className="form-group checkbox-group">
                    <label htmlFor="generateScenario" className="checkbox-label">
                      <input 
                        type="checkbox" 
                        id="generateScenario" 
                        checked={generateScenario}
                        onChange={(e) => setGenerateScenario(e.target.checked)}
                      />
                      <span className="checkbox-text">🎲 Auto-Generate Scenario</span>
                    </label>
                    <span className="checkbox-hint">
                      Use AI to create unique scenarios inspired by existing tasks
                    </span>
                  </div>
                </div>
                
                {generateScenario && (
                  <div className="form-row">
                    <div className="form-group" style={{ flex: 2 }}>
                      <label htmlFor="taskIds">Filter by Task IDs (optional)</label>
                      <input 
                        type="text" 
                        id="taskIds" 
                        value={taskIds}
                        onChange={(e) => setTaskIds(e.target.value)}
                        placeholder="e.g., 1,5,12,23 (comma-separated)"
                      />
                      <span className="form-hint">
                        Limit scenario generation to specific task IDs for focused training
                      </span>
                    </div>
                  </div>
                )}
                
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="userModel">User Model</label>
                    <input 
                      type="text" 
                      id="userModel" 
                      value={userModel}
                      onChange={(e) => setUserModel(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="userProvider">User Provider</label>
                    <input 
                      type="text" 
                      id="userProvider" 
                      value={userProvider}
                      onChange={(e) => setUserProvider(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="agentModel">Agent Model (for assistance)</label>
                    <input 
                      type="text" 
                      id="agentModel" 
                      value={agentModel}
                      onChange={(e) => setAgentModel(e.target.value)}
                    />
                  </div>
                </div>
                
                <div className="form-row">
                  <div className="form-group" style={{ flex: 2 }}>
                    <label htmlFor="customPersona">Custom Persona (optional)</label>
                    <textarea 
                      id="customPersona" 
                      value={customPersona}
                      onChange={(e) => setCustomPersona(e.target.value)}
                      placeholder="E.g., You are John Smith. You want to cancel order #W1234567 because you ordered by mistake."
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Wiki popup shown while session is starting */}
      {showWikiPopup && (
        <WikiPopup
          content={wikiContent}
          contentType={wikiContentType}
          loading={loadingWiki}
          envName={selectedEnv}
          sessionReady={sessionReady}
          onBeginSession={handleBeginSession}
          onClose={handleCloseWikiPopup}
        />
      )}
    </div>
  )
}

export default SetupPanel
