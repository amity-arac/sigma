import { useState, useEffect, useCallback } from 'react'
import { useToast } from '../context/ToastContext'
import { fetchEnvironments, fetchEnvironmentWiki, createSession, listTrajectories } from '../services/api'
import WikiPopup from './WikiPopup'
import '../styles/components.css'
import './SetupPanel.css'

function SetupPanel({ onNavigate }) {
  const [environments, setEnvironments] = useState([])
  const [selectedEnv, setSelectedEnv] = useState('')
  const [envDescription, setEnvDescription] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Wiki popup state
  const [showWikiPopup, setShowWikiPopup] = useState(false)
  const [wikiContent, setWikiContent] = useState(null)
  const [wikiContentType, setWikiContentType] = useState(null)
  const [loadingWiki, setLoadingWiki] = useState(false)
  
  // Session ready state - when session is created and ready to begin
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionData, setSessionData] = useState(null)
  const [createdTrajectoryId, setCreatedTrajectoryId] = useState(null)
  
  // Trajectory selection mode
  const [trajectoryMode, setTrajectoryMode] = useState('new') // 'new' or 'continue'
  const [existingTrajectories, setExistingTrajectories] = useState([])
  const [selectedTrajectoryId, setSelectedTrajectoryId] = useState('')
  const [loadingTrajectories, setLoadingTrajectories] = useState(false)
  
  // Scenario source selection
  const [scenarioSource, setScenarioSource] = useState('generate') // 'generate' or 'existing'
  
  // Options for "Use Existing Task"
  const [taskIndex, setTaskIndex] = useState('')
  
  // Options for "Generate New Scenario"
  const [taskIds, setTaskIds] = useState('')  // Comma-separated list of task IDs to use as inspiration
  
  // Advanced options (common)
  const [userModel, setUserModel] = useState('gpt-5-mini')
  const [userProvider, setUserProvider] = useState('openai')
  const [agentModel, setAgentModel] = useState('gpt-5.2')
  const [customPersona, setCustomPersona] = useState('')

  const { showToast } = useToast()

  // Cookie helper functions
  const getCookie = (name) => {
    const value = `; ${document.cookie}`
    const parts = value.split(`; ${name}=`)
    if (parts.length === 2) return parts.pop().split(';').shift()
    return null
  }

  const setCookie = (name, value, days = 365) => {
    const expires = new Date(Date.now() + days * 864e5).toUTCString()
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`
  }

  // Load taskIds from cookie on mount
  useEffect(() => {
    const savedTaskIds = getCookie('sigma_inspiration_task_ids')
    if (savedTaskIds) {
      setTaskIds(decodeURIComponent(savedTaskIds))
    }
  }, [])

  useEffect(() => {
    loadEnvironments()
  }, [])

  // Load trajectories when environment changes and continue mode is selected
  useEffect(() => {
    if (trajectoryMode === 'continue' && selectedEnv) {
      loadTrajectories()
    }
  }, [trajectoryMode, selectedEnv])

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

  const loadTrajectories = async () => {
    if (!selectedEnv) return
    setLoadingTrajectories(true)
    try {
      const result = await listTrajectories(selectedEnv, 50)
      // Sort by created_at descending and filter incomplete ones first
      const sorted = (result.trajectories || [])
        .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
      setExistingTrajectories(sorted)
      setSelectedTrajectoryId('')
    } catch (error) {
      showToast('Failed to load trajectories', 'error')
      setExistingTrajectories([])
    } finally {
      setLoadingTrajectories(false)
    }
  }

  const handleEnvChange = (e) => {
    const envName = e.target.value
    setSelectedEnv(envName)
    const env = environments.find(env => env.name === envName)
    setEnvDescription(env?.description || '')
    // Reset trajectory selection when environment changes
    setSelectedTrajectoryId('')
    setExistingTrajectories([])
  }

  const handleTrajectoryModeChange = (mode) => {
    setTrajectoryMode(mode)
    if (mode === 'continue' && selectedEnv && existingTrajectories.length === 0) {
      loadTrajectories()
    }
  }

  const handleStartSimulation = async () => {
    if (!selectedEnv) {
      showToast('Please select an environment', 'error')
      return
    }

    // For continue mode, just navigate directly
    if (trajectoryMode === 'continue') {
      if (!selectedTrajectoryId) {
        showToast('Please select a trajectory to continue', 'error')
        return
      }
      // Navigate to the simulation page
      onNavigate(`/trajectories/${selectedTrajectoryId}/simulation`)
      return
    }

    setIsLoading(true)
    setShowWikiPopup(true)
    setSessionReady(false)
    setSessionData(null)
    setCreatedTrajectoryId(null)
    
    // Immediately start fetching wiki content to show while session initializes
    setLoadingWiki(true)
    fetchEnvironmentWiki(selectedEnv)
      .then(wikiData => {
        setWikiContent(wikiData.content)
        setWikiContentType(wikiData.content_type)
      })
      .catch((err) => {
        // Log error but don't show to user - wiki will fallback to session response
        console.warn('Wiki fetch failed, will use session response fallback:', err.message)
      })
      .finally(() => {
        setLoadingWiki(false)
      })

    try {
      // Save taskIds to cookie if using generate scenario
      if (scenarioSource === 'generate' && taskIds) {
        setCookie('sigma_inspiration_task_ids', taskIds)
      }

      // Create new trajectory - this now creates, starts, AND saves in one call
      const payload = {
        env_name: selectedEnv,
        user_model: userModel,
        user_provider: userProvider,
        agent_model: agentModel || null,
        task_index: scenarioSource === 'existing' && taskIndex ? parseInt(taskIndex) : null,
        task_split: 'test',
        persona: customPersona || null,
        generate_scenario: scenarioSource === 'generate',
        task_ids: scenarioSource === 'generate' && taskIds ? taskIds.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id)) : null
      }

      // createSession now calls POST /trajectories which creates + starts + saves
      const data = await createSession(payload)
      
      // The trajectory_id is returned directly (session_id is the same)
      const trajectoryId = data.trajectory_id || data.session_id
      setCreatedTrajectoryId(trajectoryId)

      // Store session data and mark as ready (don't start yet)
      // The response already has all the data we need
      setSessionData({
        session_id: trajectoryId,
        initial_message: data.initial_message,
        persona: data.persona,
        tools: data.tools,
        wiki: data.wiki,
        generated_scenario: data.generated_scenario,
      })
      
      // Use wiki from session response as fallback if parallel fetch hasn't completed
      // This ensures wiki is always shown even if the separate wiki fetch failed/timed out
      // Use functional update to check current state
      setWikiContent(currentContent => {
        if (!currentContent && data.wiki) {
          setWikiContentType('markdown')  // Session response wiki is always markdown
          setLoadingWiki(false)
          return data.wiki
        }
        return currentContent
      })
      
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

  // Called when user clicks "Begin Trajectory" in the popup
  const handleBeginSession = useCallback(() => {
    if (!sessionData || !createdTrajectoryId) return
    
    // Close popup and clear state
    setShowWikiPopup(false)
    setWikiContent(null)
    setWikiContentType(null)
    setSessionReady(false)
    setSessionData(null)

    // Navigate to the simulation page
    onNavigate(`/trajectories/${createdTrajectoryId}/simulation`)
    showToast('Trajectory created!', 'success')
  }, [sessionData, createdTrajectoryId, onNavigate, showToast])

  // Called when user closes the popup without beginning
  const handleCloseWikiPopup = useCallback(() => {
    setShowWikiPopup(false)
    setWikiContent(null)
    setWikiContentType(null)
    setSessionReady(false)
    setSessionData(null)
    setCreatedTrajectoryId(null)
    setIsLoading(false)
  }, [])

  // Format trajectory for display
  const formatTrajectoryOption = (trajectory) => {
    const date = trajectory.created_at ? new Date(trajectory.created_at).toLocaleString() : 'Unknown date'
    const status = trajectory.is_done ? '✓' : '○'
    const reward = trajectory.reward !== null && trajectory.reward !== undefined ? ` (${trajectory.reward})` : ''
    return `${status} ${date}${reward} - ${trajectory.id?.slice(0, 8)}...`
  }

  return (
    <div className="setup-panel">
      <h2>📋 Trajectory Setup</h2>
      
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

          {/* Trajectory Mode Selection */}
          <div className="form-row trajectory-mode-row">
            <div className="trajectory-mode-toggle">
              <button 
                className={`mode-btn ${trajectoryMode === 'new' ? 'active' : ''}`}
                onClick={() => handleTrajectoryModeChange('new')}
              >
                ✨ New Trajectory
              </button>
              <button 
                className={`mode-btn ${trajectoryMode === 'continue' ? 'active' : ''}`}
                onClick={() => handleTrajectoryModeChange('continue')}
              >
                📂 Continue Existing
              </button>
            </div>
          </div>

          {/* Trajectory Selection (only in continue mode) */}
          {trajectoryMode === 'continue' && (
            <div className="form-row">
              <div className="form-group" style={{ flex: 2 }}>
                <label htmlFor="trajectorySelect">Select Trajectory to Continue</label>
                <select 
                  id="trajectorySelect" 
                  value={selectedTrajectoryId} 
                  onChange={(e) => setSelectedTrajectoryId(e.target.value)}
                  disabled={loadingTrajectories}
                >
                  <option value="">
                    {loadingTrajectories ? 'Loading trajectories...' : '-- Select a trajectory --'}
                  </option>
                  {existingTrajectories.map(traj => (
                    <option key={traj.id} value={traj.id}>
                      {formatTrajectoryOption(traj)}
                    </option>
                  ))}
                </select>
                {existingTrajectories.length === 0 && !loadingTrajectories && (
                  <div className="form-hint">
                    No trajectories found for this environment.
                  </div>
                )}
                <div className="form-hint">
                  ✓ = completed, ○ = in progress
                </div>
              </div>
              <button 
                className="btn btn-secondary btn-refresh" 
                onClick={loadTrajectories}
                disabled={loadingTrajectories}
                title="Refresh trajectory list"
              >
                🔄
              </button>
            </div>
          )}

          <div className="form-row">
            <button 
              className="btn btn-success" 
              onClick={handleStartSimulation}
              disabled={isLoading || (trajectoryMode === 'continue' && !selectedTrajectoryId)}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span> Starting...
                </>
              ) : trajectoryMode === 'continue' ? (
                '▶️ Continue Trajectory'
              ) : (
                '🚀 Start New Trajectory'
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
                {/* Scenario Source Selection */}
                <div className="scenario-source-section">
                  <label className="section-label">Scenario Source</label>
                  <div className="scenario-source-toggle">
                    <button 
                      className={`source-btn ${scenarioSource === 'generate' ? 'active' : ''}`}
                      onClick={() => setScenarioSource('generate')}
                      type="button"
                    >
                      🎲 Generate New Scenario
                    </button>
                    <button 
                      className={`source-btn ${scenarioSource === 'existing' ? 'active' : ''}`}
                      onClick={() => setScenarioSource('existing')}
                      type="button"
                    >
                      📋 Use Existing Task
                    </button>
                  </div>
                </div>

                {/* Options for Generate New Scenario */}
                {scenarioSource === 'generate' && (
                  <div className="scenario-options-box">
                    <div className="options-description">
                      AI will create a unique scenario inspired by existing tasks
                    </div>
                    <div className="form-row">
                      <div className="form-group" style={{ flex: 2 }}>
                        <label htmlFor="taskIds">Inspiration Task IDs (optional)</label>
                        <input 
                          type="text" 
                          id="taskIds" 
                          value={taskIds}
                          onChange={(e) => setTaskIds(e.target.value)}
                          placeholder="e.g., 1,5,12,23 (comma-separated)"
                        />
                        <span className="form-hint">
                          Limit which tasks to sample from when generating scenarios. Leave empty to use all tasks.
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Options for Use Existing Task */}
                {scenarioSource === 'existing' && (
                  <div className="scenario-options-box">
                    <div className="options-description">
                      Use a pre-defined task from the task bank
                    </div>
                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="taskIndex">Task Index</label>
                        <input 
                          type="number" 
                          id="taskIndex" 
                          value={taskIndex}
                          onChange={(e) => setTaskIndex(e.target.value)}
                          placeholder="Random if empty"
                          min="0"
                        />
                        <span className="form-hint">
                          Specify the array index of the task to use. Leave empty for random selection.
                        </span>
                      </div>
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

      {/* Wiki popup shown while session is starting */}
      {showWikiPopup && (
        <WikiPopup
          content={wikiContent}
          contentType={wikiContentType}
          loading={loadingWiki}
          envName={selectedEnv}
          sessionReady={sessionReady}
          generatedScenario={sessionData?.generated_scenario}
          onBeginSession={handleBeginSession}
          onClose={handleCloseWikiPopup}
        />
      )}
    </div>
  )
}

export default SetupPanel