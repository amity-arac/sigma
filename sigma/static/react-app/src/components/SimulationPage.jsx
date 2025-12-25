import { useState, useEffect, useCallback } from 'react'
import { useSession } from '../context/SessionContext'
import { useToast } from '../context/ToastContext'
import { getTrajectory, continueTrajectory, updateTrajectory } from '../services/api'
import ChatPanel from './chat/ChatPanel'
import SidePanel from './sidebar/SidePanel'
import MobileInfoPanel from './chat/MobileInfoPanel'
import ConfirmDialog from './common/ConfirmDialog'
import './MainContent.css'
import './SimulationPage.css'

function SimulationPage({ trajectoryId, onNavigate }) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [isTrajectoryDone, setIsTrajectoryDone] = useState(false)
  const [wasOriginallyCompleted, setWasOriginallyCompleted] = useState(false)  // Track if opened as completed
  const [showConfirmDialog, setShowConfirmDialog] = useState(false)
  const [confirmDialogConfig, setConfirmDialogConfig] = useState({})
  const [trajectoryEnv, setTrajectoryEnv] = useState(null)
  
  const { 
    sessionId,
    setSessionId,
    setIsSimulationActive,
    setTools,
    setPersona,
    setWiki,
    setInjectedData,
    messages,
    setMessages,
    clearMessages,
    resetSession,
    setTrajectoryId,
    markMessagesSaved,
    finalResult,
    setAutopilotTurnCount,
    setIsAutopilotEnabled,
    setIsAutoApproveEnabled
  } = useSession()
  const { showToast } = useToast()

  // Load trajectory and initialize session
  useEffect(() => {
    if (!trajectoryId) {
      setError('No trajectory ID provided')
      setIsLoading(false)
      return
    }

    loadTrajectory()
  }, [trajectoryId])

  const loadTrajectory = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // First, we need to get the trajectory to find its env_name
      // We'll try to get trajectory info from the list endpoint or a dedicated endpoint
      // For now, we'll need to get the env from the trajectory itself
      
      // Try to continue the trajectory - this will restore the session
      // We need to find the env_name first - try listing all environments and finding the trajectory
      const { getTrajectoryByIdAnyEnv } = await import('../services/api')
      
      let trajectoryData
      try {
        trajectoryData = await getTrajectoryByIdAnyEnv(trajectoryId)
      } catch (e) {
        // Fallback: try to continue directly (the API might handle it)
        throw new Error('Trajectory not found. It may have been deleted or the ID is invalid.')
      }

      const envName = trajectoryData.env_name
      setTrajectoryEnv(envName)
      
      // Continue the trajectory to get a live session
      const data = await continueTrajectory(trajectoryId, envName, {
        userModel: 'gpt-5-mini',
        userProvider: 'openai'
      })
      
      setSessionId(data.session_id)
      setTrajectoryId(trajectoryId)
      setTools(data.tools)
      setPersona(data.persona)
      setWiki(data.wiki)
      
      // Set injected data from persona_data (augmented_data)
      if (data.persona_data) {
        setInjectedData(data.persona_data)
      }
      
      // Reset autopilot turn count BEFORE setting simulation active
      // This prevents the turn limit check from firing with stale count
      setAutopilotTurnCount(0)
      
      // Restore messages from trajectory BEFORE setting simulation active
      // This ensures messages are ready when autopilot checks kick in
      if (data.messages && data.messages.length > 0) {
        const restoredMessages = data.messages.map((msg, index) => ({
          // Preserve the original message ID from the stored trajectory
          // This is critical for edit operations to work correctly
          id: msg.id || (Date.now() + index + Math.random()),
          role: msg.role,
          content: msg.content,
          reasoning: msg.reasoning || null,
          isTemporary: false,
          timestamp: msg.timestamp || new Date().toISOString(),
          tool_name: msg.tool_name || null,
          tool_arguments: msg.tool_arguments || null,
          rejected: msg.rejected || null,
        }))
        setMessages(restoredMessages)
        // Mark these messages as already saved to prevent immediate auto-save
        // Also pass the saved finalResult to include completion state in the saved state key
        const savedResult = trajectoryData.is_done ? {
          is_done: trajectoryData.is_done,
          reward: trajectoryData.reward,
          reward_info: trajectoryData.reward_info
        } : null
        markMessagesSaved(restoredMessages, savedResult)
      }
      
      // Check if trajectory was already completed
      const isDone = trajectoryData.is_done
      setIsTrajectoryDone(isDone)
      setWasOriginallyCompleted(isDone)  // Remember original completion state
      
      // If trajectory is completed, disable autopilot and auto-approve to prevent auto-generation
      if (isDone) {
        setIsAutopilotEnabled(false)
        setIsAutoApproveEnabled(false)
      }
      
      // Set simulation active LAST after all state is ready
      // This prevents autopilot from triggering before state is initialized
      // Even if done, we allow simulation to be active so user can edit
      setIsSimulationActive(true)
      
      setIsLoading(false)
    } catch (err) {
      console.error('Failed to load trajectory:', err)
      setError(err.message || 'Failed to load trajectory')
      setIsLoading(false)
    }
  }

  const handleSimulationEnd = useCallback((resultData) => {
    setIsTrajectoryDone(resultData.done)
    if (resultData.done) {
      setWasOriginallyCompleted(true)  // Mark as completed now
    }
  }, [])

  // Handler to mark trajectory as incomplete when editing a completed one
  const handleMarkIncomplete = useCallback(async () => {
    if (!wasOriginallyCompleted || !trajectoryEnv) return
    
    try {
      await updateTrajectory(trajectoryId, trajectoryEnv, { is_done: false })
      setIsTrajectoryDone(false)
      setWasOriginallyCompleted(false)
      showToast('Trajectory marked as incomplete for editing', 'info')
    } catch (err) {
      showToast(`Failed to update trajectory status: ${err.message}`, 'error')
    }
  }, [wasOriginallyCompleted, trajectoryEnv, trajectoryId, showToast])

  const handleNewSession = useCallback(() => {
    // Navigate back to setup page
    clearMessages()
    resetSession()
    onNavigate('/')
  }, [clearMessages, resetSession, onNavigate])

  const handleNewSessionConfirm = useCallback(() => {
    if (messages.length > 0) {
      setConfirmDialogConfig({
        title: '✨ Start New Trajectory',
        message: 'Are you sure you want to start a new trajectory? You can always come back to this one from the Trajectory page.',
        onConfirm: () => {
          handleNewSession()
          setShowConfirmDialog(false)
        },
        onCancel: () => setShowConfirmDialog(false)
      })
      setShowConfirmDialog(true)
    } else {
      handleNewSession()
    }
  }, [messages, handleNewSession])

  if (isLoading) {
    return (
      <div className="simulation-page">
        <div className="simulation-loading">
          <div className="spinner"></div>
          <p>Loading trajectory...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="simulation-page">
        <div className="simulation-error">
          <h2>⚠️ Error Loading Trajectory</h2>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={() => onNavigate('/')}>
            ← Back to Setup
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="simulation-page">
      <div className="simulation-header">
        <button className="btn btn-header btn-back" onClick={() => onNavigate('/')}>
          ←
        </button>
        <span className="trajectory-badge">
          📂 {trajectoryId.slice(0, 8)}...
        </span>
        <span className={`status-badge ${isTrajectoryDone ? 'complete' : 'incomplete'}`}>
          {isTrajectoryDone ? '✓ Complete' : '○ In Progress'}
        </span>
        {wasOriginallyCompleted && !isTrajectoryDone && (
          <span className="status-badge editing" title="This trajectory was marked complete but is now being edited">
            ✏️ Editing
          </span>
        )}
        {/* Spacer to push content to edges */}
        <div className="header-spacer"></div>
      </div>
      
      <div className="main-content">
        <ChatPanel 
          onSimulationEnd={handleSimulationEnd}
          onNewSession={handleNewSessionConfirm}
          wasOriginallyCompleted={wasOriginallyCompleted}
          onMarkIncomplete={handleMarkIncomplete}
        />
        <SidePanel />
      </div>
      
      {/* Mobile info panel - shows tools, persona, wiki, settings on mobile */}
      <MobileInfoPanel onNewSession={handleNewSessionConfirm} />
      
      {showConfirmDialog && (
        <ConfirmDialog
          title={confirmDialogConfig.title}
          message={confirmDialogConfig.message}
          onConfirm={confirmDialogConfig.onConfirm}
          onCancel={confirmDialogConfig.onCancel}
        />
      )}
    </div>
  )
}

export default SimulationPage
