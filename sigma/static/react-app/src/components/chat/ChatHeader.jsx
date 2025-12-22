import { useState, useEffect } from 'react'
import { useSession } from '../../context/SessionContext'
import { useToast } from '../../context/ToastContext'
import { checkTrajectoryStorageStatus } from '../../services/api'
import './ChatHeader.css'

function ChatHeader({ onNewSession }) {
  const { 
    sessionId, 
    isSimulationActive, 
    messages,
    isAutopilotEnabled,
    setIsAutopilotEnabled,
    // Auto-save state
    trajectoryId,
    isAutoSaving,
    lastSaveTime
  } = useSession()
  const { showToast } = useToast()
  
  const [storageBackend, setStorageBackend] = useState(null)
  const [storageChecked, setStorageChecked] = useState(false)

  // Check storage configuration on mount
  useEffect(() => {
    async function checkStorage() {
      try {
        const status = await checkTrajectoryStorageStatus()
        setStorageBackend(status.backend)
      } catch (error) {
        console.error('Failed to check trajectory storage status:', error)
        setStorageBackend('local')  // Default to local
      }
      setStorageChecked(true)
    }
    checkStorage()
  }, [])

  const getStatusClass = () => {
    if (isSimulationActive) return 'active'
    return 'inactive'
  }

  const getStatusText = () => {
    if (isSimulationActive) return 'Active'
    return 'Inactive'
  }

  const getAutoSaveStatus = () => {
    if (isAutoSaving) {
      return { icon: '💾', text: 'Saving...', className: 'saving' }
    }
    if (trajectoryId && lastSaveTime) {
      const timeAgo = formatTimeAgo(lastSaveTime)
      return { icon: '✓', text: `Saved ${timeAgo}`, className: 'saved' }
    }
    if (sessionId && messages.length > 0) {
      return { icon: '○', text: 'Unsaved', className: 'unsaved' }
    }
    return null
  }

  const formatTimeAgo = (date) => {
    const seconds = Math.floor((new Date() - date) / 1000)
    if (seconds < 60) return 'just now'
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ago`
  }

  const autoSaveStatus = getAutoSaveStatus()

  return (
    <div className="chat-header">
      <div className="chat-header-left">
        <h3>💬 <span className="header-title-text">Conversation</span></h3>
        <span className={`status-badge ${getStatusClass()}`}>
          {getStatusText()}
        </span>
      </div>
      <div className="chat-header-right">
        {storageChecked && autoSaveStatus && (
          <span 
            className={`auto-save-status ${autoSaveStatus.className}`}
            title={trajectoryId ? `Trajectory ID: ${trajectoryId}` : 'Auto-save is enabled'}
          >
            <span className="save-icon">{autoSaveStatus.icon}</span>
            <span className="save-text">{autoSaveStatus.text}</span>
          </span>
        )}
        <div className="autopilot-toggle" title="When enabled, automatically triggers 'Auto' after each user/tool response">
          <span className="autopilot-label">🚀</span>
          <span className="autopilot-label-text">Autopilot</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={isAutopilotEnabled}
              onChange={(e) => setIsAutopilotEnabled(e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
        <button 
          className="btn btn-header btn-new-trajectory" 
          onClick={onNewSession}
          title="Start a new trajectory"
        >
          <span className="btn-icon">✨</span>
          <span className="btn-text">New Trajectory</span>
        </button>
      </div>
    </div>
  )
}

export default ChatHeader
