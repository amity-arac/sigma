import { useState, useEffect } from 'react'
import { useSession } from '../../context/SessionContext'
import { checkTrajectoryStorageStatus } from '../../services/api'
import './ChatHeader.css'

function ChatHeader({ approvalLogCount = 0, onViewApprovalLogs }) {
  const { 
    sessionId, 
    messages,
    isAutopilotEnabled,
    setIsAutopilotEnabled,
    isAutoApproveEnabled,
    setIsAutoApproveEnabled,
    autopilotTurnCount,
    setAutopilotTurnCount,
    AUTOPILOT_TURN_LIMIT,
    // Auto-save state
    trajectoryId,
    isAutoSaving,
    lastSaveTime
  } = useSession()
  
  const [storageChecked, setStorageChecked] = useState(false)

  // Check storage configuration on mount
  useEffect(() => {
    async function checkStorage() {
      try {
        await checkTrajectoryStorageStatus()
      } catch (error) {
        console.error('Failed to check trajectory storage status:', error)
      }
      setStorageChecked(true)
    }
    checkStorage()
  }, [])

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
        <div 
          className="autopilot-toggle" 
          title={`Autopilot: Auto-generate agent actions after each user/tool response\n\nTurns: ${autopilotTurnCount}/${AUTOPILOT_TURN_LIMIT}`}
        >
          <span className="autopilot-label">🚀</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={isAutopilotEnabled}
              onChange={(e) => {
                setIsAutopilotEnabled(e.target.checked)
                // Reset turn counter when enabling autopilot
                if (e.target.checked) {
                  setAutopilotTurnCount(0)
                }
              }}
            />
            <span className="toggle-slider"></span>
          </label>
          {isAutopilotEnabled && (
            <span 
              className={`turn-counter ${autopilotTurnCount >= AUTOPILOT_TURN_LIMIT * 0.8 ? 'warning' : ''}`}
            >
              {autopilotTurnCount}/{AUTOPILOT_TURN_LIMIT}
            </span>
          )}
        </div>
        <div 
          className={`auto-approve-toggle ${!isAutopilotEnabled ? 'disabled' : ''}`} 
          title={isAutopilotEnabled 
            ? "Auto-Approve: Policy AI auto-approves compliant actions. Uncertain/non-compliant actions require human review." 
            : "Enable Autopilot first to use Auto-Approve"
          }
        >
          <span className="auto-approve-label">🛡️</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={isAutoApproveEnabled}
              onChange={(e) => setIsAutoApproveEnabled(e.target.checked)}
              disabled={!isAutopilotEnabled}
            />
            <span className="toggle-slider"></span>
          </label>
          {isAutoApproveEnabled && approvalLogCount > 0 && (
            <button 
              className="approval-log-btn"
              onClick={onViewApprovalLogs}
              title="View approval AI logs"
            >
              📋 {approvalLogCount}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChatHeader
