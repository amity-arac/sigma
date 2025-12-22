import { useState, useCallback } from 'react'
import { useSession } from '../context/SessionContext'
import { useToast } from '../context/ToastContext'
import ChatPanel from './chat/ChatPanel'
import SidePanel from './sidebar/SidePanel'
import MobileInfoPanel from './chat/MobileInfoPanel'
import ConfirmDialog from './common/ConfirmDialog'
import './MainContent.css'

function MainContent({ onSimulationEnd, onNewSession }) {
  const [showConfirmDialog, setShowConfirmDialog] = useState(false)
  const [confirmDialogConfig, setConfirmDialogConfig] = useState({})
  
  const { 
    messages,
    clearMessages,
    resetSession
  } = useSession()
  const { showToast } = useToast()

  const handleNewSession = useCallback(() => {
    if (messages.length > 0) {
      setConfirmDialogConfig({
        title: '✨ Start New Trajectory',
        message: 'Are you sure you want to start a new trajectory? All current progress will be lost.',
        onConfirm: () => {
          clearMessages()
          resetSession()
          onNewSession()
          setShowConfirmDialog(false)
        },
        onCancel: () => setShowConfirmDialog(false)
      })
      setShowConfirmDialog(true)
    } else {
      clearMessages()
      resetSession()
      onNewSession()
    }
  }, [messages, clearMessages, resetSession, onNewSession])

  return (
    <>
      <div className="main-content">
        <ChatPanel 
          onSimulationEnd={onSimulationEnd}
          onNewSession={handleNewSession}
        />
        <SidePanel />
      </div>
      
      {/* Mobile info panel - shows tools, persona, wiki, settings on mobile */}
      <MobileInfoPanel onNewSession={handleNewSession} />
      
      {showConfirmDialog && (
        <ConfirmDialog
          title={confirmDialogConfig.title}
          message={confirmDialogConfig.message}
          onConfirm={confirmDialogConfig.onConfirm}
          onCancel={confirmDialogConfig.onCancel}
        />
      )}
    </>
  )
}

export default MainContent
