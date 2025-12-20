import { useState, useRef, useEffect, useCallback } from 'react'
import { useSession } from '../../context/SessionContext'
import { useToast } from '../../context/ToastContext'
import { parseAction, sendResponse, callTool, rollbackToPoint, regenerateUserResponse } from '../../services/api'
import ChatHeader from './ChatHeader'
import StickyUserMessage from './StickyUserMessage'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import ActionSuggestion from './ActionSuggestion'
import LoadingIndicator from './LoadingIndicator'
import PromptDialog from '../common/PromptDialog'
import './ChatPanel.css'

function ChatPanel({ onSimulationEnd, onNewSession }) {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [pendingAction, setPendingAction] = useState(null)
  const [inputDisabled, setInputDisabled] = useState(false)
  const [showRegenerateDialog, setShowRegenerateDialog] = useState(false)
  const [regenerateTargetIndex, setRegenerateTargetIndex] = useState(null)
  
  const messagesEndRef = useRef(null)
  
  const { 
    sessionId, 
    isSimulationActive, 
    setIsSimulationActive,
    messages, 
    setMessages,
    addMessage,
    persona,
    addRejectedSuggestion,
    setFinalResult,
    isAutopilotEnabled
  } = useSession()
  const { showToast } = useToast()

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const showLoading = useCallback((text) => {
    setIsLoading(true)
    setLoadingText(text)
  }, [])

  const hideLoading = useCallback(() => {
    setIsLoading(false)
    setLoadingText('')
  }, [])

  const handleSimulationEnd = useCallback((data) => {
    setIsSimulationActive(false)
    setInputDisabled(true)
    setFinalResult({
      is_done: data.done,
      reward: data.reward,
      reward_info: data.reward_info
    })
    onSimulationEnd(data)
    showToast('Simulation completed!', 'success')
  }, [setIsSimulationActive, setFinalResult, onSimulationEnd, showToast])

  const handleToolCall = useCallback(async (toolName, args, reasoning) => {
    addMessage('tool', `🔧 Calling ${toolName}\n${JSON.stringify(args, null, 2)}`, { 
      reasoning,
      toolName: toolName,
      toolArguments: args
    })
    showLoading('Executing tool')

    try {
      const data = await callTool(sessionId, toolName, args)
      hideLoading()
      addMessage('tool-result', data.observation, {
        toolName: toolName
      })

      if (data.done) {
        handleSimulationEnd(data)
      }
    } catch (error) {
      hideLoading()
      showToast(error.message, 'error')
    }
  }, [sessionId, addMessage, showLoading, hideLoading, handleSimulationEnd, showToast])

  const handleActionResult = useCallback((data) => {
    if (data.done) {
      handleSimulationEnd(data)
    } else if (data.observation) {
      addMessage('user', data.observation)
    }
  }, [handleSimulationEnd, addMessage])

  const executeApprovedAction = useCallback(async (action) => {
    setPendingAction(null)
    setInputDisabled(false)

    if (action.action_type === 'respond') {
      addMessage('agent', action.content, { reasoning: action.reasoning })
      showLoading('User is responding')

      try {
        const data = await sendResponse(sessionId, action.content)
        hideLoading()
        handleActionResult(data)
      } catch (error) {
        hideLoading()
        showToast(error.message, 'error')
      }
    } else if (action.action_type === 'tool_call') {
      await handleToolCall(action.tool_name, action.arguments, action.reasoning)
    }
  }, [sessionId, addMessage, showLoading, hideLoading, handleActionResult, handleToolCall, showToast])

  const handleActionApprove = useCallback((action) => {
    executeApprovedAction(action)
  }, [executeApprovedAction])

  const handleActionReject = useCallback(() => {
    // Track the rejected suggestion
    if (pendingAction) {
      addRejectedSuggestion({
        action_type: pendingAction.action_type,
        content: pendingAction.content,
        tool_name: pendingAction.tool_name,
        arguments: pendingAction.arguments,
        reasoning: pendingAction.reasoning
      })
    }
    setPendingAction(null)
    setInputDisabled(false)
    showToast('Action cancelled', 'info')
  }, [pendingAction, addRejectedSuggestion, showToast])

  const handleSendMessage = useCallback(async (message) => {
    if (!message || !isSimulationActive) return

    setInputDisabled(true)
    showLoading('Thinking')

    try {
      const parsedAction = await parseAction(sessionId, message)
      hideLoading()
      setPendingAction(parsedAction)
    } catch (error) {
      hideLoading()
      setInputDisabled(false)
      showToast(error.message, 'error')
    }
  }, [sessionId, isSimulationActive, showLoading, hideLoading, showToast])

  const handleAutoGenerate = useCallback(async () => {
    const autoPrompt = "Based on the conversation history, user's request, and the policy/wiki guidelines, determine and execute the most appropriate next action. Follow the standard operating procedures."
    await handleSendMessage(autoPrompt)
  }, [handleSendMessage])

  // Autopilot: automatically trigger auto-generate when user/tool-result message arrives
  useEffect(() => {
    if (!isAutopilotEnabled || !isSimulationActive || isLoading || pendingAction) {
      return
    }
    
    // Check the last message to see if we should trigger auto
    const lastMessage = messages[messages.length - 1]
    if (!lastMessage) return
    
    // Trigger auto when the last message is from user or tool-result
    if (lastMessage.role === 'user' || lastMessage.role === 'tool-result') {
      // Small delay to let the UI update
      const timer = setTimeout(() => {
        handleAutoGenerate()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [messages, isAutopilotEnabled, isSimulationActive, isLoading, pendingAction, handleAutoGenerate])

  // Handle rollback to a specific message index
  const handleRollback = useCallback(async (messageIndex) => {
    if (!sessionId || messageIndex < 1) return
    
    try {
      showLoading('Rolling back...')
      // The backend expects the index in the conversation history
      // We need to map the frontend message index to backend index
      // Frontend messages include the initial user message at index 0
      const data = await rollbackToPoint(sessionId, messageIndex)
      hideLoading()
      
      if (data.success) {
        // Remove messages from the frontend state
        setMessages(prev => prev.slice(0, messageIndex))
        // Clear any pending action so user can regenerate
        setPendingAction(null)
        setInputDisabled(false)
        if (!isSimulationActive) {
          setIsSimulationActive(true)
        }
        showToast(`Rolled back ${data.removed_count} message(s)`, 'success')
      } else {
        showToast(data.error || 'Cannot rollback', 'error')
      }
    } catch (error) {
      hideLoading()
      showToast('Failed to rollback: ' + error.message, 'error')
    }
  }, [sessionId, isSimulationActive, setMessages, setIsSimulationActive, showLoading, hideLoading, showToast])

  // Handle regenerate user response - opens dialog for additional note
  const handleRegenerateUserClick = useCallback((messageIndex) => {
    setRegenerateTargetIndex(messageIndex)
    setShowRegenerateDialog(true)
  }, [])

  // Execute the regenerate with optional note
  const handleRegenerateConfirm = useCallback(async (additionalNote) => {
    setShowRegenerateDialog(false)
    
    if (!sessionId || regenerateTargetIndex === null) return
    
    try {
      showLoading('Regenerating user response...')
      
      // First, check if we need to rollback (if the target is not the last user message)
      // Find the index of the message right after the target user message
      const rollbackIndex = regenerateTargetIndex + 1
      
      // If there are messages after the target user message, rollback first
      if (rollbackIndex < messages.length) {
        const rollbackData = await rollbackToPoint(sessionId, rollbackIndex)
        if (!rollbackData.success) {
          hideLoading()
          showToast(rollbackData.error || 'Cannot rollback before regenerating', 'error')
          setRegenerateTargetIndex(null)
          return
        }
        // Update frontend state after rollback
        setMessages(prev => prev.slice(0, rollbackIndex))
        // Clear any pending action
        setPendingAction(null)
        setInputDisabled(false)
      }
      
      // Now regenerate the user response
      const data = await regenerateUserResponse(sessionId, additionalNote || null)
      hideLoading()
      
      if (data.success) {
        // Update the last user message in frontend state (which is now the target)
        setMessages(prev => {
          const newMessages = [...prev]
          // Find the last user message and update it
          for (let i = newMessages.length - 1; i >= 0; i--) {
            if (newMessages[i].role === 'user') {
              newMessages[i] = {
                ...newMessages[i],
                content: data.observation,
                id: Date.now() + Math.random() // New ID for re-render
              }
              break
            }
          }
          return newMessages
        })
        if (!isSimulationActive) {
          setIsSimulationActive(true)
        }
        showToast('User response regenerated', 'success')
      } else {
        showToast(data.error || 'Cannot regenerate', 'error')
      }
    } catch (error) {
      hideLoading()
      showToast('Failed to regenerate: ' + error.message, 'error')
    }
    
    setRegenerateTargetIndex(null)
  }, [sessionId, regenerateTargetIndex, messages, isSimulationActive, setMessages, setIsSimulationActive, showLoading, hideLoading, showToast])

  const handleRegenerateCancel = useCallback(() => {
    setShowRegenerateDialog(false)
    setRegenerateTargetIndex(null)
  }, [])

  return (
    <div className="chat-panel">
      <ChatHeader
        onNewSession={onNewSession}
      />
      
      {persona && (
        <StickyUserMessage content={persona} />
      )}
      
      <div className="chat-messages">
        <MessageList 
          messages={messages}
          onRollback={handleRollback}
          onRegenerateUser={handleRegenerateUserClick}
          isSimulationActive={isSimulationActive}
        />
        
        {pendingAction && (
          <ActionSuggestion
            action={pendingAction}
            onApprove={() => handleActionApprove(pendingAction)}
            onReject={handleActionReject}
          />
        )}
        
        {isLoading && <LoadingIndicator text={loadingText} />}
        
        <div ref={messagesEndRef} />
      </div>
      
      <ChatInput
        onSend={handleSendMessage}
        onAutoGenerate={handleAutoGenerate}
        disabled={inputDisabled || !isSimulationActive}
      />
      
      {showRegenerateDialog && (
        <PromptDialog
          title="🔄 Regenerate User Response"
          message="Add optional guidance for the user agent (e.g., 'be more direct', 'follow the persona more closely'):"
          placeholder="Additional guidance (optional)"
          onConfirm={handleRegenerateConfirm}
          onCancel={handleRegenerateCancel}
          allowEmpty={true}
        />
      )}
    </div>
  )
}

export default ChatPanel
