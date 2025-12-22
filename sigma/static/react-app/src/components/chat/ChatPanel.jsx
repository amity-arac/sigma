import { useState, useRef, useEffect, useCallback } from 'react'
import { useSession } from '../../context/SessionContext'
import { useToast } from '../../context/ToastContext'
import { parseAction, sendResponse, callTool, rollbackToPoint, regenerateUserResponse, regenerateAction } from '../../services/api'
import ChatHeader from './ChatHeader'
import StickyUserMessage from './StickyUserMessage'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import ActionSuggestion from './ActionSuggestion'
import LoadingIndicator from './LoadingIndicator'
import RegenerateUserDialog from '../common/RegenerateUserDialog'
import RegenerateActionDialog from '../common/RegenerateActionDialog'
import './ChatPanel.css'

function ChatPanel({ onSimulationEnd, onNewSession }) {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [pendingAction, setPendingAction] = useState(null)
  const [inputDisabled, setInputDisabled] = useState(false)
  const [showRegenerateUserDialog, setShowRegenerateUserDialog] = useState(false)
  const [regenerateTargetIndex, setRegenerateTargetIndex] = useState(null)
  const [rejectedUserMessage, setRejectedUserMessage] = useState(null)
  const [showRegenerateActionDialog, setShowRegenerateActionDialog] = useState(false)
  const [rejectedActionForRegenerate, setRejectedActionForRegenerate] = useState(null)
  
  const messagesEndRef = useRef(null)
  
  const { 
    sessionId, 
    isSimulationActive, 
    setIsSimulationActive,
    messages, 
    setMessages,
    addMessage,
    removeMessage,
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
    // Store the rejected action and show regenerate dialog
    if (pendingAction) {
      setRejectedActionForRegenerate({
        action_type: pendingAction.action_type,
        content: pendingAction.content,
        tool_name: pendingAction.tool_name,
        arguments: pendingAction.arguments,
        reasoning: pendingAction.reasoning
      })
      setPendingAction(null)  // Clear immediately so it doesn't show during regen
      setShowRegenerateActionDialog(true)
    }
  }, [pendingAction])

  // Handle regenerate action with feedback
  const handleRegenerateActionConfirm = useCallback(async (feedback) => {
    setShowRegenerateActionDialog(false)
    
    if (!sessionId || !rejectedActionForRegenerate) {
      setInputDisabled(false)
      setRejectedActionForRegenerate(null)
      return
    }
    
    // Don't add rejected suggestion to messages during regeneration
    // The rejected action is already sent to the backend for context
    
    showLoading('Regenerating action...')
    
    try {
      const newAction = await regenerateAction(sessionId, rejectedActionForRegenerate, feedback)
      hideLoading()
      setPendingAction(newAction)
      setRejectedActionForRegenerate(null)
    } catch (error) {
      hideLoading()
      setInputDisabled(false)
      setRejectedActionForRegenerate(null)
      showToast(error.message, 'error')
    }
  }, [sessionId, rejectedActionForRegenerate, showLoading, hideLoading, showToast])

  const handleRegenerateActionCancel = useCallback(() => {
    setShowRegenerateActionDialog(false)
    setInputDisabled(false)
    setRejectedActionForRegenerate(null)
    showToast('Action cancelled', 'info')
  }, [showToast])

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

  // Auto-generate user response on page load if last message is from agent
  // This handles the case where user closed page during user regeneration
  const hasCheckedLastAgentMessage = useRef(false)
  useEffect(() => {
    // Only run this check once per session
    if (hasCheckedLastAgentMessage.current) return
    if (!sessionId || !isSimulationActive || isLoading || pendingAction) {
      return
    }
    
    const lastMessage = messages[messages.length - 1]
    if (!lastMessage) return
    
    // If last message is agent (text response, not tool call), generate user response
    if (lastMessage.role === 'agent') {
      hasCheckedLastAgentMessage.current = true
      const timer = setTimeout(async () => {
        showLoading('Generating user response...')
        try {
          const data = await regenerateUserResponse(sessionId, null, null)
          hideLoading()
          if (data.success) {
            addMessage('user', data.observation)
          }
        } catch (error) {
          hideLoading()
          showToast('Failed to generate user response: ' + error.message, 'error')
        }
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [sessionId, messages, isSimulationActive, isLoading, pendingAction, showLoading, hideLoading, addMessage, showToast])

  // Handle rollback to a specific message index
  const handleRollback = useCallback(async (messageIndex) => {
    if (!sessionId || messageIndex < 1) return
    
    try {
      showLoading('Rolling back...')
      
      // Calculate the backend index by excluding rejected messages
      // Rejected messages only exist in frontend, not in backend conversation history
      let backendIndex = 0
      for (let i = 0; i < messageIndex; i++) {
        if (messages[i].role !== 'rejected') {
          backendIndex++
        }
      }
      
      // Call backend with the corrected index
      const data = await rollbackToPoint(sessionId, backendIndex)
      hideLoading()
      
      if (data.success) {
        // Remove messages from the frontend state (including any rejected messages after this point)
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
  }, [sessionId, messages, isSimulationActive, setMessages, setIsSimulationActive, showLoading, hideLoading, showToast])

  // Handle regenerate user response - opens dialog with the current message
  const handleRegenerateUserClick = useCallback((messageIndex) => {
    const targetMessage = messages[messageIndex]
    if (targetMessage && targetMessage.role === 'user') {
      setRejectedUserMessage(targetMessage.content)
      setRegenerateTargetIndex(messageIndex)
      setShowRegenerateUserDialog(true)
    }
  }, [messages])

  // Execute the regenerate with rejected message + feedback
  const handleRegenerateUserConfirm = useCallback(async (feedback) => {
    setShowRegenerateUserDialog(false)
    
    if (!sessionId || regenerateTargetIndex === null) {
      setRegenerateTargetIndex(null)
      setRejectedUserMessage(null)
      return
    }
    
    try {
      showLoading('Regenerating user response...')
      
      // First, check if we need to rollback (if the target is not the last user message)
      // Find the index of the message right after the target user message
      const rollbackIndex = regenerateTargetIndex + 1
      
      // Calculate backend index excluding rejected messages
      let backendRollbackIndex = 0
      for (let i = 0; i < rollbackIndex; i++) {
        if (messages[i].role !== 'rejected') {
          backendRollbackIndex++
        }
      }
      
      // If there are messages after the target user message, rollback first
      if (rollbackIndex < messages.length) {
        const rollbackData = await rollbackToPoint(sessionId, backendRollbackIndex)
        if (!rollbackData.success) {
          hideLoading()
          showToast(rollbackData.error || 'Cannot rollback before regenerating', 'error')
          setRegenerateTargetIndex(null)
          setRejectedUserMessage(null)
          return
        }
        // Update frontend state after rollback
        setMessages(prev => prev.slice(0, rollbackIndex))
        // Clear any pending action
        setPendingAction(null)
        setInputDisabled(false)
      }
      
      // Now regenerate the user response with the rejected message and feedback
      const data = await regenerateUserResponse(sessionId, rejectedUserMessage, feedback)
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
    setRejectedUserMessage(null)
  }, [sessionId, regenerateTargetIndex, rejectedUserMessage, messages, isSimulationActive, setMessages, setIsSimulationActive, showLoading, hideLoading, showToast])

  const handleRegenerateUserCancel = useCallback(() => {
    setShowRegenerateUserDialog(false)
    setRegenerateTargetIndex(null)
    setRejectedUserMessage(null)
  }, [])

  // Handle removing a rejected message
  const handleRemoveRejected = useCallback((messageId) => {
    removeMessage(messageId)
    showToast('Rejected action removed from trajectory', 'info')
  }, [removeMessage, showToast])

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
          onRemoveRejected={handleRemoveRejected}
          isSimulationActive={isSimulationActive}
          isConversationEnded={!isSimulationActive}
          onNewSession={onNewSession}
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
      
      {showRegenerateUserDialog && rejectedUserMessage && (
        <RegenerateUserDialog
          rejectedMessage={rejectedUserMessage}
          onConfirm={handleRegenerateUserConfirm}
          onCancel={handleRegenerateUserCancel}
        />
      )}
      
      {showRegenerateActionDialog && rejectedActionForRegenerate && (
        <RegenerateActionDialog
          rejectedAction={rejectedActionForRegenerate}
          onConfirm={handleRegenerateActionConfirm}
          onCancel={handleRegenerateActionCancel}
        />
      )}
    </div>
  )
}

export default ChatPanel
