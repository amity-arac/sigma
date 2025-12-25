import { useState, useRef, useEffect, useCallback } from 'react'
import { useSession } from '../../context/SessionContext'
import { useToast } from '../../context/ToastContext'
import { parseAction, sendResponse, callTool, rollbackToPoint, regenerateUserResponse, regenerateAction, checkPolicyCompliance, editTrajectoryMessage } from '../../services/api'
import ChatHeader from './ChatHeader'
import StickyUserMessage from './StickyUserMessage'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import ActionSuggestion from './ActionSuggestion'
import LoadingIndicator from './LoadingIndicator'
import PromptDialog from '../common/PromptDialog'
import RegenerateActionDialog from '../common/RegenerateActionDialog'
import EditMessageDialog from '../common/EditMessageDialog'
import ConfirmDialog from '../common/ConfirmDialog'
import './ChatPanel.css'

function ChatPanel({ onSimulationEnd, onNewSession, wasOriginallyCompleted = false, onMarkIncomplete }) {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [pendingAction, setPendingAction] = useState(null)
  const [inputDisabled, setInputDisabled] = useState(false)
  const [showRegenerateDialog, setShowRegenerateDialog] = useState(false)
  const [regenerateTargetIndex, setRegenerateTargetIndex] = useState(null)
  const [pendingPolicyResult, setPendingPolicyResult] = useState(null)
  const [isCheckingPolicy, setIsCheckingPolicy] = useState(false)
  const [approvalLogs, setApprovalLogs] = useState([])
  // New state for regenerating agent actions
  const [showRegenerateActionDialog, setShowRegenerateActionDialog] = useState(false)
  const [rejectedAction, setRejectedAction] = useState(null)
  // Flag to skip auto-approval for regenerated actions (if user had to regenerate, auto-approval already failed)
  const [skipAutoApproval, setSkipAutoApproval] = useState(false)
  // State for editing messages
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [editTarget, setEditTarget] = useState(null) // { index, role, content, messageId }
  // State for completed trajectory warning
  const [showCompletedWarning, setShowCompletedWarning] = useState(false)
  const [pendingRegenerateAction, setPendingRegenerateAction] = useState(null) // { type: 'rollback'|'regenerateUser', index: number }
  
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
    isAutopilotEnabled,
    isAutoApproveEnabled
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
    setPendingPolicyResult(null)
    setInputDisabled(false)
    setSkipAutoApproval(false)  // Reset for next action

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
    // Track the rejected suggestion and show the regenerate dialog
    if (pendingAction) {
      addRejectedSuggestion({
        action_type: pendingAction.action_type,
        content: pendingAction.content,
        tool_name: pendingAction.tool_name,
        arguments: pendingAction.arguments,
        reasoning: pendingAction.reasoning
      })
      // Store the rejected action and show the regenerate dialog
      setRejectedAction({
        action_type: pendingAction.action_type,
        content: pendingAction.content,
        tool_name: pendingAction.tool_name,
        arguments: pendingAction.arguments,
        reasoning: pendingAction.reasoning
      })
      setShowRegenerateActionDialog(true)
    }
    setPendingAction(null)
    setPendingPolicyResult(null)
  }, [pendingAction, addRejectedSuggestion])

  // Handle regenerating agent action with feedback
  const handleRegenerateActionConfirm = useCallback(async (feedback) => {
    setShowRegenerateActionDialog(false)
    
    if (!sessionId || !rejectedAction) {
      setInputDisabled(false)
      setRejectedAction(null)
      return
    }

    showLoading('Regenerating action...')
    
    try {
      const newAction = await regenerateAction(sessionId, rejectedAction, feedback)
      hideLoading()
      // Set the new action as pending for approval
      // Skip auto-approval since the user had to manually reject - auto-approval already failed
      setSkipAutoApproval(true)
      setPendingAction(newAction)
      setRejectedAction(null)
    } catch (error) {
      hideLoading()
      setInputDisabled(false)
      setRejectedAction(null)
      showToast(error.message, 'error')
    }
  }, [sessionId, rejectedAction, showLoading, hideLoading, showToast])

  const handleRegenerateActionCancel = useCallback(() => {
    setShowRegenerateActionDialog(false)
    setInputDisabled(false)
    setRejectedAction(null)
    showToast('Action cancelled', 'info')
  }, [showToast])

  const handleSendMessage = useCallback(async (message) => {
    if (!message || !isSimulationActive) return

    setInputDisabled(true)
    showLoading('Thinking')
    // Clear any previous policy result and reset skip flag for fresh actions
    setPendingPolicyResult(null)
    setIsCheckingPolicy(false)
    setSkipAutoApproval(false)  // Reset since this is a new action, not regenerated

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

  // Auto-approve policy check: when a pending action is set and auto-approve is enabled
  useEffect(() => {
    // Skip if no pending action, auto-approve disabled, already checking, or this is a regenerated action
    // For regenerated actions, skip auto-approval since if it could catch the issue, the user wouldn't have had to regenerate
    if (!pendingAction || !isAutoApproveEnabled || isCheckingPolicy || pendingPolicyResult || skipAutoApproval) {
      return
    }

    let isCancelled = false
    
    const checkPolicy = async () => {
      console.log('[Auto-Approve] Starting policy check for action:', pendingAction.action_type)
      setIsCheckingPolicy(true)
      
      try {
        const result = await checkPolicyCompliance(sessionId, {
          action_type: pendingAction.action_type,
          content: pendingAction.content,
          tool_name: pendingAction.tool_name,
          arguments: pendingAction.arguments,
          reasoning: pendingAction.reasoning
        })
        
        if (isCancelled) {
          console.log('[Auto-Approve] Policy check cancelled')
          return
        }
        
        console.log('[Auto-Approve] Policy result:', result)
        setPendingPolicyResult(result)
        setIsCheckingPolicy(false)
        
        // Log this approval check
        setApprovalLogs(prev => [...prev, {
          timestamp: new Date().toISOString(),
          action: pendingAction,
          result
        }])
        
        // If approved with high confidence, auto-execute
        if (result.approved && result.confidence === 'high') {
          console.log('[Auto-Approve] High confidence approval - auto-executing')
          // Use a small delay to ensure state updates have propagated
          setTimeout(() => {
            if (!isCancelled) {
              executeApprovedAction(pendingAction)
            }
          }, 100)
        }
      } catch (error) {
        console.error('[Auto-Approve] Policy check failed:', error)
        if (!isCancelled) {
          setIsCheckingPolicy(false)
          // On error, just show the action without policy result
        }
      }
    }

    checkPolicy()
    
    return () => {
      isCancelled = true
    }
  }, [pendingAction, isAutoApproveEnabled, sessionId, executeApprovedAction])

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

  // Handle rollback to a specific message
  // When clicking regenerate on an agent message:
  // 1. Rollback to that state (by message ID for idempotency)
  // 2. Mark the message as rejected
  // 3. Show the regeneration dialog for user to provide feedback
  const handleRollback = useCallback(async (messageIndex) => {
    if (!sessionId || messageIndex < 1) return
    
    try {
      // Capture the message being rolled back (the one we're regenerating from)
      // This is the message at messageIndex that will be removed
      const targetMessage = messages[messageIndex]
      
      if (!targetMessage?.id) {
        showToast('Cannot rollback: message has no ID', 'error')
        return
      }
      
      showLoading('Rolling back...')
      // Use message_id for idempotent rollback operation
      const data = await rollbackToPoint(sessionId, targetMessage.id)
      hideLoading()
      
      if (data.success) {
        // Remove messages from the frontend state
        setMessages(prev => prev.slice(0, messageIndex))
        // Clear any pending action
        setPendingAction(null)
        
        if (!isSimulationActive) {
          setIsSimulationActive(true)
        }
        
        // Mark the message we're rolling back from as rejected and show regeneration dialog
        if (targetMessage && (targetMessage.role === 'agent' || targetMessage.role === 'tool')) {
          const rejectedInfo = {
            action_type: targetMessage.role === 'tool' ? 'tool_call' : 'respond',
            content: targetMessage.content,
            tool_name: targetMessage.tool_name || null,
            arguments: targetMessage.tool_arguments || null,
            reasoning: targetMessage.reasoning || null
          }
          
          // Add to rejected suggestions
          addRejectedSuggestion(rejectedInfo)
          
          // Store the rejected action and show the regenerate dialog
          setRejectedAction(rejectedInfo)
          setSkipAutoApproval(true)  // Skip auto-approval since user explicitly rejected
          setShowRegenerateActionDialog(true)
        } else {
          // If not an agent/tool message, just enable input
          setInputDisabled(false)
        }
        
        showToast(`Rolled back ${data.removed_count} message(s)`, 'success')
      } else {
        showToast(data.error || 'Cannot rollback', 'error')
      }
    } catch (error) {
      hideLoading()
      showToast('Failed to rollback: ' + error.message, 'error')
    }
  }, [sessionId, messages, isSimulationActive, setMessages, setIsSimulationActive, addRejectedSuggestion, showLoading, hideLoading, showToast])

  // Handle regenerate user response - opens dialog for additional note
  const handleRegenerateUserClick = useCallback((messageIndex) => {
    setRegenerateTargetIndex(messageIndex)
    setShowRegenerateDialog(true)
  }, [])

  // Execute the regenerate with optional note
  const handleRegenerateConfirm = useCallback(async (additionalNote) => {
    setShowRegenerateDialog(false)
    
    if (!sessionId || regenerateTargetIndex === null) return
    
    // Get the rejected message content before rollback
    const targetUserMessage = messages[regenerateTargetIndex]
    const rejectedMessageContent = targetUserMessage?.content || null
    
    try {
      showLoading('Regenerating user response...')
      
      // First, check if we need to rollback (if the target is not the last user message)
      // Find the index of the message right after the target user message
      const rollbackIndex = regenerateTargetIndex + 1
      
      // If there are messages after the target user message, rollback first using message ID
      if (rollbackIndex < messages.length) {
        const messageToRollbackFrom = messages[rollbackIndex]
        if (!messageToRollbackFrom?.id) {
          hideLoading()
          showToast('Cannot rollback: message has no ID', 'error')
          setRegenerateTargetIndex(null)
          return
        }
        
        const rollbackData = await rollbackToPoint(sessionId, messageToRollbackFrom.id)
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
      
      // Now regenerate the user response with the rejected message and feedback
      const data = await regenerateUserResponse(sessionId, rejectedMessageContent, additionalNote || null)
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

  // Handle edit message click - opens the edit dialog
  const handleEditMessageClick = useCallback((messageIndex, role, content) => {
    const message = messages[messageIndex]
    if (message) {
      setEditTarget({
        index: messageIndex,
        role: role,
        content: content,
        messageId: message.id
      })
      setShowEditDialog(true)
    }
  }, [messages])

  // Execute the edit
  const handleEditConfirm = useCallback(async (newContent) => {
    setShowEditDialog(false)
    
    if (!sessionId || !editTarget) {
      setEditTarget(null)
      return
    }

    try {
      showLoading('Saving edit...')
      
      // Call the API to edit the message
      await editTrajectoryMessage(sessionId, editTarget.messageId, newContent)
      
      hideLoading()
      
      // Update the message in frontend state
      setMessages(prev => {
        const newMessages = [...prev]
        if (newMessages[editTarget.index]) {
          newMessages[editTarget.index] = {
            ...newMessages[editTarget.index],
            content: newContent
          }
        }
        return newMessages
      })
      
      showToast('Message updated', 'success')
    } catch (error) {
      hideLoading()
      showToast('Failed to edit: ' + error.message, 'error')
    }
    
    setEditTarget(null)
  }, [sessionId, editTarget, setMessages, showLoading, hideLoading, showToast])

  const handleEditCancel = useCallback(() => {
    setShowEditDialog(false)
    setEditTarget(null)
  }, [])

  // Wrapper for rollback that handles completed trajectory warning
  const handleRollbackWithWarning = useCallback((messageIndex) => {
    if (wasOriginallyCompleted) {
      // Show warning dialog and store the pending action
      setPendingRegenerateAction({ type: 'rollback', index: messageIndex })
      setShowCompletedWarning(true)
    } else {
      // Proceed directly
      handleRollback(messageIndex)
    }
  }, [wasOriginallyCompleted, handleRollback])

  // Wrapper for regenerate user that handles completed trajectory warning
  const handleRegenerateUserWithWarning = useCallback((messageIndex) => {
    if (wasOriginallyCompleted) {
      // Show warning dialog and store the pending action
      setPendingRegenerateAction({ type: 'regenerateUser', index: messageIndex })
      setShowCompletedWarning(true)
    } else {
      // Proceed directly
      handleRegenerateUserClick(messageIndex)
    }
  }, [wasOriginallyCompleted, handleRegenerateUserClick])

  // Handle completed warning confirmation
  const handleCompletedWarningConfirm = useCallback(async () => {
    setShowCompletedWarning(false)
    
    if (!pendingRegenerateAction) return
    
    // First, mark the trajectory as incomplete
    if (onMarkIncomplete) {
      await onMarkIncomplete()
    }
    
    // Then proceed with the pending action
    if (pendingRegenerateAction.type === 'rollback') {
      handleRollback(pendingRegenerateAction.index)
    } else if (pendingRegenerateAction.type === 'regenerateUser') {
      handleRegenerateUserClick(pendingRegenerateAction.index)
    }
    
    setPendingRegenerateAction(null)
  }, [pendingRegenerateAction, onMarkIncomplete, handleRollback, handleRegenerateUserClick])

  // Handle completed warning cancel
  const handleCompletedWarningCancel = useCallback(() => {
    setShowCompletedWarning(false)
    setPendingRegenerateAction(null)
  }, [])

  return (
    <div className="chat-panel">
      <ChatHeader />
      
      {persona && (
        <StickyUserMessage content={persona} />
      )}
      
      <div className="chat-messages">
        <MessageList 
          messages={messages}
          onRollback={handleRollbackWithWarning}
          onRegenerateUser={handleRegenerateUserWithWarning}
          onEditMessage={handleEditMessageClick}
          onRemoveRejected={removeMessage}
          isSimulationActive={isSimulationActive}
          isConversationEnded={!isSimulationActive}
          onNewSession={onNewSession}
          wasOriginallyCompleted={wasOriginallyCompleted}
        />
        
        {pendingAction && (
          <ActionSuggestion
            action={pendingAction}
            onApprove={() => handleActionApprove(pendingAction)}
            onReject={handleActionReject}
            policyResult={pendingPolicyResult}
            isCheckingPolicy={isCheckingPolicy}
          />
        )}
        
        {isLoading && <LoadingIndicator text={loadingText} />}
        
        <div ref={messagesEndRef} />
      </div>
      
      <ChatInput
        onSend={handleSendMessage}
        onAutoGenerate={handleAutoGenerate}
        disabled={inputDisabled || !isSimulationActive || isCheckingPolicy}
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
      
      {showRegenerateActionDialog && rejectedAction && (
        <RegenerateActionDialog
          rejectedAction={rejectedAction}
          onConfirm={handleRegenerateActionConfirm}
          onCancel={handleRegenerateActionCancel}
        />
      )}

      {showEditDialog && editTarget && (
        <EditMessageDialog
          messageContent={editTarget.content}
          messageRole={editTarget.role}
          onConfirm={handleEditConfirm}
          onCancel={handleEditCancel}
        />
      )}

      {showCompletedWarning && (
        <ConfirmDialog
          title="⚠️ Modify Completed Trajectory"
          message="This trajectory is marked as complete. Regenerating will change its status to incomplete. Do you want to continue?"
          onConfirm={handleCompletedWarningConfirm}
          onCancel={handleCompletedWarningCancel}
        />
      )}
    </div>
  )
}

export default ChatPanel
