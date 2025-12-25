import { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'
import { saveTrajectory } from '../services/api'

const SessionContext = createContext(null)

// Cookie helpers
const getCookie = (name) => {
  const value = `; ${document.cookie}`
  const parts = value.split(`; ${name}=`)
  if (parts.length === 2) return parts.pop().split(';').shift()
  return null
}

const setCookie = (name, value, days = 365) => {
  const expires = new Date()
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000)
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;SameSite=Lax`
}

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(null)
  const [isSimulationActive, setIsSimulationActive] = useState(false)
  const [tools, setTools] = useState([])
  const [persona, setPersona] = useState('')
  const [wiki, setWiki] = useState('')
  const [injectedData, setInjectedData] = useState(null)  // Injected scenario data (augmented_data from persona_data)
  const [messages, setMessages] = useState([])
  const [stickyUserMessage, setStickyUserMessage] = useState('')
  const [finalResult, setFinalResult] = useState(null)
  
  // Initialize from cookies
  const [isAutopilotEnabled, setIsAutopilotEnabledState] = useState(() => {
    const saved = getCookie('sigma_autopilot')
    return saved === 'true'
  })
  const [isAutoApproveEnabled, setIsAutoApproveEnabledState] = useState(() => {
    const saved = getCookie('sigma_autoapprove')
    return saved === 'true'
  })
  
  // Wrappers that save to cookies
  const setIsAutopilotEnabled = useCallback((value) => {
    const newValue = typeof value === 'function' ? value(isAutopilotEnabled) : value
    setIsAutopilotEnabledState(newValue)
    setCookie('sigma_autopilot', String(newValue))
  }, [isAutopilotEnabled])
  
  const setIsAutoApproveEnabled = useCallback((value) => {
    const newValue = typeof value === 'function' ? value(isAutoApproveEnabled) : value
    setIsAutoApproveEnabledState(newValue)
    setCookie('sigma_autoapprove', String(newValue))
  }, [isAutoApproveEnabled])
  
  const [autopilotTurnCount, setAutopilotTurnCount] = useState(0)
  const [trajectoryId, setTrajectoryId] = useState(null)
  
  // Hard limit for autopilot turns to prevent infinite loops
  const AUTOPILOT_TURN_LIMIT = 30
  const [isAutoSaving, setIsAutoSaving] = useState(false)
  const [lastSaveTime, setLastSaveTime] = useState(null)
  
  // Ref to track if auto-save is needed
  const autoSaveTimeoutRef = useRef(null)
  const lastSavedMessagesRef = useRef(null)

  const addMessage = useCallback((role, content, options = {}) => {
    const { 
      isTemporary = false, 
      reasoning = null,
      // For tool calls
      toolName = null,
      toolArguments = null,
      // For rejected suggestions (role='rejected')
      rejectedActionType = null,
      rejectedContent = null,
      rejectedToolName = null,
      rejectedToolArguments = null,
      rejectedReasoning = null,
    } = typeof options === 'boolean' 
      ? { isTemporary: options } 
      : options
    
    const newMessage = {
      id: Date.now() + Math.random(),
      role,
      content,
      reasoning,
      isTemporary,
      timestamp: new Date().toISOString(),
      // For tool calls
      tool_name: toolName,
      tool_arguments: toolArguments,
      // For rejected suggestions
      rejected_action_type: rejectedActionType,
      rejected_content: rejectedContent,
      rejected_tool_name: rejectedToolName,
      rejected_tool_arguments: rejectedToolArguments,
      rejected_reasoning: rejectedReasoning,
    }
    
    // Debug: Log when tool messages are added
    if (role === 'tool') {
      console.log('[addMessage] Tool message created:', {
        id: newMessage.id,
        role: newMessage.role,
        tool_name: newMessage.tool_name,
        tool_arguments: newMessage.tool_arguments,
        optionsReceived: { toolName, toolArguments }
      })
    }
    
    setMessages(prev => [...prev, newMessage])
    
    // Set sticky message for first user message
    if (role === 'user' && !isTemporary) {
      setStickyUserMessage(prev => prev || content)
    }
    
    return newMessage.id
  }, [])

  // Add a rejected suggestion as an inline message
  const addRejectedSuggestion = useCallback((suggestion) => {
    const rejectedMessage = {
      id: Date.now() + Math.random(),
      role: 'rejected',
      content: `Rejected: ${suggestion.action_type === 'respond' ? 'Response' : 'Tool call'}`,
      reasoning: null,
      isTemporary: false,
      timestamp: new Date().toISOString(),
      // Rejected suggestion in same format as normal message
      rejected: {
        content: suggestion.content,
        reasoning: suggestion.reasoning,
        tool_name: suggestion.tool_name,
        tool_arguments: suggestion.arguments,
      },
    }
    setMessages(prev => [...prev, rejectedMessage])
    return rejectedMessage.id
  }, [])

  const removeMessage = useCallback((messageId) => {
    setMessages(prev => prev.filter(m => m.id !== messageId))
  }, [])

  const removeLastMessages = useCallback((count) => {
    setMessages(prev => prev.slice(0, -count))
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setStickyUserMessage('')
  }, [])

  const resetSession = useCallback(() => {
    // Clear any pending auto-save
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current)
      autoSaveTimeoutRef.current = null
    }
    setSessionId(null)
    setIsSimulationActive(false)
    setTools([])
    setPersona('')
    setWiki('')
    setInjectedData(null)
    setMessages([])
    setStickyUserMessage('')
    setFinalResult(null)
    // Don't reset autopilot/auto-approve - they're persisted in cookies
    setAutopilotTurnCount(0)
    setTrajectoryId(null)
    setIsAutoSaving(false)
    setLastSaveTime(null)
    lastSavedMessagesRef.current = null
    lastSavedStateRef.current = null
  }, [])

  // Ref to track last saved state (messages + finalResult)
  const lastSavedStateRef = useRef(null)

  // Mark current messages as already saved (used when restoring a trajectory)
  const markMessagesSaved = useCallback((messagesToMark, savedFinalResult = null) => {
    const nonTempMessages = messagesToMark.filter(m => !m.isTemporary)
    const messagesKey = JSON.stringify(nonTempMessages.map(m => m.id))
    lastSavedMessagesRef.current = messagesKey
    // Also update the state ref to prevent immediate re-save
    lastSavedStateRef.current = JSON.stringify({
      messageIds: nonTempMessages.map(m => m.id),
      isDone: savedFinalResult?.is_done || false,
      reward: savedFinalResult?.reward
    })
    console.log('[markMessagesSaved] Marked', nonTempMessages.length, 'messages as saved, isDone:', savedFinalResult?.is_done)
  }, [])

  // Auto-save effect - debounced save whenever messages or finalResult change
  useEffect(() => {
    // Don't auto-save if no session or no messages
    if (!sessionId || messages.length === 0) {
      return
    }
    
    // Don't auto-save temporary messages only
    const nonTempMessages = messages.filter(m => !m.isTemporary)
    if (nonTempMessages.length === 0) {
      return
    }
    
    // Build a state key that includes both messages AND finalResult
    // This ensures we save when simulation completes (finalResult changes)
    const stateKey = JSON.stringify({
      messageIds: nonTempMessages.map(m => m.id),
      isDone: finalResult?.is_done || false,
      reward: finalResult?.reward
    })
    
    // Check if state actually changed
    if (lastSavedStateRef.current === stateKey) {
      return
    }
    
    // Clear existing timeout
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current)
    }
    
    // Capture current values for the async callback
    const currentFinalResult = finalResult
    const currentStateKey = stateKey
    
    // Debounce: save after 2 seconds of no changes
    autoSaveTimeoutRef.current = setTimeout(async () => {
      setIsAutoSaving(true)
      try {
        // Use sessionId which is now the same as trajectoryId
        const result = await saveTrajectory(
          sessionId,
          nonTempMessages,
          currentFinalResult || {}
        )
        
        if (result.success) {
          setTrajectoryId(result.trajectory_id)
          setLastSaveTime(new Date())
          lastSavedStateRef.current = currentStateKey
          // Also update message ref for backward compatibility
          lastSavedMessagesRef.current = JSON.stringify(nonTempMessages.map(m => m.id))
          console.log('[Auto-save] Trajectory saved:', result.trajectory_id, 'isDone:', currentFinalResult?.is_done)
        } else {
          console.error('[Auto-save] Failed:', result.error)
        }
      } catch (error) {
        console.error('[Auto-save] Error:', error.message)
      } finally {
        setIsAutoSaving(false)
      }
    }, 2000)
    
    // Cleanup on unmount or when dependencies change
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current)
      }
    }
  }, [sessionId, messages, finalResult])

  const value = {
    sessionId,
    setSessionId,
    isSimulationActive,
    setIsSimulationActive,
    tools,
    setTools,
    persona,
    setPersona,
    wiki,
    setWiki,
    injectedData,
    setInjectedData,
    messages,
    setMessages,
    stickyUserMessage,
    setStickyUserMessage,
    finalResult,
    setFinalResult,
    addMessage,
    addRejectedSuggestion,
    removeMessage,
    removeLastMessages,
    clearMessages,
    resetSession,
    markMessagesSaved,
    isAutopilotEnabled,
    setIsAutopilotEnabled,
    isAutoApproveEnabled,
    setIsAutoApproveEnabled,
    autopilotTurnCount,
    setAutopilotTurnCount,
    AUTOPILOT_TURN_LIMIT,
    // Auto-save state
    trajectoryId,
    setTrajectoryId,
    isAutoSaving,
    lastSaveTime
  }

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  )
}

export function useSession() {
  const context = useContext(SessionContext)
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider')
  }
  return context
}