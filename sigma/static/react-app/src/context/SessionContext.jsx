import { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'
import { saveTrajectory } from '../services/api'

const SessionContext = createContext(null)

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(null)
  const [isSimulationActive, setIsSimulationActive] = useState(false)
  const [tools, setTools] = useState([])
  const [persona, setPersona] = useState('')
  const [wiki, setWiki] = useState('')
  const [messages, setMessages] = useState([])
  const [stickyUserMessage, setStickyUserMessage] = useState('')
  const [finalResult, setFinalResult] = useState(null)
  const [isAutopilotEnabled, setIsAutopilotEnabled] = useState(true)
  const [trajectoryId, setTrajectoryId] = useState(null)
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
    setMessages([])
    setStickyUserMessage('')
    setFinalResult(null)
    setIsAutopilotEnabled(false)
    setTrajectoryId(null)
    setIsAutoSaving(false)
    setLastSaveTime(null)
    lastSavedMessagesRef.current = null
  }, [])

  // Mark current messages as already saved (used when restoring a trajectory)
  const markMessagesSaved = useCallback((messagesToMark) => {
    const nonTempMessages = messagesToMark.filter(m => !m.isTemporary)
    const messagesKey = JSON.stringify(nonTempMessages.map(m => m.id))
    lastSavedMessagesRef.current = messagesKey
    console.log('[markMessagesSaved] Marked', nonTempMessages.length, 'messages as saved')
  }, [])

  // Auto-save effect - debounced save whenever messages change
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
    
    // Check if messages actually changed (compare stringified versions)
    const messagesKey = JSON.stringify(nonTempMessages.map(m => m.id))
    if (lastSavedMessagesRef.current === messagesKey) {
      return
    }
    
    // Clear existing timeout
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current)
    }
    
    // Debounce: save after 2 seconds of no changes
    autoSaveTimeoutRef.current = setTimeout(async () => {
      setIsAutoSaving(true)
      try {
        // Use sessionId which is now the same as trajectoryId
        const result = await saveTrajectory(
          sessionId,
          nonTempMessages,
          finalResult || {}
        )
        
        if (result.success) {
          setTrajectoryId(result.trajectory_id)
          setLastSaveTime(new Date())
          lastSavedMessagesRef.current = messagesKey
          console.log('[Auto-save] Trajectory saved:', result.trajectory_id)
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