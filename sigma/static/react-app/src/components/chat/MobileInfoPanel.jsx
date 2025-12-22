import { useState, useRef, useEffect } from 'react'
import { useSession } from '../../context/SessionContext'
import { generateResponse } from '../../services/api'
import ToolsList from '../sidebar/ToolsList'
import './MobileInfoPanel.css'

function MobileInfoPanel({ onNewSession }) {
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('wiki')  // Default to policy tab
  const [policyQuestion, setPolicyQuestion] = useState('')
  const [isAskingPolicy, setIsAskingPolicy] = useState(false)
  const [qaTrayExpanded, setQaTrayExpanded] = useState(false)
  const [qaConversation, setQaConversation] = useState([])  // Array of {question, answer}
  const qaScrollRef = useRef(null)
  const qaInputRef = useRef(null)
  const { 
    tools, 
    persona, 
    wiki,
    isAutopilotEnabled,
    setIsAutopilotEnabled,
    trajectoryId,
    sessionId,
    isAutoSaving,
    lastSaveTime
  } = useSession()

  // Auto-scroll to bottom of Q&A when new answer arrives
  useEffect(() => {
    if (qaScrollRef.current && qaConversation.length > 0) {
      qaScrollRef.current.scrollTop = qaScrollRef.current.scrollHeight
    }
  }, [qaConversation])

  const expandAndFocusInput = () => {
    setQaTrayExpanded(true)
    // Focus after a short delay to allow expansion animation
    setTimeout(() => {
      qaInputRef.current?.focus()
    }, 100)
  }

  const tabs = [
    { id: 'wiki', label: '📖 Policy', icon: '📖' },
    { id: 'persona', label: '👤 Persona', icon: '👤' },
    { id: 'tools', label: '🔧 Tools', icon: '🔧' },
    { id: 'settings', label: '⚙️ Settings', icon: '⚙️' },
  ]

  const handleAskPolicy = async () => {
    if (!policyQuestion.trim() || !sessionId) return
    
    const currentQuestion = policyQuestion.trim()
    setIsAskingPolicy(true)
    setPolicyQuestion('')
    setQaTrayExpanded(true)  // Auto-expand tray when asking
    
    // Add question to conversation immediately
    setQaConversation(prev => [...prev, { question: currentQuestion, answer: null, isLoading: true }])
    
    try {
      const prompt = `You are a Policy Support AI assisting a customer service agent. Your role is to help the agent understand and apply company policies correctly.

Here is the company policy document:

---POLICY DOCUMENT---
${wiki}
---END POLICY DOCUMENT---

Agent's Question: ${currentQuestion}

Provide a clear, actionable answer. IMPORTANT: Always cite the specific policy section or rule you're basing your answer on by quoting the relevant text from the policy. Format your response as:

1. Answer: [Your direct answer]
2. Policy Basis: [Quote the exact policy text that supports this answer]`
      const result = await generateResponse(sessionId, prompt)
      const answer = result.response || 'Unable to generate answer'
      
      // Update the last conversation item with the answer
      setQaConversation(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { question: currentQuestion, answer, isLoading: false }
        return updated
      })
    } catch (error) {
      setQaConversation(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { question: currentQuestion, answer: `Error: ${error.message}`, isLoading: false }
        return updated
      })
    } finally {
      setIsAskingPolicy(false)
    }
  }

  const formatTimeAgo = (date) => {
    if (!date) return ''
    const seconds = Math.floor((new Date() - date) / 1000)
    if (seconds < 60) return 'just now'
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ago`
  }

  const getAutoSaveStatus = () => {
    if (isAutoSaving) return { icon: '💾', text: 'Saving...' }
    if (trajectoryId && lastSaveTime) return { icon: '✓', text: `Saved ${formatTimeAgo(lastSaveTime)}` }
    return { icon: '○', text: 'Unsaved' }
  }

  return (
    <>
      {/* Toggle button - visible on mobile */}
      <button 
        className="mobile-info-toggle"
        onClick={() => setIsOpen(true)}
        aria-label="Show info panel"
      >
        <span className="info-icon">ℹ️</span>
      </button>

      {/* Backdrop */}
      {isOpen && (
        <div 
          className="mobile-info-backdrop"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Slide-in panel */}
      <div className={`mobile-info-panel ${isOpen ? 'open' : ''}`}>
        <div className="mobile-info-header">
          <h3>Context Info</h3>
          <button 
            className="close-btn"
            onClick={() => setIsOpen(false)}
            aria-label="Close panel"
          >
            ✕
          </button>
        </div>

        {/* Tab navigation */}
        <div className="mobile-info-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span className="tab-label">{tab.label.split(' ')[1]}</span>
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="mobile-info-content">
          {activeTab === 'settings' && (
            <div className="info-section">
              <div className="mobile-settings">
                {/* Autopilot Toggle */}
                <div className="mobile-setting-row">
                  <span className="setting-label">🚀 Autopilot</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={isAutopilotEnabled}
                      onChange={(e) => setIsAutopilotEnabled(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                <p className="setting-hint">Auto-triggers action after each user/tool response</p>

                {/* Save Status */}
                <div className="mobile-setting-row">
                  <span className="setting-label">💾 Save Status</span>
                  <span className="save-status">
                    {getAutoSaveStatus().icon} {getAutoSaveStatus().text}
                  </span>
                </div>
                {trajectoryId && (
                  <p className="setting-hint">ID: {trajectoryId.slice(0, 12)}...</p>
                )}

                {/* New Trajectory Button */}
                {onNewSession && (
                  <button 
                    className="btn btn-new-trajectory-mobile"
                    onClick={() => { setIsOpen(false); onNewSession(); }}
                  >
                    ✨ New Trajectory
                  </button>
                )}
              </div>
            </div>
          )}

          {activeTab === 'persona' && (
            <div className="info-section">
              <div className="persona-display">
                {persona || 'No persona loaded'}
              </div>
            </div>
          )}

          {activeTab === 'tools' && (
            <div className="info-section">
              <ToolsList tools={tools} />
            </div>
          )}

          {activeTab === 'wiki' && (
            <div className="info-section wiki-section">
              {/* Scrollable Policy Content */}
              <div className="wiki-content-wrapper">
                <div className="wiki-content">
                  {wiki || 'No wiki loaded'}
                </div>
              </div>
              
              {/* Slide-up Q&A Tray */}
              {wiki && sessionId && (
                <div className={`policy-qa-tray ${qaTrayExpanded ? 'expanded' : ''}`}>
                  {/* Collapsed State - Shows compose bar */}
                  {!qaTrayExpanded && (
                    <div className="policy-qa-collapsed" onClick={expandAndFocusInput}>
                      <div className="tray-handle-bar"></div>
                      <div className="collapsed-compose-bar">
                        <span className="collapsed-compose-icon">🤖</span>
                        <span className="collapsed-compose-placeholder">Ask Policy Support AI...</span>
                        {qaConversation.length > 0 && (
                          <span className="collapsed-compose-badge">{qaConversation.length}</span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Expanded State - Full tray */}
                  {qaTrayExpanded && (
                    <>
                      {/* Tray Handle */}
                      <div 
                        className="policy-qa-tray-handle"
                        onClick={() => setQaTrayExpanded(false)}
                      >
                        <div className="tray-handle-bar"></div>
                        <span className="tray-handle-label">
                          ▼ Policy Q&A
                          {qaConversation.length > 0 && ` (${qaConversation.length})`}
                        </span>
                      </div>
                      
                      {/* Q&A Conversation Area */}
                      <div className="policy-qa-conversation" ref={qaScrollRef}>
                        {qaConversation.length === 0 ? (
                          <div className="qa-empty-state">
                            <span>🤖</span>
                            <p>Ask questions about the policy and I'll help you find the right answer with citations.</p>
                          </div>
                        ) : (
                          qaConversation.map((item, index) => (
                            <div key={index} className="qa-exchange">
                              <div className="qa-question">
                                <span className="qa-role">👤 You</span>
                                <p>{item.question}</p>
                              </div>
                              <div className="qa-answer">
                                <span className="qa-role">🤖 Policy AI</span>
                                {item.isLoading ? (
                                  <p className="qa-loading">Searching policy...</p>
                                ) : (
                                  <p>{item.answer}</p>
                                )}
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                      
                      {/* Input Row */}
                      <div className="policy-qa-input-row">
                        <input
                          ref={qaInputRef}
                          type="text"
                          className="policy-qa-input"
                          placeholder="Ask about policy..."
                          value={policyQuestion}
                          onChange={(e) => setPolicyQuestion(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleAskPolicy()}
                          disabled={isAskingPolicy}
                        />
                        <button
                          className="policy-qa-btn"
                          onClick={handleAskPolicy}
                          disabled={isAskingPolicy || !policyQuestion.trim()}
                        >
                          {isAskingPolicy ? '...' : '↑'}
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default MobileInfoPanel
