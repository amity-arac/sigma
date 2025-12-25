import { useState, useRef, useEffect } from 'react'
import { useSession } from '../../context/SessionContext'
import { generateResponse } from '../../services/api'
import ToolsList from '../sidebar/ToolsList'
import InjectedDataPanel from '../sidebar/InjectedDataPanel'
import './MobileInfoPanel.css'

// Helper to parse and format policy answer with highlighted citations
function PolicyAnswer({ answer }) {
  if (!answer) return null
  
  // Split by common policy citation patterns
  const parts = answer.split(/(Policy Basis:|Relevant Policy:|From the policy:|Policy Reference:|\*\*Policy Basis:\*\*|\*\*Answer:\*\*)/gi)
  
  if (parts.length <= 1) {
    // No explicit citation found, try to detect quoted text
    const quotedParts = answer.split(/("[^"]+"|'[^']+')/g)
    return (
      <div className="policy-answer-content">
        {quotedParts.map((part, i) => {
          if (part.startsWith('"') || part.startsWith("'")) {
            return <span key={i} className="policy-quote-highlight">{part}</span>
          }
          return <span key={i}>{part}</span>
        })}
      </div>
    )
  }
  
  return (
    <div className="policy-answer-content">
      {parts.map((part, i) => {
        if (/Policy Basis:|\*\*Policy Basis:\*\*/i.test(part)) {
          return <span key={i} className="policy-section-label">📜 Policy Basis:</span>
        }
        if (/\*\*Answer:\*\*/i.test(part)) {
          return <span key={i} className="policy-section-label">💡 Answer:</span>
        }
        if (/Relevant Policy:|From the policy:|Policy Reference:/i.test(part)) {
          return <span key={i} className="policy-section-label">{part}</span>
        }
        // Check if this part comes after a policy label
        if (i > 0 && /Policy Basis:|Relevant Policy:|From the policy:|Policy Reference:|\*\*Policy Basis:\*\*/i.test(parts[i - 1])) {
          return <div key={i} className="policy-citation-block">{part.trim()}</div>
        }
        return <span key={i}>{part}</span>
      })}
    </div>
  )
}

function MobileInfoPanel({ onNewSession }) {
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('policyai')  // Default to Policy AI tab
  const [policyQuestion, setPolicyQuestion] = useState('')
  const [isAskingPolicy, setIsAskingPolicy] = useState(false)
  const [qaConversation, setQaConversation] = useState([])  // Array of {question, answer}
  const qaScrollRef = useRef(null)
  const qaInputRef = useRef(null)
  const { 
    tools, 
    persona, 
    wiki,
    injectedData,
    isAutopilotEnabled,
    setIsAutopilotEnabled,
    isAutoApproveEnabled,
    setIsAutoApproveEnabled,
    autopilotTurnCount,
    setAutopilotTurnCount,
    AUTOPILOT_TURN_LIMIT,
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

  // Focus input when switching to policy AI tab
  useEffect(() => {
    if (activeTab === 'policyai' && isOpen) {
      setTimeout(() => {
        qaInputRef.current?.focus()
      }, 100)
    }
  }, [activeTab, isOpen])

  const tabs = [
    { id: 'policyai', label: '🤖 Policy AI', icon: '🤖' },
    ...(injectedData ? [{ id: 'data', label: '📊 Data', icon: '📊' }] : []),
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
    
    // Add question to conversation immediately
    setQaConversation(prev => [...prev, { question: currentQuestion, answer: null, isLoading: true }])
    
    try {
      const prompt = `You are a Policy Support AI assisting a customer service agent. Your role is to help the agent understand and apply company policies correctly.

Here is the company policy document:

---POLICY DOCUMENT---
${wiki}
---END POLICY DOCUMENT---

Agent's Question: ${currentQuestion}

IMPORTANT: You MUST structure your response EXACTLY as follows:

**Answer:** [Provide a clear, direct, and actionable answer to the agent's question]

**Policy Basis:** [Quote the EXACT text from the policy document that supports your answer. Use quotation marks around the exact policy text. If multiple sections apply, list each one.]

Be specific and cite the actual policy text. Do not paraphrase - use exact quotes from the policy.`
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
          {activeTab === 'policyai' && (
            <div className="info-section policy-ai-section">
              {wiki && sessionId ? (
                <div className="policy-ai-container">
                  {/* Q&A Conversation Area */}
                  <div className="policy-qa-conversation-full" ref={qaScrollRef}>
                    {qaConversation.length === 0 ? (
                      <div className="qa-empty-state">
                        <span>🤖</span>
                        <h4>Policy AI Assistant</h4>
                        <p>Ask questions about the policy and I'll help you find the right answer with exact citations from the policy document.</p>
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
                              <PolicyAnswer answer={item.answer} />
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                  
                  {/* Input Row - Fixed at bottom */}
                  <div className="policy-qa-input-row-fixed">
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
                </div>
              ) : (
                <div className="qa-empty-state">
                  <span>🤖</span>
                  <p>Load a policy document to use Policy AI</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'data' && (
            <div className="info-section">
              <InjectedDataPanel embedded />
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="info-section">
              <div className="mobile-settings">
                {/* Autopilot Toggle */}
                <div className="mobile-setting-row">
                  <span className="setting-label">🚀 Autopilot</span>
                  <div className="setting-controls">
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={isAutopilotEnabled}
                        onChange={(e) => {
                          setIsAutopilotEnabled(e.target.checked)
                          if (e.target.checked) {
                            setAutopilotTurnCount(0)
                          }
                        }}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    {isAutopilotEnabled && (
                      <span className={`turn-counter-mobile ${autopilotTurnCount >= AUTOPILOT_TURN_LIMIT * 0.8 ? 'warning' : ''}`}>
                        {autopilotTurnCount}/{AUTOPILOT_TURN_LIMIT}
                      </span>
                    )}
                  </div>
                </div>
                <p className="setting-hint">Auto-triggers action after each user/tool response</p>

                {/* Auto-Approve Toggle */}
                <div className={`mobile-setting-row ${!isAutopilotEnabled ? 'disabled' : ''}`}>
                  <span className="setting-label">🛡️ Auto-Approve</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={isAutoApproveEnabled}
                      onChange={(e) => setIsAutoApproveEnabled(e.target.checked)}
                      disabled={!isAutopilotEnabled}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                <p className="setting-hint">
                  {isAutopilotEnabled 
                    ? "Policy AI auto-approves compliant actions" 
                    : "Enable Autopilot first to use Auto-Approve"
                  }
                </p>

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
            <div className="info-section">
              <div className="wiki-content">
                {wiki || 'No wiki loaded'}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default MobileInfoPanel
