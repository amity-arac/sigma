import { useState, useRef, useEffect } from 'react'
import { useSession } from '../../context/SessionContext'
import { generateResponse } from '../../services/api'
import ToolsList from './ToolsList'
import InjectedDataPanel from './InjectedDataPanel'
import './SidePanel.css'

// Tab configuration
const TABS = [
  { id: 'policy-ai', label: '🤖 Policy AI', icon: '🤖' },
  { id: 'data', label: '📊 Data', icon: '📊' },
  { id: 'tools', label: '🔧 Tools', icon: '🔧' },
  { id: 'persona', label: '👤 Persona', icon: '👤' },
  { id: 'wiki', label: '📖 Wiki', icon: '📖' },
]

// Helper to parse and format policy answer with highlighted citations
function PolicyAnswer({ answer }) {
  if (!answer) return null
  
  // Split by common policy citation patterns
  const parts = answer.split(/(Policy Basis:|Relevant Policy:|From the policy:|Policy Reference:)/gi)
  
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
        if (/Policy Basis:|Relevant Policy:|From the policy:|Policy Reference:/i.test(part)) {
          return <span key={i} className="policy-section-label">{part}</span>
        }
        // Check if this part comes after a policy label (odd indices after split)
        if (i > 0 && /Policy Basis:|Relevant Policy:|From the policy:|Policy Reference:/i.test(parts[i - 1])) {
          return <div key={i} className="policy-citation-block">{part.trim()}</div>
        }
        return <span key={i}>{part}</span>
      })}
    </div>
  )
}

function SidePanel() {
  const { tools, persona, wiki, sessionId, injectedData } = useSession()
  const [activeTab, setActiveTab] = useState('policy-ai')
  const [policyQuestion, setPolicyQuestion] = useState('')
  const [isAskingPolicy, setIsAskingPolicy] = useState(false)
  const [qaConversation, setQaConversation] = useState([])
  const qaScrollRef = useRef(null)
  const qaInputRef = useRef(null)

  // Auto-scroll to bottom of Q&A when new answer arrives
  useEffect(() => {
    if (qaScrollRef.current && qaConversation.length > 0) {
      qaScrollRef.current.scrollTop = qaScrollRef.current.scrollHeight
    }
  }, [qaConversation])

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

  // Filter tabs - only show Data tab if injectedData exists
  const visibleTabs = TABS.filter(tab => {
    if (tab.id === 'data') return !!injectedData
    return true
  })

  const renderTabContent = () => {
    switch (activeTab) {
      case 'policy-ai':
        return (
          <div className="tab-content-inner">
            {wiki && sessionId ? (
              <div className="desktop-policy-qa-standalone">
                {/* Q&A Conversation Area */}
                <div className="policy-qa-conversation-desktop" ref={qaScrollRef}>
                  {qaConversation.length === 0 ? (
                    <div className="qa-empty-state-desktop">
                      <span>🤖</span>
                      <p>Ask questions about the policy and I'll help you find the right answer with exact citations.</p>
                    </div>
                  ) : (
                    qaConversation.map((item, index) => (
                      <div key={index} className="qa-exchange-desktop">
                        <div className="qa-question-desktop">
                          <span className="qa-role-desktop">👤 You</span>
                          <p>{item.question}</p>
                        </div>
                        <div className="qa-answer-desktop">
                          <span className="qa-role-desktop">🤖 Policy AI</span>
                          {item.isLoading ? (
                            <p className="qa-loading-desktop">
                              <span className="loading-dots">Searching policy</span>
                            </p>
                          ) : (
                            <PolicyAnswer answer={item.answer} />
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
                
                {/* Input Row */}
                <div className="policy-qa-input-row-desktop">
                  <input
                    ref={qaInputRef}
                    type="text"
                    className="policy-qa-input-desktop"
                    placeholder="Ask about policy..."
                    value={policyQuestion}
                    onChange={(e) => setPolicyQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAskPolicy()}
                    disabled={isAskingPolicy}
                  />
                  <button
                    className="policy-qa-btn-desktop"
                    onClick={handleAskPolicy}
                    disabled={isAskingPolicy || !policyQuestion.trim()}
                    title="Ask Policy AI"
                  >
                    {isAskingPolicy ? '...' : '↑'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="qa-empty-state-desktop">
                <span>🤖</span>
                <p>Load a policy document to use Policy AI</p>
              </div>
            )}
          </div>
        )
      
      case 'data':
        return (
          <div className="tab-content-inner tab-content-scrollable">
            <InjectedDataPanel embedded />
          </div>
        )
      
      case 'tools':
        return (
          <div className="tab-content-inner tab-content-scrollable">
            <ToolsList tools={tools} />
          </div>
        )
      
      case 'persona':
        return (
          <div className="tab-content-inner tab-content-scrollable">
            <div className="persona-display-full">
              {persona || 'No persona loaded'}
            </div>
          </div>
        )
      
      case 'wiki':
        return (
          <div className="tab-content-inner tab-content-scrollable">
            <div className="wiki-content-full">
              {wiki || 'No wiki loaded'}
            </div>
          </div>
        )
      
      default:
        return null
    }
  }

  return (
    <div className="side-panel">
      {/* Tab Headers */}
      <div className="side-panel-tabs">
        {visibleTabs.map(tab => (
          <button
            key={tab.id}
            className={`side-panel-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            title={tab.label}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label.replace(/^[^\s]+\s/, '')}</span>
          </button>
        ))}
      </div>
      
      {/* Tab Content */}
      <div className="side-panel-content">
        {renderTabContent()}
      </div>
    </div>
  )
}

export default SidePanel
