import { useMemo, useState, useRef, useEffect } from 'react'
import './WikiPopup.css'

/**
 * WikiPopup - Shows wiki content in a popup while waiting for session to start
 * Supports both HTML (rendered in iframe) and Markdown content types
 * Shows a "Begin Session" button that lights up when session is ready
 */
function WikiPopup({ content, contentType, loading, envName, sessionReady, generatedScenario, onBeginSession, onClose }) {
  const [collapsedSections, setCollapsedSections] = useState({})
  const [activeSection, setActiveSection] = useState(null)
  const contentRef = useRef(null)

  // Extract table of contents from markdown content
  const tableOfContents = useMemo(() => {
    if (!content || contentType !== 'markdown') return []
    
    const toc = []
    const lines = content.split('\n')
    
    lines.forEach((line, index) => {
      const h1Match = line.match(/^# (.+)$/)
      const h2Match = line.match(/^## (.+)$/)
      
      if (h1Match) {
        toc.push({ level: 1, text: h1Match[1], id: `section-${index}` })
      } else if (h2Match) {
        toc.push({ level: 2, text: h2Match[1], id: `section-${index}` })
      }
    })
    
    return toc
  }, [content, contentType])
  // Enhanced markdown to HTML conversion with collapsible sections
  const renderedMarkdown = useMemo(() => {
    if (!content || contentType !== 'markdown') return null
    
    // Process content line by line for better list handling
    const lines = content.split('\n')
    let html = ''
    let inList = false
    let inOrderedList = false
    let currentSectionId = null
    let sectionContent = ''
    let lineIndex = 0
    
    const flushSection = () => {
      if (currentSectionId && sectionContent) {
        html += `<div class="collapsible-content" data-section="${currentSectionId}">${sectionContent}</div></div>`
        sectionContent = ''
      }
    }
    
    lines.forEach((line, index) => {
      lineIndex = index
      
      // Check for list items
      const unorderedMatch = line.match(/^\s*[-*+]\s+(.*)$/)
      const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)$/)
      
      if (unorderedMatch) {
        if (!inList) {
          sectionContent += '<ul class="policy-list">'
          inList = true
        }
        sectionContent += `<li>${unorderedMatch[1]}</li>`
        return
      } else if (orderedMatch) {
        if (!inOrderedList) {
          sectionContent += '<ol class="policy-list">'
          inOrderedList = true
        }
        sectionContent += `<li>${orderedMatch[2]}</li>`
        return
      } else {
        if (inList) {
          sectionContent += '</ul>'
          inList = false
        }
        if (inOrderedList) {
          sectionContent += '</ol>'
          inOrderedList = false
        }
      }
      
      // Headers with collapsible sections
      if (line.match(/^# /)) {
        flushSection()
        const headerText = line.replace(/^# /, '')
        html += `<h1 class="policy-h1" id="section-${index}">${headerText}</h1>`
        currentSectionId = null
      } else if (line.match(/^## /)) {
        flushSection()
        const headerText = line.replace(/^## /, '')
        const isDomainSection = /^(Domain|User|Flight|Reservation|Baggage|Payment|Refund|Ticket|Policy)/i.test(headerText)
        const sectionId = `section-${index}`
        currentSectionId = sectionId
        html += `<div class="collapsible-section" data-section="${sectionId}">
          <h2 class="policy-h2 collapsible-header${isDomainSection ? ' domain-section' : ''}" id="${sectionId}" data-section="${sectionId}">
            <span class="collapse-icon">▼</span>
            ${headerText}
          </h2>`
        sectionContent = ''
      } else if (line.match(/^### /)) {
        sectionContent += line.replace(/^### (.*$)/, '<h3 class="policy-h3">$1</h3>')
      } else if (line.trim() === '') {
        sectionContent += '<br/>'
      } else {
        sectionContent += `<p>${line}</p>`
      }
    })
    
    // Close any open lists
    if (inList) sectionContent += '</ul>'
    if (inOrderedList) sectionContent += '</ol>'
    flushSection()
    
    // Apply inline formatting
    html = html
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/_([^_]+)_/g, '<em>$1</em>')
      // Code blocks
      .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
    
    return html
  }, [content, contentType])

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  // Handle section collapse/expand
  const toggleSection = (sectionId) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }))
  }

  // Handle click on collapsible headers
  const handleContentClick = (e) => {
    const header = e.target.closest('.collapsible-header')
    if (header) {
      const sectionId = header.dataset.section
      if (sectionId) {
        toggleSection(sectionId)
      }
    }
  }

  // Scroll to section when TOC item is clicked
  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId)
    // Expand the section if collapsed
    if (collapsedSections[sectionId]) {
      setCollapsedSections(prev => ({
        ...prev,
        [sectionId]: false
      }))
    }
    // Scroll to the element
    const element = contentRef.current?.querySelector(`#${sectionId}`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  // Apply collapsed state to content
  useEffect(() => {
    if (!contentRef.current) return
    
    const sections = contentRef.current.querySelectorAll('.collapsible-section')
    sections.forEach(section => {
      const sectionId = section.dataset.section
      const content = section.querySelector('.collapsible-content')
      const icon = section.querySelector('.collapse-icon')
      
      if (content && icon) {
        if (collapsedSections[sectionId]) {
          content.classList.add('collapsed')
          icon.classList.add('collapsed')
        } else {
          content.classList.remove('collapsed')
          icon.classList.remove('collapsed')
        }
      }
    })
  }, [collapsedSections, renderedMarkdown])

  // Render content based on type
  const renderContent = () => {
    // Don't show loading for wiki - it's just static files and loads quickly
    // The footer already shows "Initializing session..." for the actual work
    if (!content) {
      return (
        <div className="wiki-popup-placeholder">
          <p>Loading documentation...</p>
        </div>
      )
    }
    
    // For HTML content, use an iframe for proper style isolation
    if (contentType === 'html') {
      return (
        <iframe
          className="wiki-popup-iframe"
          srcDoc={content}
          title={`Wiki - ${envName}`}
          sandbox="allow-scripts allow-same-origin"
        />
      )
    }
    
    // For markdown, render with TOC sidebar
    return (
      <div className="wiki-content-wrapper">
        {/* Table of Contents Sidebar */}
        {tableOfContents.length > 0 && (
          <nav className="wiki-toc">
            <div className="toc-header">
              <span className="toc-icon">📑</span>
              <span className="toc-title">Contents</span>
            </div>
            <ul className="toc-list">
              {tableOfContents.map((item, idx) => (
                <li 
                  key={idx}
                  className={`toc-item toc-level-${item.level} ${activeSection === item.id ? 'active' : ''}`}
                  onClick={() => scrollToSection(item.id)}
                >
                  {item.level === 2 && <span className="toc-bullet">›</span>}
                  <span className="toc-text">{item.text}</span>
                </li>
              ))}
            </ul>
            <div className="toc-actions">
              <button 
                className="toc-btn" 
                onClick={() => setCollapsedSections({})}
                title="Expand all sections"
              >
                Expand All
              </button>
              <button 
                className="toc-btn" 
                onClick={() => {
                  const allCollapsed = {}
                  tableOfContents.filter(t => t.level === 2).forEach(t => {
                    allCollapsed[t.id] = true
                  })
                  setCollapsedSections(allCollapsed)
                }}
                title="Collapse all sections"
              >
                Collapse All
              </button>
            </div>
          </nav>
        )}
        <div 
          ref={contentRef}
          className="wiki-popup-body"
          onClick={handleContentClick}
          dangerouslySetInnerHTML={{ __html: renderedMarkdown }}
        />
      </div>
    )
  }

  return (
    <div className="wiki-popup-overlay" onClick={handleOverlayClick}>
      <div className={`wiki-popup ${contentType === 'html' ? 'wiki-popup-fullsize' : ''}`}>
        <div className="wiki-popup-header">
          <div className="wiki-popup-title-section">
            <span className="wiki-popup-icon">👋</span>
            <span className="wiki-popup-title">Welcome! Please review the agent policy before starting</span>
          </div>
          <button className="wiki-popup-close" onClick={onClose} title="Close">
            ✕
          </button>
        </div>
        
        <div className={`wiki-popup-content ${contentType === 'html' ? 'wiki-popup-content-iframe' : ''}`}>
          {renderContent()}
        </div>
        
        {/* Show generated scenario inspiration if available */}
        {generatedScenario && generatedScenario.seed_task_instruction && (
          <div className="wiki-popup-inspiration">
            <div className="inspiration-header">
              <span className="inspiration-icon">💡</span>
              <span className="inspiration-title">Generated Scenario</span>
            </div>
            <div className="inspiration-content">
              <div className="inspiration-label">Inspired by task:</div>
              <div className="inspiration-text">{generatedScenario.seed_task_instruction}</div>
            </div>
          </div>
        )}

        <div className="wiki-popup-footer">
          <div className="wiki-popup-status">
            {sessionReady ? (
              <span className="status-ready">✓ Trajectory Ready</span>
            ) : (
              <span className="status-loading">
                <span className="mini-spinner"></span>
                Initializing trajectory...
              </span>
            )}
          </div>
          <div className="wiki-popup-actions">
            <button 
              className="btn btn-secondary" 
              onClick={onClose}
            >
              Cancel
            </button>
            <button 
              className={`btn btn-begin ${sessionReady ? 'ready pulse' : ''}`}
              onClick={onBeginSession}
              disabled={!sessionReady}
            >
              {sessionReady ? '🚀 Begin Trajectory' : 'Waiting...'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WikiPopup
