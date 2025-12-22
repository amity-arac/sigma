import { useState } from 'react'
import './Sidebar.css'

function Sidebar({ currentView, onNavigate }) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)

  const menuItems = [
    { id: 'simulator', label: 'Simulator', icon: '🎮' },
    { id: 'trajectory', label: 'Trajectory', icon: '📊' },
    { id: 'environments', label: 'Environments', icon: '⚙️' },
  ]

  const handleNavigation = (viewId) => {
    onNavigate(viewId)
    setIsMobileOpen(false)
  }

  return (
    <>
      {/* Mobile hamburger button */}
      <button 
        className="mobile-menu-toggle"
        onClick={() => setIsMobileOpen(!isMobileOpen)}
        aria-label="Toggle menu"
      >
        <span className={`hamburger ${isMobileOpen ? 'open' : ''}`}>
          <span></span>
          <span></span>
          <span></span>
        </span>
      </button>

      {/* Backdrop for mobile */}
      {isMobileOpen && (
        <div 
          className="sidebar-backdrop"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''} ${isMobileOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="logo-icon">Σ</span>
            {!isCollapsed && <span className="logo-text">Sigma</span>}
          </div>
          <button 
            className="collapse-btn desktop-only"
            onClick={() => setIsCollapsed(!isCollapsed)}
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? '→' : '←'}
          </button>
        </div>

        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${currentView === item.id ? 'active' : ''}`}
              onClick={() => handleNavigation(item.id)}
              title={isCollapsed ? item.label : undefined}
            >
              <span className="nav-icon">{item.icon}</span>
              {!isCollapsed && <span className="nav-label">{item.label}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          {!isCollapsed && (
            <p className="sidebar-tagline">LLM User Simulator</p>
          )}
        </div>
      </aside>
    </>
  )
}

export default Sidebar
