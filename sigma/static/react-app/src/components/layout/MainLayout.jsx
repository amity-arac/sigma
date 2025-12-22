import Sidebar from './Sidebar'
import './MainLayout.css'

function MainLayout({ currentView, onNavigate, children }) {
  return (
    <div className="app-layout">
      <Sidebar currentView={currentView} onNavigate={onNavigate} />
      <main className="main-content-area">
        {children}
      </main>
    </div>
  )
}

export default MainLayout
