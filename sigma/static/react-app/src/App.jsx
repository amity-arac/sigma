import { useState, useCallback, useEffect } from 'react'
import { SessionProvider } from './context/SessionContext'
import { ToastProvider } from './context/ToastContext'
import { MainLayout } from './components/layout'
import Header from './components/Header'
import SetupPanel from './components/SetupPanel'
import ToastContainer from './components/common/ToastContainer'
import { AdminPage } from './components/admin'
import { EnvironmentsPage } from './components/environments'
import SimulationPage from './components/SimulationPage'

// Simple URL-based router
function parseRoute(pathname) {
  // Match /trajectories/:id/simulation
  const simulationMatch = pathname.match(/^\/trajectories\/([^/]+)\/simulation$/)
  if (simulationMatch) {
    return { view: 'simulation', trajectoryId: simulationMatch[1] }
  }
  
  // Match /trajectory or /admin (legacy)
  if (pathname === '/trajectory' || pathname === '/admin') {
    return { view: 'trajectory', trajectoryId: null }
  }
  
  // Match /env-config
  if (pathname === '/env-config') {
    return { view: 'environments', trajectoryId: null }
  }
  
  // Default to simulator setup
  return { view: 'simulator', trajectoryId: null }
}

function App() {
  const [route, setRoute] = useState(() => parseRoute(window.location.pathname))

  // Handle URL-based routing
  useEffect(() => {
    const handleRouteChange = () => {
      setRoute(parseRoute(window.location.pathname))
    }

    // Listen for popstate (browser back/forward)
    window.addEventListener('popstate', handleRouteChange)
    return () => window.removeEventListener('popstate', handleRouteChange)
  }, [])

  const navigateTo = useCallback((path) => {
    window.history.pushState({}, '', path)
    setRoute(parseRoute(path))
  }, [])

  // Map route view to sidebar view
  const currentView = route.view

  return (
    <ToastProvider>
      <MainLayout currentView={currentView} onNavigate={(view) => {
        if (view === 'trajectory') navigateTo('/trajectory')
        else if (view === 'environments') navigateTo('/env-config')
        else navigateTo('/')
      }}>
        {route.view === 'trajectory' ? (
          <AdminPage />
        ) : route.view === 'environments' ? (
          <EnvironmentsPage />
        ) : route.view === 'simulation' ? (
          <SessionProvider>
            <SimulationPage 
              trajectoryId={route.trajectoryId} 
              onNavigate={navigateTo}
            />
            <ToastContainer />
          </SessionProvider>
        ) : (
          <div className="simulator-page">
            <Header />
            <SetupPanel onNavigate={navigateTo} />
          </div>
        )}
        {route.view !== 'simulation' && <ToastContainer />}
      </MainLayout>
    </ToastProvider>
  )
}

export default App
