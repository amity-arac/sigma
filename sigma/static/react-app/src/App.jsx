import { useState, useCallback, useEffect } from 'react'
import { SessionProvider } from './context/SessionContext'
import { ToastProvider } from './context/ToastContext'
import Header from './components/Header'
import SetupPanel from './components/SetupPanel'
import MainContent from './components/MainContent'
import ResultsPanel from './components/ResultsPanel'
import ToastContainer from './components/common/ToastContainer'
import { AdminPage } from './components/admin'

function App() {
  const [isSimulationStarted, setIsSimulationStarted] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [results, setResults] = useState(null)
  const [currentView, setCurrentView] = useState('simulator') // 'simulator' or 'admin'

  // Handle URL-based routing
  useEffect(() => {
    const handleRouteChange = () => {
      const path = window.location.pathname
      if (path === '/admin') {
        setCurrentView('admin')
      } else {
        setCurrentView('simulator')
      }
    }

    // Initial route check
    handleRouteChange()

    // Listen for popstate (browser back/forward)
    window.addEventListener('popstate', handleRouteChange)
    return () => window.removeEventListener('popstate', handleRouteChange)
  }, [])

  const navigateTo = useCallback((view) => {
    const path = view === 'admin' ? '/admin' : '/'
    window.history.pushState({}, '', path)
    setCurrentView(view)
  }, [])

  const handleSimulationStart = useCallback(() => {
    setIsSimulationStarted(true)
    setShowResults(false)
  }, [])

  const handleSimulationEnd = useCallback((resultData) => {
    setShowResults(true)
    setResults(resultData)
  }, [])

  const handleNewSession = useCallback(() => {
    setShowResults(false)
    setResults(null)
  }, [])

  // Render Admin view
  if (currentView === 'admin') {
    return (
      <ToastProvider>
        <div className="container">
          <AdminPage onBack={() => navigateTo('simulator')} />
          <ToastContainer />
        </div>
      </ToastProvider>
    )
  }

  // Render Simulator view
  return (
    <ToastProvider>
      <SessionProvider>
        <div className="container">
          <Header onAdminClick={() => navigateTo('admin')} />
          <SetupPanel 
            onSimulationStart={handleSimulationStart}
            isCollapsed={isSimulationStarted}
          />
          {isSimulationStarted && (
            <MainContent 
              onSimulationEnd={handleSimulationEnd}
              onNewSession={handleNewSession}
            />
          )}
          {showResults && (
            <ResultsPanel 
              results={results}
              onNewSession={handleNewSession}
            />
          )}
          <ToastContainer />
        </div>
      </SessionProvider>
    </ToastProvider>
  )
}

export default App
