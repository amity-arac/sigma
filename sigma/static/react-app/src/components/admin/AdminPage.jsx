import { useState, useEffect, useCallback } from 'react'
import { listTrajectories, getTrajectory, deleteTrajectory, updateTrajectory, fetchEnvironments, exportTrajectories } from '../../services/api'
import TrajectoryTable from './TrajectoryTable'
import TrajectoryViewer from './TrajectoryViewer'
import './AdminPage.css'

function AdminPage({ onBack }) {
  const [trajectories, setTrajectories] = useState([])
  const [environments, setEnvironments] = useState([])
  const [selectedEnv, setSelectedEnv] = useState('')
  const [dateFilter, setDateFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedTrajectory, setSelectedTrajectory] = useState(null)
  const [viewerOpen, setViewerOpen] = useState(false)
  const [loadingTrajectory, setLoadingTrajectory] = useState(false)
  const [exportFormat, setExportFormat] = useState('dpo')
  const [exporting, setExporting] = useState(false)

  const loadTrajectories = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await listTrajectories(selectedEnv || null, 500)
      let filtered = result.trajectories || []
      
      // Filter by date if specified
      if (dateFilter) {
        filtered = filtered.filter(t => {
          const createdAt = t.created_at || ''
          return createdAt.startsWith(dateFilter)
        })
      }
      
      // Sort by created_at descending
      filtered.sort((a, b) => {
        const dateA = a.created_at || ''
        const dateB = b.created_at || ''
        return dateB.localeCompare(dateA)
      })
      
      setTrajectories(filtered)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [selectedEnv, dateFilter])

  const loadEnvironments = useCallback(async () => {
    try {
      const envs = await fetchEnvironments()
      setEnvironments(envs)
    } catch (err) {
      console.error('Failed to load environments:', err)
    }
  }, [])

  useEffect(() => {
    loadEnvironments()
  }, [loadEnvironments])

  useEffect(() => {
    loadTrajectories()
  }, [loadTrajectories])

  const handleViewTrajectory = async (trajectory) => {
    setLoadingTrajectory(true)
    try {
      const fullTrajectory = await getTrajectory(trajectory.id, trajectory.env_name)
      setSelectedTrajectory(fullTrajectory)
      setViewerOpen(true)
    } catch (err) {
      setError(`Failed to load trajectory: ${err.message}`)
    } finally {
      setLoadingTrajectory(false)
    }
  }

  const handleDeleteTrajectory = async (trajectory) => {
    if (!window.confirm(`Are you sure you want to delete this trajectory?\n\nID: ${trajectory.id}\nEnv: ${trajectory.env_name}`)) {
      return
    }
    
    try {
      await deleteTrajectory(trajectory.id, trajectory.env_name)
      // Refresh the list
      loadTrajectories()
    } catch (err) {
      setError(`Failed to delete trajectory: ${err.message}`)
    }
  }

  const handleMarkComplete = async (trajectory) => {
    try {
      await updateTrajectory(trajectory.id, trajectory.env_name, { is_done: true })
      // Refresh the list
      loadTrajectories()
      // Update the selected trajectory if it's the one being viewed
      if (selectedTrajectory && selectedTrajectory.id === trajectory.id) {
        setSelectedTrajectory({ ...selectedTrajectory, is_done: true })
      }
    } catch (err) {
      setError(`Failed to mark trajectory as complete: ${err.message}`)
    }
  }

  const handleCloseViewer = () => {
    setViewerOpen(false)
    setSelectedTrajectory(null)
  }

  const handleExport = async () => {
    if (trajectories.length === 0) {
      setError('No trajectories to export')
      return
    }
    
    setExporting(true)
    setError(null)
    
    try {
      // Get trajectory IDs from current filtered view
      const trajectoryIds = trajectories.map(t => t.id)
      
      const result = await exportTrajectories(
        exportFormat,
        selectedEnv || null,
        trajectoryIds,
        dateFilter || null
      )
      
      if (result.success) {
        // Create and download the file
        const blob = new Blob([result.data], { type: 'application/x-ndjson' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        const timestamp = new Date().toISOString().split('T')[0]
        const envSuffix = selectedEnv ? `_${selectedEnv}` : ''
        a.download = `${exportFormat}_training_data${envSuffix}_${timestamp}.jsonl`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        
        // Show success message
        alert(`Successfully exported ${result.count} records in ${exportFormat.toUpperCase()} format`)
      } else {
        setError(`Export failed: ${result.error}`)
      }
    } catch (err) {
      setError(`Export failed: ${err.message}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div className="admin-header-left">
          <button className="back-button" onClick={onBack}>
            ← Back to Simulator
          </button>
          <h1>📊 Trajectory Admin</h1>
        </div>
        <div className="admin-stats">
          <span className="stat-badge">{trajectories.length} trajectories</span>
        </div>
      </div>

      <div className="admin-filters">
        <div className="filter-group">
          <label>Environment</label>
          <select 
            value={selectedEnv} 
            onChange={(e) => setSelectedEnv(e.target.value)}
          >
            <option value="">All Environments</option>
            {environments.map(env => (
              <option key={env.name} value={env.name}>{env.display_name}</option>
            ))}
          </select>
        </div>
        
        <div className="filter-group">
          <label>Date</label>
          <input 
            type="date" 
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
        </div>
        
        <button className="refresh-button" onClick={loadTrajectories} disabled={loading}>
          🔄 Refresh
        </button>
        
        {dateFilter && (
          <button className="clear-filter-button" onClick={() => setDateFilter('')}>
            ✕ Clear Date Filter
          </button>
        )}
        
        <div className="export-group">
          <select 
            value={exportFormat} 
            onChange={(e) => setExportFormat(e.target.value)}
            className="export-format-select"
          >
            <option value="dpo">DPO Format</option>
            <option value="grpo">GRPO Format</option>
            <option value="sft">SFT Format</option>
          </select>
          <button 
            className="export-button" 
            onClick={handleExport} 
            disabled={exporting || trajectories.length === 0}
            title={exportFormat === 'dpo' 
              ? 'Export as DPO (Direct Preference Optimization) format - creates chosen/rejected pairs from rejected suggestions' 
              : exportFormat === 'grpo'
              ? 'Export as GRPO (Group Relative Policy Optimization) format - extracts tool call sequences for verifiable rewards'
              : 'Export as SFT (Supervised Fine-Tuning) format - creates one training sample per assistant turn'
            }
          >
            {exporting ? '⏳ Exporting...' : '📥 Export Training Data'}
          </button>
        </div>
      </div>

      {error && (
        <div className="admin-error">
          ⚠️ {error}
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {loading ? (
        <div className="admin-loading">
          <div className="spinner"></div>
          <span>Loading trajectories...</span>
        </div>
      ) : (
        <TrajectoryTable 
          trajectories={trajectories}
          onView={handleViewTrajectory}
          onDelete={handleDeleteTrajectory}
          onMarkComplete={handleMarkComplete}
          loadingId={loadingTrajectory ? selectedTrajectory?.id : null}
        />
      )}

      {viewerOpen && selectedTrajectory && (
        <TrajectoryViewer 
          trajectory={selectedTrajectory}
          onClose={handleCloseViewer}
          onDelete={() => {
            handleDeleteTrajectory(selectedTrajectory)
            handleCloseViewer()
          }}
          onMarkComplete={() => handleMarkComplete(selectedTrajectory)}
        />
      )}
    </div>
  )
}

export default AdminPage
