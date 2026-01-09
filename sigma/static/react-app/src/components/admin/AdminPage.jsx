import { useState, useEffect, useCallback } from 'react'
import ReactDiffViewer, { DiffMethod } from 'react-diff-viewer-continued'
import { listTrajectories, getTrajectory, deleteTrajectory, updateTrajectory, fetchEnvironments, exportTrajectories, updateEnvironmentFile } from '../../services/api'
import TrajectoryTable from './TrajectoryTable'
import TrajectoryViewer from './TrajectoryViewer'
import './AdminPage.css'

// Custom styles for the diff viewer to match our dark theme
const diffViewerStyles = {
  variables: {
    dark: {
      diffViewerBackground: '#1a1a2e',
      diffViewerColor: '#e0e0e0',
      addedBackground: 'rgba(76, 175, 80, 0.2)',
      addedColor: '#7be082',
      removedBackground: 'rgba(244, 67, 54, 0.2)',
      removedColor: '#f48c85',
      wordAddedBackground: 'rgba(76, 175, 80, 0.4)',
      wordRemovedBackground: 'rgba(244, 67, 54, 0.4)',
      addedGutterBackground: 'rgba(76, 175, 80, 0.15)',
      removedGutterBackground: 'rgba(244, 67, 54, 0.15)',
      gutterBackground: '#16162a',
      gutterBackgroundDark: '#0d0d1a',
      highlightBackground: 'rgba(255, 193, 7, 0.2)',
      highlightGutterBackground: 'rgba(255, 193, 7, 0.15)',
      codeFoldGutterBackground: '#1a1a2e',
      codeFoldBackground: '#252538',
      emptyLineBackground: '#1a1a2e',
      codeFoldContentColor: '#888',
    },
  },
  line: {
    padding: '4px 8px',
    fontSize: '13px',
    fontFamily: "'SF Mono', Monaco, 'Courier New', monospace",
  },
  gutter: {
    padding: '4px 10px',
    minWidth: '40px',
  },
  contentText: {
    fontFamily: "'SF Mono', Monaco, 'Courier New', monospace",
  },
}

// Helper to read URL search params
function getUrlParams() {
  const params = new URLSearchParams(window.location.search)
  return {
    env: params.get('env') || '',
    date: params.get('date') || '',
    status: params.get('status') || 'all'
  }
}

// Helper to update URL search params
function updateUrlParams(params) {
  const url = new URL(window.location.href)
  Object.entries(params).forEach(([key, value]) => {
    if (value && value !== 'all' && value !== '') {
      url.searchParams.set(key, value)
    } else {
      url.searchParams.delete(key)
    }
  })
  window.history.replaceState({}, '', url.toString())
}

function AdminPage() {
  // Initialize state from URL params
  const initialParams = getUrlParams()
  
  const [rawTrajectories, setRawTrajectories] = useState([]) // Unfiltered data from API
  const [trajectories, setTrajectories] = useState([]) // Filtered data for display
  const [totalCount, setTotalCount] = useState(0) // Total before filtering
  const [environments, setEnvironments] = useState([])
  const [selectedEnv, setSelectedEnv] = useState(initialParams.env)
  const [dateFilter, setDateFilter] = useState(initialParams.date)
  const [statusFilter, setStatusFilter] = useState(initialParams.status) // 'all', 'complete', 'incomplete'
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedTrajectory, setSelectedTrajectory] = useState(null)
  const [viewerOpen, setViewerOpen] = useState(false)
  const [loadingTrajectory, setLoadingTrajectory] = useState(false)
  const [exportFormat, setExportFormat] = useState('dpo')
  const [exporting, setExporting] = useState(false)
  const [optimizedPolicyData, setOptimizedPolicyData] = useState(null)
  const [showPolicyModal, setShowPolicyModal] = useState(false)
  const [copiedPolicy, setCopiedPolicy] = useState(false)
  const [diffViewMode, setDiffViewMode] = useState('unified') // 'split', 'unified', 'final'
  const [showUpdateConfirm, setShowUpdateConfirm] = useState(false)
  const [updatingPolicy, setUpdatingPolicy] = useState(false)
  const [policyUpdateSuccess, setPolicyUpdateSuccess] = useState(false)

  // Apply filters to raw data (instant, no API call)
  useEffect(() => {
    let filtered = [...rawTrajectories]
    
    // Filter by date if specified
    if (dateFilter) {
      filtered = filtered.filter(t => {
        const createdAt = t.created_at || ''
        return createdAt.startsWith(dateFilter)
      })
    }
    
    // Filter by status
    if (statusFilter === 'complete') {
      filtered = filtered.filter(t => t.is_done === true)
    } else if (statusFilter === 'incomplete') {
      filtered = filtered.filter(t => t.is_done !== true)
    }
    
    // Sort by created_at descending
    filtered.sort((a, b) => {
      const dateA = a.created_at || ''
      const dateB = b.created_at || ''
      return dateB.localeCompare(dateA)
    })
    
    setTrajectories(filtered)
  }, [rawTrajectories, dateFilter, statusFilter])

  // Update URL when filters change
  useEffect(() => {
    updateUrlParams({
      env: selectedEnv,
      date: dateFilter,
      status: statusFilter
    })
  }, [selectedEnv, dateFilter, statusFilter])

  // Set first environment as default when environments load (only if not set from URL)
  useEffect(() => {
    if (environments.length > 0 && !selectedEnv) {
      setSelectedEnv(environments[0].name)
    }
  }, [environments, selectedEnv])

  const loadTrajectories = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await listTrajectories(selectedEnv || null, 500)
      const data = result.trajectories || []
      
      // Store total count and raw data (filtering happens in useEffect)
      setTotalCount(data.length)
      setRawTrajectories(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [selectedEnv]) // Only re-fetch when environment changes

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

  const handleSimulate = (trajectory) => {
    // Navigate to simulation page for this trajectory
    window.location.href = `/trajectories/${trajectory.id}/simulation`
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
        // Handle optimized_policy format differently - show modal instead of download
        if (exportFormat === 'optimized_policy') {
          // Parse the response (it's a single JSON object in the JSONL)
          const policyData = JSON.parse(result.data)
          
          // Check if there was an error (like no rejections found)
          if (policyData.success === false) {
            setError(policyData.error || 'Failed to generate optimized policy')
            return
          }
          
          // Store the data and show modal
          setOptimizedPolicyData(policyData)
          setShowPolicyModal(true)
        } else {
          // Standard training data export
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
        }
      } else {
        setError(`Export failed: ${result.error}`)
      }
    } catch (err) {
      setError(`Export failed: ${err.message}`)
    } finally {
      setExporting(false)
    }
  }

  const handleUpdatePolicy = async () => {
    if (!optimizedPolicyData?.revised_policy || !optimizedPolicyData?.env_name) {
      setError('Missing policy data or environment name')
      return
    }

    setUpdatingPolicy(true)
    try {
      await updateEnvironmentFile(
        optimizedPolicyData.env_name,
        'policy.md',
        optimizedPolicyData.revised_policy
      )
      setShowUpdateConfirm(false)
      setShowPolicyModal(false)
      setOptimizedPolicyData(null)
    } catch (err) {
      setError(`Failed to update policy: ${err.message}`)
    } finally {
      setUpdatingPolicy(false)
    }
  }

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div className="admin-header-left">
          <h1>📊 Trajectory Admin</h1>
        </div>
        <div className="admin-stats">
          <span className="stat-badge">
            {(dateFilter || statusFilter !== 'all') 
              ? `${trajectories.length} of ${totalCount} trajectories`
              : `${trajectories.length} trajectories`
            }
          </span>
        </div>
      </div>

      <div className="admin-filters">
        <div className="filter-group">
          <label>Environment</label>
          <select 
            value={selectedEnv} 
            onChange={(e) => setSelectedEnv(e.target.value)}
          >
            {environments.map(env => (
              <option key={env.name} value={env.name}>{env.display_name}</option>
            ))}
          </select>
        </div>
        
        <div className="filter-group">
          <label>Status</label>
          <select 
            value={statusFilter} 
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Status</option>
            <option value="complete">✓ Complete</option>
            <option value="incomplete">○ Incomplete</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label>Date (YYYY-MM-DD)</label>
          <input 
            type="text" 
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            placeholder="e.g. 2025-12-20"
            pattern="\d{4}-\d{2}-\d{2}"
          />
        </div>
        
        <button className="refresh-button" onClick={loadTrajectories} disabled={loading}>
          🔄 Refresh
        </button>
        
        {(dateFilter || statusFilter !== 'all') && (
          <button className="clear-filter-button" onClick={() => { setDateFilter(''); setStatusFilter('all'); }}>
            ✕ Clear Filters
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
            <option value="raw">Raw Trajectory</option>
            <option value="optimized_policy">Optimized Policy</option>
          </select>
          <button 
            className="export-button" 
            onClick={handleExport} 
            disabled={exporting || trajectories.length === 0}
            title={exportFormat === 'dpo' 
              ? 'Export as DPO (Direct Preference Optimization) format - creates chosen/rejected pairs from rejected suggestions' 
              : exportFormat === 'grpo'
              ? 'Export as GRPO (Group Relative Policy Optimization) format - extracts tool call sequences for verifiable rewards'
              : exportFormat === 'sft'
              ? 'Export as SFT (Supervised Fine-Tuning) format - creates one training sample per assistant turn'
              : exportFormat === 'raw'
              ? 'Export raw trajectories as-is in JSONL format - for archival or custom processing'
              : 'Generate an optimized policy based on rejection patterns - AI analyzes rejections and suggests policy improvements'
            }
          >
            {exporting ? '⏳ Exporting...' : exportFormat === 'optimized_policy' ? '🧠 Generate Optimized Policy' : '📥 Export Training Data'}
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
          onSimulate={() => handleSimulate(selectedTrajectory)}
        />
      )}

      {showPolicyModal && optimizedPolicyData && (
        <div className="policy-modal-overlay" onClick={() => setShowPolicyModal(false)}>
          <div className="policy-modal policy-modal-fullwidth" onClick={(e) => e.stopPropagation()}>
            <div className="policy-modal-header">
              <h2>🧠 Optimized Policy Analysis</h2>
              <button className="policy-modal-close" onClick={() => setShowPolicyModal(false)}>✕</button>
            </div>
            
            <div className="policy-modal-content-split">
              {/* Left Panel - Compact Analysis */}
              <div className="policy-analysis-compact">
                {/* Compact Stats Row */}
                <div className="policy-stats-compact">
                  <div className="stat-compact">
                    <span className="stat-num">{optimizedPolicyData.total_rejections_analyzed || 0}</span>
                    <span className="stat-lbl">rejections</span>
                  </div>
                  <div className="stat-compact">
                    <span className="stat-num">{optimizedPolicyData.patterns_identified?.length || 0}</span>
                    <span className="stat-lbl">patterns</span>
                  </div>
                  <div className="stat-compact">
                    <span className="stat-num">{optimizedPolicyData.recommended_changes?.length || 0}</span>
                    <span className="stat-lbl">changes</span>
                  </div>
                </div>

                {/* Scrollable Analysis Content */}
                <div className="policy-analysis-scroll">
                  {optimizedPolicyData.patterns_identified?.length > 0 && (
                    <div className="policy-section-compact">
                      <h4>📋 Patterns</h4>
                      {optimizedPolicyData.patterns_identified.map((pattern, idx) => (
                        <div key={idx} className="item-compact">
                          <div className="item-title">{pattern.pattern_name}</div>
                          <div className="item-desc">{pattern.description}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {optimizedPolicyData.root_causes?.length > 0 && (
                    <div className="policy-section-compact">
                      <h4>🔍 Root Causes</h4>
                      {optimizedPolicyData.root_causes.map((cause, idx) => (
                        <div key={idx} className="item-compact">
                          <div className="item-title">{cause.cause}</div>
                          {cause.affected_patterns?.length > 0 && (
                            <div className="item-meta">Affects: {cause.affected_patterns.join(', ')}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {optimizedPolicyData.recommended_changes?.length > 0 && (
                    <div className="policy-section-compact">
                      <h4>💡 Changes</h4>
                      {optimizedPolicyData.recommended_changes.map((change, idx) => (
                        <div key={idx} className="change-compact">
                          <div className="change-header-compact">
                            <span className="change-section-compact">{change.section}</span>
                            <span className={`change-badge change-badge-${change.change_type}`}>{change.change_type}</span>
                          </div>
                          <div className="item-desc">{change.description}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Panel - Diff Viewer (larger) */}
              <div className="policy-diff-panel">
                <div className="policy-diff-toolbar">
                  <div className="view-mode-selector">
                    <button 
                      className={`view-mode-btn ${diffViewMode === 'split' ? 'active' : ''}`}
                      onClick={() => setDiffViewMode('split')}
                      title="Side-by-side comparison"
                    >
                      Split
                    </button>
                    <button 
                      className={`view-mode-btn ${diffViewMode === 'unified' ? 'active' : ''}`}
                      onClick={() => setDiffViewMode('unified')}
                      title="Unified diff view"
                    >
                      Unified
                    </button>
                    <button 
                      className={`view-mode-btn ${diffViewMode === 'final' ? 'active' : ''}`}
                      onClick={() => setDiffViewMode('final')}
                      title="Final policy only"
                    >
                      Final
                    </button>
                  </div>
                  <div className="policy-toolbar-actions">
                    <button 
                      className={`copy-policy-button ${copiedPolicy ? 'copied' : ''}`}
                      onClick={() => {
                        navigator.clipboard.writeText(optimizedPolicyData.revised_policy || '')
                        setCopiedPolicy(true)
                        setTimeout(() => setCopiedPolicy(false), 2000)
                      }}
                    >
                      {copiedPolicy ? '✓ Copied!' : '📋 Copy'}
                    </button>
                    <button 
                      className={`update-policy-button ${policyUpdateSuccess ? 'success' : ''}`}
                      onClick={() => setShowUpdateConfirm(true)}
                      disabled={updatingPolicy || policyUpdateSuccess}
                    >
                      {policyUpdateSuccess ? '✓ Updated!' : updatingPolicy ? 'Updating...' : '💾 Update Policy'}
                    </button>
                  </div>
                </div>

                {/* Confirmation Dialog */}
                {showUpdateConfirm && (
                  <div className="confirm-dialog-overlay">
                    <div className="confirm-dialog">
                      <h3>⚠️ Update Agent Policy?</h3>
                      <p>
                        This will replace the current Agent Policy for <strong>{optimizedPolicyData?.env_name}</strong> with the optimized version shown above.
                      </p>
                      <p className="confirm-warning">This action cannot be undone.</p>
                      <div className="confirm-dialog-actions">
                        <button 
                          className="confirm-cancel-btn"
                          onClick={() => setShowUpdateConfirm(false)}
                        >
                          Cancel
                        </button>
                        <button 
                          className="confirm-update-btn"
                          onClick={handleUpdatePolicy}
                          disabled={updatingPolicy}
                        >
                          {updatingPolicy ? 'Updating...' : 'Yes, Update'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                <div className="policy-diff-viewer">
                  {diffViewMode === 'final' ? (
                    <pre className="policy-final-view">
                      {optimizedPolicyData.revised_policy || 'No revised policy generated.'}
                    </pre>
                  ) : (
                    <ReactDiffViewer
                      oldValue={optimizedPolicyData.original_policy || ''}
                      newValue={optimizedPolicyData.revised_policy || ''}
                      splitView={diffViewMode === 'split'}
                      useDarkTheme={true}
                      styles={diffViewerStyles}
                      compareMethod={DiffMethod.WORDS}
                      leftTitle="Original Policy"
                      rightTitle="Optimized Policy"
                      showDiffOnly={false}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdminPage
