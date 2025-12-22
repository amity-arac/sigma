import './TrajectoryTable.css'

function TrajectoryTable({ trajectories, onView, onDelete, onMarkComplete, loadingId }) {
  const formatDate = (dateStr) => {
    if (!dateStr || dateStr === 'null' || dateStr === 'undefined') return 'N/A'
    try {
      // Handle ISO format strings
      let normalizedStr = String(dateStr).trim()
      
      // If the timestamp doesn't have timezone info, assume UTC
      if (normalizedStr.includes('T') && !normalizedStr.endsWith('Z') && !normalizedStr.includes('+') && !/\d{2}:\d{2}:\d{2}-\d{2}/.test(normalizedStr)) {
        normalizedStr = normalizedStr + 'Z'
      }
      
      const date = new Date(normalizedStr)
      // Check for Invalid Date
      if (isNaN(date.getTime())) {
        // Try parsing without timezone assumption
        const fallbackDate = new Date(dateStr)
        if (!isNaN(fallbackDate.getTime())) {
          return fallbackDate.toLocaleString()
        }
        return dateStr
      }
      return date.toLocaleString()
    } catch {
      return dateStr || 'N/A'
    }
  }

  if (trajectories.length === 0) {
    return (
      <div className="trajectory-table-empty">
        <span className="empty-icon">📭</span>
        <h3>No trajectories found</h3>
        <p>Try adjusting your filters or create some simulation sessions first.</p>
      </div>
    )
  }

  return (
    <div className="trajectory-table-container">
      <table className="trajectory-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Environment</th>
            <th>Created At</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {trajectories.map(trajectory => (
            <tr key={trajectory.id}>
              <td className="id-cell">
                <span className="trajectory-id" title={trajectory.id}>
                  {trajectory.id?.substring(0, 8)}...
                </span>
              </td>
              <td>
                <span className="env-badge">{trajectory.env_name}</span>
              </td>
              <td className="date-cell">
                {formatDate(trajectory.created_at)}
              </td>
              <td>
                <span className={`status-badge ${trajectory.is_done ? 'done' : 'incomplete'}`}>
                  {trajectory.is_done ? '✓ Complete' : '○ Incomplete'}
                </span>
              </td>
              <td className="actions-cell">
                <button 
                  className="action-button view"
                  onClick={() => onView(trajectory)}
                  disabled={loadingId === trajectory.id}
                  title="View trajectory"
                >
                  {loadingId === trajectory.id ? '⏳' : '👁️'} View
                </button>
                {!trajectory.is_done && (
                  <button 
                    className="action-button complete"
                    onClick={() => onMarkComplete(trajectory)}
                    title="Mark as complete"
                  >
                    ✓ Complete
                  </button>
                )}
                <button 
                  className="action-button delete"
                  onClick={() => onDelete(trajectory)}
                  title="Delete trajectory"
                >
                  🗑️ Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default TrajectoryTable
