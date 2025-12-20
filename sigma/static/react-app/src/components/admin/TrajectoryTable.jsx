import './TrajectoryTable.css'

function TrajectoryTable({ trajectories, onView, onDelete, onMarkComplete, loadingId }) {
  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    try {
      // If the timestamp doesn't have timezone info, assume UTC
      let normalizedStr = dateStr
      if (!dateStr.endsWith('Z') && !dateStr.includes('+') && !dateStr.includes('-', 10)) {
        normalizedStr = dateStr + 'Z'
      }
      const date = new Date(normalizedStr)
      // Check for Invalid Date
      if (isNaN(date.getTime())) {
        return dateStr
      }
      return date.toLocaleString()
    } catch {
      return dateStr
    }
  }

  const formatReward = (reward) => {
    if (reward === null || reward === undefined) return '—'
    return reward.toFixed(2)
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
            <th>Reward</th>
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
              <td className="reward-cell">
                <span className={`reward-value ${trajectory.reward >= 1 ? 'success' : trajectory.reward > 0 ? 'partial' : 'fail'}`}>
                  {formatReward(trajectory.reward)}
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
