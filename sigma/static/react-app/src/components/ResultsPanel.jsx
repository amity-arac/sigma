import './ResultsPanel.css'

function ResultsPanel({ results, onNewSession }) {
  if (!results) return null

  return (
    <div className="results-panel">
      <h3>🏆 Simulation Results</h3>
      
      <div className="results-content">
        <div className="result-item">
          <span className="result-label">Reward</span>
          <span className={`result-value ${results.reward === 1.0 ? 'success' : 'failure'}`}>
            {results.reward}
          </span>
        </div>
        
        {results.reward_info && (
          <>
            {results.reward_info.r_actions !== undefined && (
              <div className="result-item">
                <span className="result-label">Actions Correct</span>
                <span className="result-value">{results.reward_info.r_actions}</span>
              </div>
            )}
            {results.reward_info.r_outputs !== undefined && (
              <div className="result-item">
                <span className="result-label">Outputs Correct</span>
                <span className="result-value">{results.reward_info.r_outputs}</span>
              </div>
            )}
          </>
        )}
      </div>
      
      <div className="results-actions">
        <button className="btn btn-primary" onClick={onNewSession}>
          ✨ Start New Session
        </button>
      </div>
    </div>
  )
}

export default ResultsPanel
