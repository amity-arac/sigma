import './Header.css'

function Header({ onAdminClick }) {
  return (
    <header className="header">
      <div className="header-left">
        <h1>Sigma</h1>
        <p>Act as an agent, interact with an LLM-simulated user</p>
      </div>
      {onAdminClick && (
        <div className="header-right">
          <button className="admin-link" onClick={onAdminClick}>
            📊 Admin
          </button>
        </div>
      )}
    </header>
  )
}

export default Header
