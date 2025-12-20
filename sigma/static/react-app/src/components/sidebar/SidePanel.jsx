import { useSession } from '../../context/SessionContext'
import PanelCard from './PanelCard'
import ToolsList from './ToolsList'
import './SidePanel.css'

function SidePanel() {
  const { tools, persona, wiki } = useSession()

  return (
    <div className="side-panel">
      <PanelCard title="🔧 Tools" defaultExpanded>
        <ToolsList tools={tools} />
      </PanelCard>
      
      <PanelCard title="👤 User Persona">
        <div className="persona-display">
          {persona || 'No persona loaded'}
        </div>
      </PanelCard>
      
      <PanelCard title="📖 Policy / Wiki">
        <div className="wiki-content">
          {wiki || 'No wiki loaded'}
        </div>
      </PanelCard>
    </div>
  )
}

export default SidePanel
