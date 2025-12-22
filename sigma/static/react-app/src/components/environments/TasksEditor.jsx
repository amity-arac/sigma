import { useState, useMemo } from 'react'
import './TasksEditor.css'

// Parse task to get displayable summary
function getTaskSummary(task) {
  const instructions = task.user_scenario?.instructions || {}
  return {
    id: task.id,
    knownInfo: instructions.known_info || '',
    reasonForCall: instructions.reason_for_call || '',
    unknownInfo: instructions.unknown_info || '',
    taskInstructions: instructions.task_instructions || '',
    actionsCount: task.evaluation_criteria?.actions?.length || 0,
  }
}

function TaskCard({ task, isSelected, onClick, onEdit }) {
  const summary = getTaskSummary(task)
  
  return (
    <div 
      className={`task-card ${isSelected ? 'selected' : ''}`}
      onClick={onClick}
    >
      <div className="task-card-header">
        <span className="task-id">Task #{summary.id}</span>
        <span className="task-actions-count">{summary.actionsCount} actions</span>
      </div>
      <div className="task-card-body">
        <p className="task-known-info">{summary.knownInfo}</p>
        <p className="task-reason">{summary.reasonForCall.substring(0, 150)}...</p>
      </div>
      <div className="task-card-footer">
        <button className="edit-btn" onClick={(e) => { e.stopPropagation(); onEdit(task); }}>
          Edit
        </button>
      </div>
    </div>
  )
}

function TaskEditModal({ task, onSave, onClose }) {
  const [editedTask, setEditedTask] = useState(JSON.parse(JSON.stringify(task)))
  
  const instructions = editedTask.user_scenario?.instructions || {}
  
  const updateInstruction = (field, value) => {
    setEditedTask(prev => ({
      ...prev,
      user_scenario: {
        ...prev.user_scenario,
        instructions: {
          ...prev.user_scenario?.instructions,
          [field]: value
        }
      }
    }))
  }
  
  const handleSave = () => {
    onSave(editedTask)
  }
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content task-edit-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Edit Task #{task.id}</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="form-section">
            <h3>User Scenario</h3>
            
            <div className="form-group">
              <label>Known Info</label>
              <textarea
                value={instructions.known_info || ''}
                onChange={e => updateInstruction('known_info', e.target.value)}
                placeholder="What the user knows (e.g., 'You are John Doe with order #123')"
                rows={2}
              />
              <span className="field-hint">Information the simulated user starts with</span>
            </div>
            
            <div className="form-group">
              <label>Reason for Call</label>
              <textarea
                value={instructions.reason_for_call || ''}
                onChange={e => updateInstruction('reason_for_call', e.target.value)}
                placeholder="Why the user is contacting support"
                rows={4}
              />
              <span className="field-hint">The main goal/problem the user wants to solve</span>
            </div>
            
            <div className="form-group">
              <label>Unknown Info</label>
              <textarea
                value={instructions.unknown_info || ''}
                onChange={e => updateInstruction('unknown_info', e.target.value)}
                placeholder="What the user doesn't know"
                rows={2}
              />
              <span className="field-hint">Information the user should NOT volunteer</span>
            </div>
            
            <div className="form-group">
              <label>Task Instructions</label>
              <textarea
                value={instructions.task_instructions || ''}
                onChange={e => updateInstruction('task_instructions', e.target.value)}
                placeholder="Behavioral instructions for the user"
                rows={2}
              />
              <span className="field-hint">How the simulated user should behave</span>
            </div>
          </div>
          
          <div className="form-section">
            <h3>Expected Actions ({editedTask.evaluation_criteria?.actions?.length || 0})</h3>
            <div className="actions-preview">
              {(editedTask.evaluation_criteria?.actions || []).map((action, idx) => (
                <div key={idx} className="action-item">
                  <span className="action-name">{action.name}</span>
                  <code className="action-args">{JSON.stringify(action.arguments)}</code>
                </div>
              ))}
            </div>
            <p className="section-note">
              Note: Edit expected actions in the raw JSON view for advanced changes.
            </p>
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
          <button className="save-btn" onClick={handleSave}>Save Changes</button>
        </div>
      </div>
    </div>
  )
}

function TasksEditor({ content, onChange }) {
  const [tasks, setTasks] = useState(() => {
    try {
      return JSON.parse(content)
    } catch {
      return []
    }
  })
  const [selectedTaskId, setSelectedTaskId] = useState(null)
  const [editingTask, setEditingTask] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [showRawJson, setShowRawJson] = useState(false)
  
  // Filter tasks based on search
  const filteredTasks = useMemo(() => {
    if (!searchTerm) return tasks
    const term = searchTerm.toLowerCase()
    return tasks.filter(task => {
      const summary = getTaskSummary(task)
      return (
        summary.id.toString().includes(term) ||
        summary.knownInfo.toLowerCase().includes(term) ||
        summary.reasonForCall.toLowerCase().includes(term)
      )
    })
  }, [tasks, searchTerm])
  
  const selectedTask = tasks.find(t => t.id === selectedTaskId)
  
  const handleTaskUpdate = (updatedTask) => {
    const newTasks = tasks.map(t => t.id === updatedTask.id ? updatedTask : t)
    setTasks(newTasks)
    onChange(JSON.stringify(newTasks, null, 2))
    setEditingTask(null)
  }
  
  const handleRawJsonChange = (e) => {
    try {
      const parsed = JSON.parse(e.target.value)
      setTasks(parsed)
      onChange(e.target.value)
    } catch {
      // Allow invalid JSON while typing
      onChange(e.target.value)
    }
  }
  
  if (showRawJson) {
    return (
      <div className="tasks-editor raw-mode">
        <div className="editor-toolbar">
          <button className="toggle-view-btn" onClick={() => setShowRawJson(false)}>
            ← Back to Card View
          </button>
        </div>
        <textarea
          className="raw-json-editor"
          value={content}
          onChange={handleRawJsonChange}
          spellCheck={false}
        />
      </div>
    )
  }
  
  return (
    <div className="tasks-editor">
      <div className="editor-toolbar">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search tasks..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
          />
        </div>
        <span className="task-count">{filteredTasks.length} of {tasks.length} tasks</span>
        <button className="toggle-view-btn" onClick={() => setShowRawJson(true)}>
          Edit Raw JSON
        </button>
      </div>
      
      <div className="tasks-content">
        <div className="tasks-list">
          {filteredTasks.map(task => (
            <TaskCard
              key={task.id}
              task={task}
              isSelected={task.id === selectedTaskId}
              onClick={() => setSelectedTaskId(task.id)}
              onEdit={setEditingTask}
            />
          ))}
        </div>
        
        {selectedTask && (
          <div className="task-detail">
            <h3>Task #{selectedTask.id} Details</h3>
            <div className="detail-section">
              <h4>User Scenario</h4>
              <div className="detail-field">
                <label>Known Info:</label>
                <p>{selectedTask.user_scenario?.instructions?.known_info || '-'}</p>
              </div>
              <div className="detail-field">
                <label>Reason for Call:</label>
                <p>{selectedTask.user_scenario?.instructions?.reason_for_call || '-'}</p>
              </div>
              <div className="detail-field">
                <label>Unknown Info:</label>
                <p>{selectedTask.user_scenario?.instructions?.unknown_info || '-'}</p>
              </div>
              <div className="detail-field">
                <label>Task Instructions:</label>
                <p>{selectedTask.user_scenario?.instructions?.task_instructions || '-'}</p>
              </div>
            </div>
            
            <div className="detail-section">
              <h4>Expected Actions</h4>
              <div className="actions-list">
                {(selectedTask.evaluation_criteria?.actions || []).map((action, idx) => (
                  <div key={idx} className="action-detail">
                    <span className="action-index">{idx + 1}</span>
                    <div className="action-content">
                      <span className="action-name">{action.name}</span>
                      <pre className="action-args">{JSON.stringify(action.arguments, null, 2)}</pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <button className="edit-task-btn" onClick={() => setEditingTask(selectedTask)}>
              Edit Task
            </button>
          </div>
        )}
      </div>
      
      {editingTask && (
        <TaskEditModal
          task={editingTask}
          onSave={handleTaskUpdate}
          onClose={() => setEditingTask(null)}
        />
      )}
    </div>
  )
}

export default TasksEditor
