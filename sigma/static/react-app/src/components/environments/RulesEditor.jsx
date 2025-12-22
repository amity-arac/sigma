import { useState } from 'react'
import './RulesEditor.css'

function RulesEditor({ content, onChange }) {
  const [rules, setRules] = useState(() => {
    try {
      return JSON.parse(content)
    } catch {
      return []
    }
  })
  const [editingIndex, setEditingIndex] = useState(null)
  const [editText, setEditText] = useState('')
  const [showRawJson, setShowRawJson] = useState(false)
  
  const updateRules = (newRules) => {
    setRules(newRules)
    onChange(JSON.stringify(newRules, null, 2))
  }
  
  const handleEdit = (index) => {
    setEditingIndex(index)
    setEditText(rules[index])
  }
  
  const handleSaveEdit = () => {
    if (editingIndex !== null) {
      const newRules = [...rules]
      newRules[editingIndex] = editText
      updateRules(newRules)
      setEditingIndex(null)
      setEditText('')
    }
  }
  
  const handleCancelEdit = () => {
    setEditingIndex(null)
    setEditText('')
  }
  
  const handleDelete = (index) => {
    if (window.confirm('Are you sure you want to delete this rule?')) {
      const newRules = rules.filter((_, i) => i !== index)
      updateRules(newRules)
    }
  }
  
  const handleAddRule = () => {
    const newRules = [...rules, 'New rule - click to edit']
    updateRules(newRules)
    setEditingIndex(newRules.length - 1)
    setEditText('New rule - click to edit')
  }
  
  const handleMoveUp = (index) => {
    if (index === 0) return
    const newRules = [...rules]
    ;[newRules[index - 1], newRules[index]] = [newRules[index], newRules[index - 1]]
    updateRules(newRules)
  }
  
  const handleMoveDown = (index) => {
    if (index === rules.length - 1) return
    const newRules = [...rules]
    ;[newRules[index], newRules[index + 1]] = [newRules[index + 1], newRules[index]]
    updateRules(newRules)
  }
  
  const handleRawJsonChange = (e) => {
    try {
      const parsed = JSON.parse(e.target.value)
      setRules(parsed)
      onChange(e.target.value)
    } catch {
      onChange(e.target.value)
    }
  }
  
  if (showRawJson) {
    return (
      <div className="rules-editor raw-mode">
        <div className="editor-toolbar">
          <button className="toggle-view-btn" onClick={() => setShowRawJson(false)}>
            ← Back to List View
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
    <div className="rules-editor">
      <div className="editor-toolbar">
        <div className="toolbar-info">
          <span className="rules-count">{rules.length} rules</span>
        </div>
        <div className="toolbar-actions">
          <button className="add-rule-btn" onClick={handleAddRule}>
            + Add Rule
          </button>
          <button className="toggle-view-btn" onClick={() => setShowRawJson(true)}>
            Edit Raw JSON
          </button>
        </div>
      </div>
      
      <div className="rules-content">
        <div className="rules-list">
          {rules.map((rule, index) => (
            <div key={index} className={`rule-item ${editingIndex === index ? 'editing' : ''}`}>
              <div className="rule-number">{index + 1}</div>
              
              {editingIndex === index ? (
                <div className="rule-edit-area">
                  <textarea
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    autoFocus
                    rows={3}
                  />
                  <div className="edit-actions">
                    <button className="save-btn" onClick={handleSaveEdit}>Save</button>
                    <button className="cancel-btn" onClick={handleCancelEdit}>Cancel</button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="rule-text" onClick={() => handleEdit(index)}>
                    {rule}
                  </div>
                  <div className="rule-actions">
                    <button 
                      className="move-btn" 
                      onClick={() => handleMoveUp(index)}
                      disabled={index === 0}
                      title="Move up"
                    >
                      ↑
                    </button>
                    <button 
                      className="move-btn" 
                      onClick={() => handleMoveDown(index)}
                      disabled={index === rules.length - 1}
                      title="Move down"
                    >
                      ↓
                    </button>
                    <button 
                      className="edit-btn" 
                      onClick={() => handleEdit(index)}
                      title="Edit"
                    >
                      ✎
                    </button>
                    <button 
                      className="delete-btn" 
                      onClick={() => handleDelete(index)}
                      title="Delete"
                    >
                      ×
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
          
          {rules.length === 0 && (
            <div className="empty-state">
              <p>No rules defined yet.</p>
              <button className="add-rule-btn" onClick={handleAddRule}>
                + Add your first rule
              </button>
            </div>
          )}
        </div>
        
        <div className="rules-help">
          <h4>About Rules</h4>
          <p>
            Rules define behavioral constraints for the agent. Each rule is a statement 
            that the agent must follow during conversations.
          </p>
          <h5>Tips:</h5>
          <ul>
            <li>Be specific and actionable</li>
            <li>Use clear language</li>
            <li>Order matters - more important rules first</li>
            <li>Click on a rule to edit it</li>
            <li>Use ↑↓ buttons to reorder</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default RulesEditor
