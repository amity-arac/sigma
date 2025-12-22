import { useState, useMemo } from 'react'
import './DatabaseEditor.css'

// Icon mapping for common data types
const ICON_MAP = {
  users: '👤',
  products: '📦',
  orders: '🛒',
  flights: '✈️',
  reservations: '🎫',
  customers: '👥',
  inventory: '📋',
  transactions: '💳',
  bookings: '📅',
  default: '📄'
}

// Get a display-friendly label from a key
function formatLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .trim()
}

// Infer column type from values
function inferColumnType(values) {
  const nonNullValues = values.filter(v => v != null && v !== '')
  if (nonNullValues.length === 0) return 'string'
  
  const sample = nonNullValues[0]
  if (typeof sample === 'boolean') return 'boolean'
  if (typeof sample === 'number') return 'number'
  if (Array.isArray(sample)) return 'array'
  if (typeof sample === 'object') return 'object'
  
  // Check if it looks like a date
  if (typeof sample === 'string' && /^\d{4}-\d{2}-\d{2}/.test(sample)) return 'date'
  
  return 'string'
}

// Auto-discover columns from data
function discoverColumns(data) {
  if (!data || data.length === 0) return []
  
  // Collect all keys from all records
  const keySet = new Set()
  const keyValues = {}
  
  data.forEach(record => {
    Object.keys(record).forEach(key => {
      keySet.add(key)
      if (!keyValues[key]) keyValues[key] = []
      keyValues[key].push(record[key])
    })
  })
  
  // Create column definitions
  const columns = []
  const priorityKeys = ['id', 'user_id', 'product_id', 'order_id', 'flight_number', 'name', 'email', 'status']
  
  // Sort keys: priority keys first, then alphabetically
  const sortedKeys = Array.from(keySet).sort((a, b) => {
    const aIdx = priorityKeys.indexOf(a)
    const bIdx = priorityKeys.indexOf(b)
    if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx
    if (aIdx !== -1) return -1
    if (bIdx !== -1) return 1
    return a.localeCompare(b)
  })
  
  sortedKeys.forEach(key => {
    const type = inferColumnType(keyValues[key])
    const column = {
      key,
      label: formatLabel(key),
      type
    }
    
    // Create appropriate render function based on type
    if (type === 'array') {
      column.render = (val) => `[${(val || []).length} items]`
    } else if (type === 'object') {
      column.render = (val, row) => {
        if (!val) return '-'
        // Special handling for common nested objects
        if (val.first_name && val.last_name) {
          return `${val.first_name} ${val.last_name}`
        }
        if (val.city) return val.city
        return '{...}'
      }
    } else if (type === 'boolean') {
      column.render = (val) => val ? '✓' : '✗'
    } else if (key === 'status') {
      column.render = (val) => (
        <span className={`status-badge ${val}`}>{val}</span>
      )
    }
    
    columns.push(column)
  })
  
  // Limit to reasonable number of columns for table view
  return columns.slice(0, 8)
}

function DataTable({ data, columns, onRowClick, selectedId }) {
  const [sortColumn, setSortColumn] = useState(null)
  const [sortDirection, setSortDirection] = useState('asc')
  const [page, setPage] = useState(0)
  const pageSize = 20
  
  const sortedData = useMemo(() => {
    if (!sortColumn) return data
    return [...data].sort((a, b) => {
      const aVal = a[sortColumn] || ''
      const bVal = b[sortColumn] || ''
      const comparison = String(aVal).localeCompare(String(bVal))
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [data, sortColumn, sortDirection])
  
  const paginatedData = sortedData.slice(page * pageSize, (page + 1) * pageSize)
  const totalPages = Math.ceil(data.length / pageSize)
  
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }
  
  return (
    <div className="data-table-container">
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map(col => (
                <th key={col.key} onClick={() => handleSort(col.key)}>
                  {col.label}
                  {sortColumn === col.key && (
                    <span className="sort-indicator">{sortDirection === 'asc' ? ' ↑' : ' ↓'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, idx) => (
              <tr 
                key={row.id || idx} 
                className={selectedId === row.id ? 'selected' : ''}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map(col => (
                  <td key={col.key}>
                    {col.render ? col.render(row[col.key], row) : String(row[col.key] ?? '-')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {totalPages > 1 && (
        <div className="table-pagination">
          <button 
            disabled={page === 0}
            onClick={() => setPage(p => p - 1)}
          >
            Previous
          </button>
          <span>Page {page + 1} of {totalPages}</span>
          <button 
            disabled={page >= totalPages - 1}
            onClick={() => setPage(p => p + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}

function RecordDetail({ record, type, onClose, onSave }) {
  const [editedRecord, setEditedRecord] = useState(JSON.parse(JSON.stringify(record)))
  
  // Generic field renderer that handles any schema
  const renderField = (key, value, path = []) => {
    const fullPath = [...path, key]
    const updateField = (newValue) => {
      setEditedRecord(prev => {
        const updated = JSON.parse(JSON.stringify(prev))
        let obj = updated
        for (let i = 0; i < path.length; i++) {
          obj = obj[path[i]]
        }
        obj[key] = newValue
        return updated
      })
    }
    
    // Skip internal id field (used by table)
    if (key === 'id' && path.length === 0) return null
    
    // Handle different value types
    if (value === null || value === undefined) {
      return (
        <div key={fullPath.join('.')} className="form-group">
          <label>{formatLabel(key)}</label>
          <input value="" onChange={e => updateField(e.target.value)} placeholder="(empty)" />
        </div>
      )
    }
    
    if (typeof value === 'boolean') {
      return (
        <div key={fullPath.join('.')} className="form-group">
          <label>{formatLabel(key)}</label>
          <select value={value.toString()} onChange={e => updateField(e.target.value === 'true')}>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </div>
      )
    }
    
    if (typeof value === 'number') {
      return (
        <div key={fullPath.join('.')} className="form-group">
          <label>{formatLabel(key)}</label>
          <input 
            type="number" 
            value={value} 
            onChange={e => updateField(parseFloat(e.target.value) || 0)} 
          />
        </div>
      )
    }
    
    if (typeof value === 'string') {
      // Check if it's a long string
      if (value.length > 100) {
        return (
          <div key={fullPath.join('.')} className="form-group full-width">
            <label>{formatLabel(key)}</label>
            <textarea 
              value={value} 
              onChange={e => updateField(e.target.value)}
              rows={3}
            />
          </div>
        )
      }
      return (
        <div key={fullPath.join('.')} className="form-group">
          <label>{formatLabel(key)}</label>
          <input value={value} onChange={e => updateField(e.target.value)} />
        </div>
      )
    }
    
    if (Array.isArray(value)) {
      return (
        <div key={fullPath.join('.')} className="form-section">
          <div className="form-section-title">{formatLabel(key)} ({value.length} items)</div>
          <div className="nested-data">
            {value.slice(0, 10).map((item, idx) => (
              <div key={idx} className="nested-item">
                {typeof item === 'object' ? (
                  <div className="nested-object-preview">
                    {Object.entries(item).slice(0, 4).map(([k, v]) => (
                      <span key={k} className="nested-field">
                        <span className="nested-label">{formatLabel(k)}:</span>
                        <span className="nested-value">{typeof v === 'object' ? '{...}' : String(v)}</span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="nested-value">{String(item)}</span>
                )}
              </div>
            ))}
            {value.length > 10 && (
              <p className="more-items">...and {value.length - 10} more items</p>
            )}
          </div>
        </div>
      )
    }
    
    if (typeof value === 'object') {
      const entries = Object.entries(value)
      // For small objects with primitive values, show inline
      if (entries.length <= 4 && entries.every(([, v]) => typeof v !== 'object')) {
        return (
          <div key={fullPath.join('.')} className="form-section">
            <div className="form-section-title">{formatLabel(key)}</div>
            <div className="form-row">
              {entries.map(([k, v]) => renderField(k, v, fullPath))}
            </div>
          </div>
        )
      }
      // For larger objects
      return (
        <div key={fullPath.join('.')} className="form-section">
          <div className="form-section-title">{formatLabel(key)} ({entries.length} fields)</div>
          <div className="nested-data">
            {entries.slice(0, 10).map(([k, v]) => (
              <div key={k} className="nested-item">
                <span className="nested-label">{formatLabel(k)}:</span>
                <span className="nested-value">
                  {typeof v === 'object' 
                    ? (Array.isArray(v) ? `[${v.length} items]` : '{...}')
                    : String(v)}
                </span>
              </div>
            ))}
            {entries.length > 10 && (
              <p className="more-items">...and {entries.length - 10} more fields</p>
            )}
          </div>
        </div>
      )
    }
    
    return null
  }
  
  // Get display title for the record
  const getRecordTitle = () => {
    // Try common naming patterns
    if (record.name) {
      if (typeof record.name === 'object') {
        return `${record.name.first_name || ''} ${record.name.last_name || ''}`.trim()
      }
      return record.name
    }
    // Try ID fields
    const idFields = ['user_id', 'product_id', 'order_id', 'flight_number', 'id', 'reservation_id']
    for (const field of idFields) {
      if (record[field]) return record[field]
    }
    return 'Record'
  }
  
  // Separate fields into primitive and complex
  const entries = Object.entries(editedRecord).filter(([key]) => key !== 'id')
  const primitiveFields = entries.filter(([, v]) => 
    v === null || v === undefined || typeof v !== 'object'
  )
  const complexFields = entries.filter(([, v]) => 
    v !== null && v !== undefined && typeof v === 'object'
  )
  
  return (
    <div className="record-detail-panel">
      <div className="panel-header">
        <h3>{formatLabel(type.replace(/s$/, ''))}: {getRecordTitle()}</h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>
      
      <div className="panel-body">
        {/* Render primitive fields in a grid */}
        {primitiveFields.length > 0 && (
          <div className="form-row flex-wrap">
            {primitiveFields.map(([key, value]) => renderField(key, value))}
          </div>
        )}
        
        {/* Render complex fields (objects and arrays) */}
        {complexFields.map(([key, value]) => renderField(key, value))}
      </div>
      
      <div className="panel-footer">
        <button className="cancel-btn" onClick={onClose}>Close</button>
        <button className="save-btn" onClick={() => onSave(editedRecord)}>Save Changes</button>
      </div>
    </div>
  )
}

function DatabaseEditor({ content, onChange }) {
  const [db, setDb] = useState(() => {
    try {
      return JSON.parse(content)
    } catch {
      return {}
    }
  })
  
  // Auto-discover tabs from the database schema
  const tabs = useMemo(() => {
    return Object.keys(db)
      .filter(key => typeof db[key] === 'object' && db[key] !== null && !Array.isArray(db[key]))
      .map(key => ({
        id: key,
        label: formatLabel(key),
        icon: ICON_MAP[key.toLowerCase()] || ICON_MAP.default
      }))
  }, [db])
  
  const [activeTab, setActiveTab] = useState(() => tabs[0]?.id || '')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedRecord, setSelectedRecord] = useState(null)
  const [showRawJson, setShowRawJson] = useState(false)
  
  // Update active tab when tabs change (e.g., when loading new content)
  useMemo(() => {
    if (tabs.length > 0 && !tabs.find(t => t.id === activeTab)) {
      setActiveTab(tabs[0].id)
    }
  }, [tabs, activeTab])
  
  // Convert object to array for table display
  const getDataArray = (type) => {
    const data = db[type] || {}
    if (Array.isArray(data)) {
      return data.map((record, idx) => ({ id: idx, ...record }))
    }
    return Object.entries(data).map(([id, record]) => ({
      id,
      ...(typeof record === 'object' ? record : { value: record }),
    }))
  }
  
  // Filter data based on search
  const filteredData = useMemo(() => {
    const data = getDataArray(activeTab)
    if (!searchTerm) return data
    
    const term = searchTerm.toLowerCase()
    return data.filter(record => 
      JSON.stringify(record).toLowerCase().includes(term)
    )
  }, [db, activeTab, searchTerm])
  
  // Auto-discover columns for current tab
  const columns = useMemo(() => {
    return discoverColumns(filteredData)
  }, [filteredData])
  
  const handleRecordSave = (updatedRecord) => {
    // Find the key field for this record
    const idFields = ['user_id', 'product_id', 'order_id', 'flight_number', 'reservation_id', 'id']
    let key = updatedRecord.id
    
    for (const field of idFields) {
      if (updatedRecord[field]) {
        key = updatedRecord[field]
        break
      }
    }
    
    // Remove the internal 'id' field we added for table tracking
    const recordToSave = { ...updatedRecord }
    delete recordToSave.id
    
    const newDb = {
      ...db,
      [activeTab]: {
        ...db[activeTab],
        [key]: recordToSave
      }
    }
    setDb(newDb)
    onChange(JSON.stringify(newDb, null, 2))
    setSelectedRecord(null)
  }
  
  const handleRawJsonChange = (e) => {
    try {
      const parsed = JSON.parse(e.target.value)
      setDb(parsed)
      onChange(e.target.value)
    } catch {
      onChange(e.target.value)
    }
  }
  
  if (showRawJson) {
    return (
      <div className="database-editor raw-mode">
        <div className="editor-toolbar">
          <button className="toggle-view-btn" onClick={() => setShowRawJson(false)}>
            ← Back to Table View
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
  
  // Handle empty or invalid JSON
  if (tabs.length === 0) {
    return (
      <div className="database-editor">
        <div className="editor-toolbar">
          <div className="toolbar-right">
            <button className="toggle-view-btn" onClick={() => setShowRawJson(true)}>
              Edit Raw JSON
            </button>
          </div>
        </div>
        <div className="empty-state">
          <p>No collections found in this database.</p>
          <p>Click "Edit Raw JSON" to add data.</p>
        </div>
      </div>
    )
  }
  
  return (
    <div className="database-editor">
      <div className="editor-toolbar">
        <div className="tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => { setActiveTab(tab.id); setSelectedRecord(null); setSearchTerm(''); }}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span className="tab-label">{tab.label}</span>
              <span className="tab-count">{Object.keys(db[tab.id] || {}).length}</span>
            </button>
          ))}
        </div>
        <div className="toolbar-right">
          <div className="search-box">
            <input
              type="text"
              placeholder={`Search ${activeTab}...`}
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>
          <button className="toggle-view-btn" onClick={() => setShowRawJson(true)}>
            Edit Raw JSON
          </button>
        </div>
      </div>
      
      <div className="database-content">
        <div className={`table-area ${selectedRecord ? 'with-detail' : ''}`}>
          <DataTable
            data={filteredData}
            columns={columns}
            onRowClick={setSelectedRecord}
            selectedId={selectedRecord?.id}
          />
        </div>
        
        {selectedRecord && (
          <RecordDetail
            record={selectedRecord}
            type={activeTab}
            onClose={() => setSelectedRecord(null)}
            onSave={handleRecordSave}
          />
        )}
      </div>
    </div>
  )
}

export default DatabaseEditor
