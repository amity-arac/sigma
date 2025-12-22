import { useState, useMemo } from 'react'
import './DatabaseEditor.css'

// Tabs for different data types
const TABS = [
  { id: 'users', label: 'Users', icon: '👤' },
  { id: 'products', label: 'Products', icon: '📦' },
  { id: 'orders', label: 'Orders', icon: '🛒' },
]

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
  
  const renderUserFields = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>User ID</label>
          <input value={editedRecord.user_id || ''} disabled />
        </div>
        <div className="form-group">
          <label>Email</label>
          <input 
            value={editedRecord.email || ''} 
            onChange={e => setEditedRecord(r => ({...r, email: e.target.value}))}
          />
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label>First Name</label>
          <input 
            value={editedRecord.name?.first_name || ''} 
            onChange={e => setEditedRecord(r => ({...r, name: {...r.name, first_name: e.target.value}}))}
          />
        </div>
        <div className="form-group">
          <label>Last Name</label>
          <input 
            value={editedRecord.name?.last_name || ''} 
            onChange={e => setEditedRecord(r => ({...r, name: {...r.name, last_name: e.target.value}}))}
          />
        </div>
      </div>
      <div className="form-section-title">Address</div>
      <div className="form-row">
        <div className="form-group flex-2">
          <label>Address 1</label>
          <input 
            value={editedRecord.address?.address1 || ''} 
            onChange={e => setEditedRecord(r => ({...r, address: {...r.address, address1: e.target.value}}))}
          />
        </div>
        <div className="form-group">
          <label>Address 2</label>
          <input 
            value={editedRecord.address?.address2 || ''} 
            onChange={e => setEditedRecord(r => ({...r, address: {...r.address, address2: e.target.value}}))}
          />
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label>City</label>
          <input 
            value={editedRecord.address?.city || ''} 
            onChange={e => setEditedRecord(r => ({...r, address: {...r.address, city: e.target.value}}))}
          />
        </div>
        <div className="form-group">
          <label>State</label>
          <input 
            value={editedRecord.address?.state || ''} 
            onChange={e => setEditedRecord(r => ({...r, address: {...r.address, state: e.target.value}}))}
          />
        </div>
        <div className="form-group">
          <label>ZIP</label>
          <input 
            value={editedRecord.address?.zip || ''} 
            onChange={e => setEditedRecord(r => ({...r, address: {...r.address, zip: e.target.value}}))}
          />
        </div>
      </div>
      <div className="form-section-title">Payment Methods ({Object.keys(editedRecord.payment_methods || {}).length})</div>
      <div className="nested-data">
        {Object.entries(editedRecord.payment_methods || {}).map(([id, method]) => (
          <div key={id} className="nested-item">
            <span className="nested-label">{method.source}</span>
            <span className="nested-value">{id}</span>
            {method.balance !== undefined && <span className="nested-badge">${method.balance}</span>}
          </div>
        ))}
      </div>
    </>
  )
  
  const renderProductFields = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>Product ID</label>
          <input value={editedRecord.product_id || ''} disabled />
        </div>
        <div className="form-group flex-2">
          <label>Name</label>
          <input 
            value={editedRecord.name || ''} 
            onChange={e => setEditedRecord(r => ({...r, name: e.target.value}))}
          />
        </div>
      </div>
      <div className="form-section-title">Variants ({(editedRecord.variants || []).length})</div>
      <div className="variants-list">
        {(editedRecord.variants || []).slice(0, 10).map((variant, idx) => (
          <div key={idx} className="variant-item">
            <div className="variant-header">
              <span className="variant-id">{variant.item_id}</span>
              <span className="variant-price">${variant.price}</span>
              <span className={`variant-stock ${variant.available ? 'in-stock' : 'out-stock'}`}>
                {variant.available ? 'In Stock' : 'Out of Stock'}
              </span>
            </div>
            <div className="variant-options">
              {Object.entries(variant.options || {}).map(([key, value]) => (
                <span key={key} className="option-tag">{key}: {value}</span>
              ))}
            </div>
          </div>
        ))}
        {(editedRecord.variants || []).length > 10 && (
          <p className="more-items">...and {editedRecord.variants.length - 10} more variants</p>
        )}
      </div>
    </>
  )
  
  const renderOrderFields = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>Order ID</label>
          <input value={editedRecord.order_id || ''} disabled />
        </div>
        <div className="form-group">
          <label>User ID</label>
          <input value={editedRecord.user_id || ''} disabled />
        </div>
        <div className="form-group">
          <label>Status</label>
          <select 
            value={editedRecord.status || ''} 
            onChange={e => setEditedRecord(r => ({...r, status: e.target.value}))}
          >
            <option value="pending">Pending</option>
            <option value="processed">Processed</option>
            <option value="delivered">Delivered</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
      </div>
      <div className="form-section-title">Items ({(editedRecord.items || []).length})</div>
      <div className="items-list">
        {(editedRecord.items || []).map((item, idx) => (
          <div key={idx} className="order-item">
            <span className="item-name">{item.name}</span>
            <span className="item-price">${item.price}</span>
            <div className="item-options">
              {Object.entries(item.options || {}).map(([key, value]) => (
                <span key={key} className="option-tag">{key}: {value}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="form-section-title">Payment History</div>
      <div className="payment-history">
        {(editedRecord.payment_history || []).map((payment, idx) => (
          <div key={idx} className="payment-item">
            <span className="payment-type">{payment.transaction_type}</span>
            <span className="payment-amount">${payment.amount}</span>
            <span className="payment-method">{payment.payment_method_id}</span>
          </div>
        ))}
      </div>
    </>
  )
  
  return (
    <div className="record-detail-panel">
      <div className="panel-header">
        <h3>
          {type === 'users' && `User: ${record.name?.first_name} ${record.name?.last_name}`}
          {type === 'products' && `Product: ${record.name}`}
          {type === 'orders' && `Order: ${record.order_id}`}
        </h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>
      
      <div className="panel-body">
        {type === 'users' && renderUserFields()}
        {type === 'products' && renderProductFields()}
        {type === 'orders' && renderOrderFields()}
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
      return { users: {}, products: {}, orders: {} }
    }
  })
  const [activeTab, setActiveTab] = useState('users')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedRecord, setSelectedRecord] = useState(null)
  const [showRawJson, setShowRawJson] = useState(false)
  
  // Convert object to array for table display
  const getDataArray = (type) => {
    const data = db[type] || {}
    return Object.entries(data).map(([id, record]) => ({
      id,
      ...record,
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
  
  // Column definitions for each type
  const getColumns = (type) => {
    switch (type) {
      case 'users':
        return [
          { key: 'user_id', label: 'User ID' },
          { key: 'name', label: 'Name', render: (_, row) => `${row.name?.first_name || ''} ${row.name?.last_name || ''}` },
          { key: 'email', label: 'Email' },
          { key: 'address', label: 'City', render: (_, row) => row.address?.city || '-' },
          { key: 'orders', label: 'Orders', render: (orders) => (orders || []).length },
        ]
      case 'products':
        return [
          { key: 'product_id', label: 'Product ID' },
          { key: 'name', label: 'Name' },
          { key: 'variants', label: 'Variants', render: (variants) => (variants || []).length },
          { key: 'price', label: 'Price Range', render: (_, row) => {
            const prices = (row.variants || []).map(v => v.price).filter(Boolean)
            if (prices.length === 0) return '-'
            return `$${Math.min(...prices)} - $${Math.max(...prices)}`
          }},
        ]
      case 'orders':
        return [
          { key: 'order_id', label: 'Order ID' },
          { key: 'user_id', label: 'User ID' },
          { key: 'status', label: 'Status', render: (status) => (
            <span className={`status-badge ${status}`}>{status}</span>
          )},
          { key: 'items', label: 'Items', render: (items) => (items || []).length },
          { key: 'total', label: 'Total', render: (_, row) => {
            const total = (row.items || []).reduce((sum, item) => sum + (item.price || 0), 0)
            return `$${total.toFixed(2)}`
          }},
        ]
      default:
        return []
    }
  }
  
  const handleRecordSave = (updatedRecord) => {
    const key = activeTab === 'users' ? updatedRecord.user_id :
                activeTab === 'products' ? updatedRecord.product_id :
                updatedRecord.order_id
    
    const newDb = {
      ...db,
      [activeTab]: {
        ...db[activeTab],
        [key]: updatedRecord
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
  
  return (
    <div className="database-editor">
      <div className="editor-toolbar">
        <div className="tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => { setActiveTab(tab.id); setSelectedRecord(null); }}
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
            columns={getColumns(activeTab)}
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
