import React, { useState, useEffect } from 'react';
import { fetchEnvironments, fetchEnvironmentFiles, fetchEnvironmentFile, updateEnvironmentFile, duplicateEnvironment, renameEnvironment, deleteEnvironment } from '../../services/api';
import TasksEditor from './TasksEditor';
import DatabaseEditor from './DatabaseEditor';
import './EnvironmentsPage.css';

// Configuration sections - maps file names to conceptual sections
const CONFIG_SECTIONS = {
  'db.json': {
    id: 'database',
    icon: '🗄️',
    title: 'Database',
    description: 'Users, products, orders, and other domain data',
    color: '#22c55e',
    detailedDescription: 'This JSON file contains all the simulated data for your environment - customer accounts, orders, products, inventory, etc. The agent can read and modify this data using tools. Changes here affect what data is available during simulations.'
  },
  'tasks.json': {
    id: 'tasks',
    icon: '📋',
    title: 'Tasks',
    description: 'Test scenarios with user instructions and expected actions',
    color: '#f59e0b',
    detailedDescription: 'Define test scenarios that simulate customer interactions. Each task includes: user persona/instructions (what the simulated customer wants), expected agent actions (what a correct agent should do), and optional reward criteria for evaluating agent performance.'
  },
  'policy.md': {
    id: 'policy',
    icon: '📜',
    title: 'Agent Policy',
    description: 'Rules, guidelines, and behavioral instructions for the agent',
    color: '#6366f1',
    detailedDescription: 'The main policy document that tells the agent how to handle customer requests. Includes: agent guidelines (tone, response style, empathy), business rules, authentication requirements, refund/exchange policies, escalation procedures, and compliance guidelines. This is the "source of truth" for agent behavior and is also used by the Auto-Approve AI to validate actions.'
  },
  'user_guidelines.md': {
    id: 'user',
    icon: '👤',
    title: 'User Simulation',
    description: 'Guidelines for how simulated users behave during testing',
    color: '#ec4899',
    detailedDescription: 'Instructions for the AI that simulates customers during testing. Controls how realistic/difficult the simulated customer is - do they provide information upfront or make the agent ask? How patient are they? Do they follow instructions? Affects testing realism.'
  },
  'scenario_generator_guidelines.md': {
    id: 'scenario_generator',
    icon: '🎲',
    title: 'Scenario Generator',
    description: 'Guidelines for generating valid, solvable test scenarios',
    color: '#8b5cf6',
    detailedDescription: 'Optional guidelines that help the scenario generator create valid scenarios. Define tool limitations, valid scenario types, scenarios to avoid, and data requirements. This helps prevent generating scenarios that are impossible to solve due to tool constraints.',
    optional: true
  },
  'tools.py': {
    id: 'tools',
    icon: '🔧',
    title: 'Tools & Actions',
    description: 'Python implementations of tools available to the agent',
    color: '#06b6d4',
    detailedDescription: 'Python code defining what actions the agent can take - looking up orders, processing refunds, checking inventory, etc. Each tool has a name, description, parameters, and implementation. Adding new tools here expands what the agent can do.'
  }
};

// Order of sections in the sidebar
const SECTION_ORDER = ['db.json', 'tasks.json', 'policy.md', 'user_guidelines.md', 'scenario_generator_guidelines.md', 'tools.py'];

const EnvironmentsPage = () => {
  const [environments, setEnvironments] = useState([]);
  const [selectedEnv, setSelectedEnv] = useState('');
  const [files, setFiles] = useState([]);
  const [selectedSection, setSelectedSection] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  
  // Modal state for duplicate/rename
  const [showModal, setShowModal] = useState(null);
  const [newEnvName, setNewEnvName] = useState('');
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState(null);

  // Load environments on mount
  const loadEnvironments = async () => {
    try {
      const data = await fetchEnvironments();
      const envNames = data.map(env => env.name);
      setEnvironments(envNames);
      return envNames;
    } catch (err) {
      setError('Failed to load environments');
      console.error(err);
      return [];
    }
  };

  useEffect(() => {
    const init = async () => {
      const envNames = await loadEnvironments();
      if (envNames.length > 0 && !selectedEnv) {
        setSelectedEnv(envNames[0]);
      }
    };
    init();
  }, []);

  // Load files when environment changes
  useEffect(() => {
    if (selectedEnv) {
      loadFiles();
    }
  }, [selectedEnv]);

  const loadFiles = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchEnvironmentFiles(selectedEnv);
      setFiles(data.files || []);
      setSelectedSection(null);
      setFileContent('');
      setOriginalContent('');
    } catch (err) {
      setError('Failed to load environment configuration');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadSectionContent = async (fileName) => {
    const file = files.find(f => f.name === fileName);
    const config = CONFIG_SECTIONS[fileName];
    
    // For optional files that don't exist yet, show empty editor to create them
    if (!file && config?.optional) {
      setLoading(false);
      setError(null);
      setSuccessMessage(null);
      setFileContent('');
      setOriginalContent('');
      setSelectedSection({ name: fileName, ...config, isNew: true, editable: true });
      return;
    }
    
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const data = await fetchEnvironmentFile(selectedEnv, fileName);
      const content = typeof data.content === 'string' 
        ? data.content 
        : JSON.stringify(data.content, null, 2);
      setFileContent(content);
      setOriginalContent(content);
      setSelectedSection({ ...file, ...config });
    } catch (err) {
      setError(`Failed to load ${config?.title || fileName}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSave = async () => {
    if (!selectedSection) return;
    
    setSaving(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      await updateEnvironmentFile(selectedEnv, selectedSection.name, fileContent);
      setOriginalContent(fileContent);
      
      // If this was a new file, refresh the file list and update the section
      if (selectedSection.isNew) {
        await loadFiles();
        // Update the section to mark it as no longer new
        setSelectedSection(prev => ({ ...prev, isNew: false }));
        setSuccessMessage(`${selectedSection.title} created successfully!`);
      } else {
        setSuccessMessage(`${selectedSection.title} saved successfully!`);
      }
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(`Failed to save: ${err.message}`);
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = fileContent !== originalContent;

  // Modal handlers
  const handleDuplicateClick = () => {
    setNewEnvName(selectedEnv + '_copy');
    setModalError(null);
    setShowModal('duplicate');
  };

  const handleRenameClick = () => {
    setNewEnvName(selectedEnv);
    setModalError(null);
    setShowModal('rename');
  };

  const handleModalClose = () => {
    setShowModal(null);
    setNewEnvName('');
    setModalError(null);
  };

  const handleDuplicateSubmit = async () => {
    if (!newEnvName.trim()) {
      setModalError('Please enter a name');
      return;
    }
    
    const targetEnvName = newEnvName.trim();
    setModalLoading(true);
    setModalError(null);
    
    try {
      await duplicateEnvironment(selectedEnv, targetEnvName);
      // Refresh the environments list
      const updatedEnvs = await loadEnvironments();
      // Verify the new environment exists and switch to it
      if (updatedEnvs.includes(targetEnvName)) {
        setSelectedEnv(targetEnvName);
      }
      setSuccessMessage(`Environment duplicated to '${targetEnvName}'`);
      setTimeout(() => setSuccessMessage(null), 3000);
      handleModalClose();
    } catch (err) {
      setModalError(err.message);
    } finally {
      setModalLoading(false);
    }
  };

  const handleRenameSubmit = async () => {
    if (!newEnvName.trim()) {
      setModalError('Please enter a name');
      return;
    }
    
    const targetEnvName = newEnvName.trim();
    
    if (targetEnvName === selectedEnv) {
      setModalError('New name must be different');
      return;
    }
    
    setModalLoading(true);
    setModalError(null);
    
    try {
      await renameEnvironment(selectedEnv, targetEnvName);
      // Refresh the environments list
      const updatedEnvs = await loadEnvironments();
      // Verify the renamed environment exists and switch to it
      if (updatedEnvs.includes(targetEnvName)) {
        setSelectedEnv(targetEnvName);
      }
      setSuccessMessage(`Environment renamed to '${targetEnvName}'`);
      setTimeout(() => setSuccessMessage(null), 3000);
      handleModalClose();
    } catch (err) {
      setModalError(err.message);
    } finally {
      setModalLoading(false);
    }
  };

  const handleDeleteClick = () => {
    setModalError(null);
    setShowModal('delete');
  };

  const handleDeleteSubmit = async () => {
    setModalLoading(true);
    setModalError(null);
    
    try {
      await deleteEnvironment(selectedEnv);
      // Refresh the environments list
      const updatedEnvs = await loadEnvironments();
      // Switch to first available environment
      if (updatedEnvs.length > 0) {
        setSelectedEnv(updatedEnvs[0]);
      }
      setSuccessMessage(`Environment '${selectedEnv}' deleted`);
      setTimeout(() => setSuccessMessage(null), 3000);
      handleModalClose();
    } catch (err) {
      setModalError(err.message);
    } finally {
      setModalLoading(false);
    }
  };

  const canDelete = environments.length > 1;

  // Get ordered sections based on available files
  const getOrderedSections = () => {
    return SECTION_ORDER
      .filter(fileName => {
        const config = CONFIG_SECTIONS[fileName];
        const fileExists = files.some(f => f.name === fileName);
        // Show optional files even if they don't exist yet (so users can create them)
        return fileExists || config?.optional;
      })
      .map(fileName => {
        const file = files.find(f => f.name === fileName);
        return { fileName, file, config: CONFIG_SECTIONS[fileName] };
      });
  };

  // Render the appropriate editor based on section type
  const renderEditor = () => {
    if (!selectedSection) {
      return (
        <div className="no-section-selected">
          <div className="welcome-content">
            <span className="welcome-icon">⚙️</span>
            <h2>Configure Your Environment</h2>
            <p>Select a configuration section from the left to view and edit its contents.</p>
            <div className="section-hints">
              {getOrderedSections().slice(0, 3).map(({ config }) => (
                <div key={config.id} className="hint-item">
                  <span className="hint-icon">{config.icon}</span>
                  <span className="hint-text">{config.title}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }

    if (loading) {
      return <div className="loading">Loading...</div>;
    }

    // Special editors for structured data
    if (selectedSection.name === 'tasks.json') {
      return (
        <TasksEditor
          content={fileContent}
          onChange={setFileContent}
          onSave={handleSave}
          saving={saving}
          hasChanges={hasChanges}
        />
      );
    }

    if (selectedSection.name === 'db.json') {
      return (
        <DatabaseEditor
          content={fileContent}
          onChange={setFileContent}
          onSave={handleSave}
          saving={saving}
          hasChanges={hasChanges}
        />
      );
    }

    // Text editor for markdown and code files
    return (
      <div className="text-editor">
        <textarea
          value={fileContent}
          onChange={(e) => setFileContent(e.target.value)}
          disabled={!selectedSection.editable}
          spellCheck={selectedSection.name.endsWith('.md')}
          placeholder={`Enter ${selectedSection.title.toLowerCase()} content...`}
        />
        <div className="editor-actions">
          <button 
            className="save-btn"
            onClick={handleSave}
            disabled={!hasChanges || saving || !selectedSection.editable}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
          <button 
            className="reset-btn"
            onClick={() => setFileContent(originalContent)}
            disabled={!hasChanges}
          >
            Reset
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="environments-page">
      {/* Header with environment selector */}
      <div className="page-header">
        <div className="header-content">
          <div className="header-title">
            <h1>Environment Configuration</h1>
            <p>Configure simulation environments for testing agent behaviors</p>
          </div>
          <div className="env-selector-inline">
            <select 
              value={selectedEnv} 
              onChange={(e) => setSelectedEnv(e.target.value)}
            >
              {environments.map(env => (
                <option key={env} value={env}>{env}</option>
              ))}
            </select>
            <button 
              className="header-action-btn" 
              onClick={handleDuplicateClick}
              title="Duplicate environment"
            >
              📋
            </button>
            <button 
              className="header-action-btn" 
              onClick={handleRenameClick}
              title="Rename environment"
            >
              ✏️
            </button>
            <button 
              className="header-action-btn delete-btn" 
              onClick={handleDeleteClick}
              title={canDelete ? "Delete environment" : "Cannot delete the last environment"}
              disabled={!canDelete}
            >
              🗑️
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {successMessage && (
        <div className="success-banner">
          {successMessage}
        </div>
      )}

      <div className="environments-content">
        {/* Configuration Sections Sidebar */}
        <div className="config-sidebar">
          <div className="sections-list">
            {loading && !selectedSection ? (
              <div className="loading">Loading...</div>
            ) : (
              getOrderedSections().map(({ fileName, file, config }) => (
                <div
                  key={fileName}
                  className={`section-item ${selectedSection?.name === fileName ? 'selected' : ''}`}
                  onClick={() => loadSectionContent(fileName)}
                  style={{ '--section-color': config.color }}
                >
                  <div className="section-icon-wrapper">
                    <span className="section-icon">{config.icon}</span>
                  </div>
                  <div className="section-info">
                    <span className="section-title">{config.title}</span>
                    <span className="section-desc">{config.description}</span>
                  </div>
                  {selectedSection?.name === fileName && hasChanges && (
                    <span className="unsaved-dot" title="Unsaved changes">●</span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Editor Panel */}
        <div className="editor-panel">
          {selectedSection && (
            <div className="editor-header" style={{ '--section-color': selectedSection.color }}>
              <div className="editor-header-icon">{selectedSection.icon}</div>
              <div className="editor-header-info">
                <h2>
                  {selectedSection.title}
                  {hasChanges && <span className="unsaved-indicator">•</span>}
                </h2>
                <span className="editor-header-desc">{selectedSection.detailedDescription}</span>
              </div>
            </div>
          )}
          
          <div className="editor-content">
            {renderEditor()}
          </div>
        </div>
      </div>

      {/* Duplicate/Rename Modal */}
      {(showModal === 'duplicate' || showModal === 'rename') && (
        <div className="modal-overlay" onClick={handleModalClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{showModal === 'duplicate' ? 'Duplicate Environment' : 'Rename Environment'}</h3>
              <button className="modal-close" onClick={handleModalClose}>×</button>
            </div>
            <div className="modal-body">
              {showModal === 'duplicate' ? (
                <p>Create a copy of <strong>{selectedEnv}</strong> with a new name:</p>
              ) : (
                <p>Enter a new name for <strong>{selectedEnv}</strong>:</p>
              )}
              <input
                type="text"
                value={newEnvName}
                onChange={(e) => setNewEnvName(e.target.value)}
                placeholder="Enter new environment name"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    showModal === 'duplicate' ? handleDuplicateSubmit() : handleRenameSubmit();
                  } else if (e.key === 'Escape') {
                    handleModalClose();
                  }
                }}
              />
              {modalError && <div className="modal-error">{modalError}</div>}
            </div>
            <div className="modal-actions">
              <button 
                className="modal-btn cancel" 
                onClick={handleModalClose}
                disabled={modalLoading}
              >
                Cancel
              </button>
              <button 
                className="modal-btn primary" 
                onClick={showModal === 'duplicate' ? handleDuplicateSubmit : handleRenameSubmit}
                disabled={modalLoading || !newEnvName.trim()}
              >
                {modalLoading ? 'Processing...' : (showModal === 'duplicate' ? 'Duplicate' : 'Rename')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showModal === 'delete' && (
        <div className="modal-overlay" onClick={handleModalClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Delete Environment</h3>
              <button className="modal-close" onClick={handleModalClose}>×</button>
            </div>
            <div className="modal-body">
              <p>Are you sure you want to delete <strong>{selectedEnv}</strong>?</p>
              <p className="delete-warning">⚠️ This action cannot be undone. All configuration files in this environment will be permanently removed.</p>
              {modalError && <div className="modal-error">{modalError}</div>}
            </div>
            <div className="modal-actions">
              <button 
                className="modal-btn cancel" 
                onClick={handleModalClose}
                disabled={modalLoading}
              >
                Cancel
              </button>
              <button 
                className="modal-btn danger" 
                onClick={handleDeleteSubmit}
                disabled={modalLoading}
              >
                {modalLoading ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnvironmentsPage;
