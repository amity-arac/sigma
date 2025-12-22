import React, { useState, useEffect } from 'react';
import { fetchEnvironments, fetchEnvironmentFiles, fetchEnvironmentFile, updateEnvironmentFile } from '../../services/api';
import TasksEditor from './TasksEditor';
import DatabaseEditor from './DatabaseEditor';
import './EnvironmentsPage.css';

const EnvironmentsPage = () => {
  const [environments, setEnvironments] = useState([]);
  const [selectedEnv, setSelectedEnv] = useState('');
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Load environments on mount
  useEffect(() => {
    const loadEnvironments = async () => {
      try {
        const data = await fetchEnvironments();
        const envNames = data.map(env => env.name);
        setEnvironments(envNames);
        if (envNames.length > 0 && !selectedEnv) {
          setSelectedEnv(envNames[0]);
        }
      } catch (err) {
        setError('Failed to load environments');
        console.error(err);
      }
    };
    loadEnvironments();
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
      setSelectedFile(null);
      setFileContent('');
      setOriginalContent('');
    } catch (err) {
      setError('Failed to load environment files');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadFileContent = async (file) => {
    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const data = await fetchEnvironmentFile(selectedEnv, file.name);
      const content = typeof data.content === 'string' 
        ? data.content 
        : JSON.stringify(data.content, null, 2);
      setFileContent(content);
      setOriginalContent(content);
      setSelectedFile(file);
    } catch (err) {
      setError(`Failed to load file: ${file.name}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!selectedFile) return;
    
    setSaving(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      await updateEnvironmentFile(selectedEnv, selectedFile.name, fileContent);
      setOriginalContent(fileContent);
      setSuccessMessage('File saved successfully!');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(`Failed to save file: ${err.message}`);
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = fileContent !== originalContent;

  const getFileIcon = (filename) => {
    if (filename.endsWith('.json')) return '📋';
    if (filename.endsWith('.md')) return '📝';
    if (filename.endsWith('.py')) return '🐍';
    return '📄';
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Render the appropriate editor based on file type
  const renderEditor = () => {
    if (!selectedFile) {
      return (
        <div className="no-file-selected">
          <p>Select a file from the list to edit</p>
        </div>
      );
    }

    if (loading) {
      return <div className="loading">Loading file content...</div>;
    }

    // Handle special editors for specific files
    if (selectedFile.name === 'tasks.json') {
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

    if (selectedFile.name === 'db.json') {
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

    // Default: Textarea editor for markdown and other files
    return (
      <div className="text-editor">
        <textarea
          value={fileContent}
          onChange={(e) => setFileContent(e.target.value)}
          disabled={!selectedFile.editable}
          spellCheck={selectedFile.name.endsWith('.md')}
        />
        <div className="editor-actions">
          <button 
            className="save-btn"
            onClick={handleSave}
            disabled={!hasChanges || saving || !selectedFile.editable}
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
      <div className="page-header">
        <h1>Environment Configuration</h1>
        <p>Manage environment data files, tasks, and settings</p>
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
        <div className="env-sidebar">
          <div className="env-selector">
            <label>Environment:</label>
            <select 
              value={selectedEnv} 
              onChange={(e) => setSelectedEnv(e.target.value)}
            >
              {environments.map(env => (
                <option key={env} value={env}>{env}</option>
              ))}
            </select>
          </div>

          <div className="files-list">
            <h3>Files</h3>
            {loading && !selectedFile ? (
              <div className="loading">Loading...</div>
            ) : (
              <ul>
                {files.map(file => (
                  <li 
                    key={file.name}
                    className={`file-item ${selectedFile?.name === file.name ? 'selected' : ''} ${!file.editable ? 'readonly' : ''}`}
                    onClick={() => loadFileContent(file)}
                  >
                    <span className="file-icon">{getFileIcon(file.name)}</span>
                    <div className="file-info">
                      <span className="file-name">{file.display_name || file.name}</span>
                      <span className="file-size">{formatFileSize(file.size)}</span>
                    </div>
                    {!file.editable && <span className="readonly-badge">Read Only</span>}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="editor-panel">
          {selectedFile && (
            <div className="editor-header">
              <h2>
                {getFileIcon(selectedFile.name)} {selectedFile.display_name || selectedFile.name}
                {hasChanges && <span className="unsaved-indicator">•</span>}
              </h2>
              <span className="file-description">{selectedFile.description}</span>
            </div>
          )}
          
          <div className="editor-content">
            {renderEditor()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnvironmentsPage;
