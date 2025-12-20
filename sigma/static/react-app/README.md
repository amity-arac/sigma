# Sigma React UI

A React-based refactored version of the Sigma simulation UI.

## Running with FastAPI

### Option 1: Production Mode (Recommended)

Build the React app and run FastAPI to serve everything:

```bash
# 1. Build the React app (outputs to sigma/static/)
cd sigma/static/react-app
npm install
npm run build

# 2. Run FastAPI server (from project root)
cd ../../..
python -m sigma.api_server
# Or: uvicorn sigma.api_server:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

### Option 2: Development Mode (Hot Reload)

Run both servers for development with hot reload:

**Terminal 1 - FastAPI Backend:**
```bash
uvicorn sigma.api_server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Vite Dev Server:**
```bash
cd sigma/static/react-app
npm run dev
```

Then open http://localhost:5173 (Vite proxies API calls to FastAPI)

### Option 3: Watch Mode

Build React with watch mode while running FastAPI:

**Terminal 1 - FastAPI:**
```bash
uvicorn sigma.api_server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Vite Watch:**
```bash
cd sigma/static/react-app
npm run build:watch
```

Then open http://localhost:8000 (refresh after changes)

## Project Structure

```
react-app/
├── src/
│   ├── components/
│   │   ├── chat/                    # Chat-related components
│   │   │   ├── ChatPanel.jsx        # Main chat container
│   │   │   ├── ChatHeader.jsx       # Chat header with status/actions
│   │   │   ├── ChatInput.jsx        # Input field for messages
│   │   │   ├── Message.jsx          # Individual message component
│   │   │   ├── MessageList.jsx      # List of messages
│   │   │   ├── StickyUserMessage.jsx# Pinned user request
│   │   │   ├── ActionSuggestion.jsx # Action approval UI
│   │   │   ├── LoadingIndicator.jsx # Loading spinner
│   │   │   └── ToolResultContent.jsx# Tool result formatter
│   │   ├── sidebar/                 # Sidebar components
│   │   │   ├── SidePanel.jsx        # Main sidebar container
│   │   │   ├── PanelCard.jsx        # Collapsible card
│   │   │   ├── ToolsList.jsx        # Available tools list
│   │   │   └── ToolForm.jsx         # Tool parameter form
│   │   ├── common/                  # Shared components
│   │   │   ├── ConfirmDialog.jsx    # Confirmation modal
│   │   │   └── ToastContainer.jsx   # Toast notifications
│   │   ├── Header.jsx               # App header
│   │   ├── SetupPanel.jsx           # Session setup form
│   │   ├── MainContent.jsx          # Main layout container
│   │   └── ResultsPanel.jsx         # Simulation results
│   ├── context/
│   │   ├── SessionContext.jsx       # Session state management
│   │   └── ToastContext.jsx         # Toast notifications state
│   ├── services/
│   │   └── api.js                   # API service functions
│   ├── utils/
│   │   └── helpers.js               # Utility functions
│   ├── styles/
│   │   ├── global.css               # Global styles & CSS variables
│   │   └── components.css           # Shared component styles
│   ├── App.jsx                      # Main app component
│   └── main.jsx                     # Entry point
├── index.html
├── package.json
└── vite.config.js
```

## Getting Started

1. Install dependencies:
   ```bash
   cd sigma/static/react-app
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Architecture

### State Management
- **SessionContext**: Manages simulation session state (session ID, messages, tools, persona, wiki)
- **ToastContext**: Manages toast notification state

### Component Hierarchy
- **App**: Root component with providers
  - **Header**: Application title
  - **SetupPanel**: Configuration form for starting simulations
  - **MainContent**: Main layout with chat and sidebar
    - **ChatPanel**: Chat interface
    - **SidePanel**: Tools, persona, and wiki panels
  - **ResultsPanel**: Simulation results display

### API Integration
All API calls are centralized in `src/services/api.js` for easy maintenance and testing.

## Features

- ✅ Environment selection with descriptions
- ✅ Advanced configuration options
- ✅ Real-time chat interface
- ✅ Action suggestions with approve/reject
- ✅ Tool execution with formatted results
- ✅ Undo functionality
- ✅ New session support
- ✅ Toast notifications
- ✅ Collapsible sidebar panels
- ✅ Sticky user request display
