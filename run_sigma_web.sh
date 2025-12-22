#!/bin/bash

# Environment variables are loaded from .env file via python-dotenv
# See .env.example for required variables

# Run sigma with auto-reload
# - React app: builds on file changes using vite build --watch
# - Python: uvicorn watches for .py file changes

# Store the root directory
ROOT_DIR="$(pwd)"

# Set PYTHONPATH so 'sigma' module can be imported from 'src' directory
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Build React app once first (so dist/assets exists before server starts)
echo "Building React app..."
cd sigma/static/react-app && npm run build

# # Now start watch mode in background (for subsequent changes)
# npm run build -- --watch &
# REACT_PID=$!

# # Go back to root directory
# cd "$ROOT_DIR"

# # Cleanup function to kill React build watcher when script exits
# cleanup() {
#     echo "Shutting down..."
#     kill $REACT_PID 2>/dev/null
#     exit 0
# }
# trap cleanup SIGINT SIGTERM

cd ../../..

# Run uvicorn with auto-reload for Python files (using 'sigma' as module name)
uvicorn sigma.api_server:app --port 8001 --reload --reload-dir sigma