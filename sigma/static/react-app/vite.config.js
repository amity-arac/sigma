import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/sessions': 'http://localhost:8000',
      '/environments': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  },
  build: {
    // Build directly into the static/assets folder that the backend serves
    outDir: '../assets',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Use consistent naming without hashes for cleaner git diffs
        assetFileNames: 'assets/[name][extname]',
        chunkFileNames: 'assets/[name].js',
        entryFileNames: 'assets/[name].js',
      }
    }
  }
})
