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
    // Build into parent static folder
    outDir: 'dist',
    emptyOutDir: false,
    rollupOptions: {
      output: {
        // Ensure assets go directly to the assets folder
        assetFileNames: 'assets/[name]-[hash][extname]',
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
      }
    }
  }
})
