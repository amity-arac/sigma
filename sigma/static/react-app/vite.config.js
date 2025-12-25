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
  base: '/assets/',
  build: {
    // Build directly into the static/assets folder that the backend serves
    outDir: '../assets',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Output directly in assets folder, not assets/assets
        assetFileNames: '[name][extname]',
        chunkFileNames: '[name].js',
        entryFileNames: '[name].js',
      }
    }
  }
})
