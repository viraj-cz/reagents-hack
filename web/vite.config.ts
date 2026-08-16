import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API lives in the Python process (`uv run resolution`). In development the
// two are separate servers, so everything under /api is proxied there rather
// than hard-coding a port in the client -- which would also break the built
// bundle, where both are served from the same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        // Without this, the dev proxy buffers text/event-stream and the run
        // appears to finish in one frame at the end.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache, no-transform'
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
