import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/actions': 'http://127.0.0.1:8000',
      '/node-runs': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/uploads': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
});
