import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/actions': 'http://127.0.0.1:7860',
      '/config': 'http://127.0.0.1:7860',
      '/gradio_api': 'http://127.0.0.1:7860',
      '/health': 'http://127.0.0.1:7860',
      '/node-runs': 'http://127.0.0.1:7860',
    },
  },
});
