import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发模式下把 API 请求代理到后端 8000 端口，避免跨域
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/qa': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
