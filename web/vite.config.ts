import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // FastAPI 后端 API
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket 进度推送
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      // 检测端点 (上传等)
      '/detect': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // 健康检查 & 认证
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // 生产构建输出 → server/static (FastAPI 直接挂载)
  build: {
    outDir: '../server/static',
    emptyOutDir: true,
    sourcemap: false,
    // Terser 压缩（比 esbuild 体积更小，支持 drop_console）
    minify: 'terser',
    terserOptions: {
      compress: { drop_console: true, drop_debugger: true },
    },
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return;
          // Framework core
          if (id.includes('react-dom') || id.includes('react-router')) return 'vendor-react';
          if (id.includes('react')) return 'vendor-react';
          // UI component library
          if (id.includes('antd') || id.includes('@ant-design')) return 'vendor-antd';
          // Chart library (only Cockpit page)
          if (id.includes('echarts')) return 'vendor-echarts';
          // Map library (Cockpit + MapView)
          if (id.includes('leaflet') || id.includes('react-leaflet')) return 'vendor-leaflet';
          // Canvas library (only RoiCalibration page)
          if (id.includes('konva') || id.includes('react-konva')) return 'vendor-konva';
          // Shared utilities (axios, dayjs, zustand, react-query)
          if (id.includes('axios') || id.includes('dayjs') || id.includes('zustand') || id.includes('tanstack')) return 'vendor-utils';
        },
      },
    },
  },
})
