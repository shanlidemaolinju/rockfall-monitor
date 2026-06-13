import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { initSentry } from './services/sentry';
import './index.css';

// Sentry 错误监控（仅 VITE_SENTRY_DSN 配置后启用）
initSentry();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
