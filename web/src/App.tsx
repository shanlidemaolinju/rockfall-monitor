/**
 * 应用根组件 — React Router 路由配置
 * 使用 React.lazy + Suspense 实现路由级代码分割
 * v2.3: 增加登录页 + 路由鉴权守卫
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { ConfigProvider, theme, App as AntApp } from 'antd';
import ErrorBoundary from './components/common/ErrorBoundary';
import AppLayout from './components/Layout/AppLayout';
import Loading from './components/common/Loading';
import { useAppStore } from './stores/useAppStore';

// ─── 路由懒加载 ───
const Cockpit = lazy(() => import('./pages/Cockpit'));
const AlertRecords = lazy(() => import('./pages/AlertRecords'));
const SiteManagement = lazy(() => import('./pages/SiteManagement'));
const VideoDetection = lazy(() => import('./pages/VideoDetection'));
const MapView = lazy(() => import('./pages/MapView'));
const RoiCalibration = lazy(() => import('./pages/RoiCalibration'));
const Settings = lazy(() => import('./pages/Settings'));
const LoginPage = lazy(() => import('./pages/LoginPage'));

/**
 * 路由鉴权守卫 — 未登录时重定向到 /login
 * 同时从 localStorage 恢复 token（防止刷新丢失状态）
 */
function AuthGuard({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated);
  const token = useAppStore((s) => s.token);
  const location = useLocation();

  // 如果有 token 但未标记已认证，自动恢复
  const effectiveAuth = isAuthenticated || !!token;

  if (!effectiveAuth) {
    const redirect = location.pathname === '/' ? '' : `?redirect=${encodeURIComponent(location.pathname)}`;
    return <Navigate to={`/login${redirect}`} replace />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#58a6ff',
          colorBgBase: '#0d1117',
          colorBgContainer: '#161b22',
          colorBorder: '#30363d',
          colorText: '#c9d1d9',
          colorTextSecondary: '#8b949e',
          borderRadius: 6,
          fontFamily: `-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif`,
        },
      }}
    >
      <AntApp>
        <ErrorBoundary>
          <BrowserRouter>
            <Suspense fallback={<Loading loading={true} />}>
              <Routes>
                {/* 登录页 — 无需认证 */}
                <Route path="/login" element={<LoginPage />} />

                {/* 受保护的路由 */}
                <Route
                  element={
                    <AuthGuard>
                      <AppLayout />
                    </AuthGuard>
                  }
                >
                  <Route path="/" element={<Cockpit />} />
                  <Route path="/alerts" element={<AlertRecords />} />
                  <Route path="/sites" element={<SiteManagement />} />
                  <Route path="/map" element={<MapView />} />
                  <Route path="/video-detect" element={<VideoDetection />} />
                  <Route path="/roi" element={<RoiCalibration />} />
                  <Route path="/settings" element={<Settings />} />
                </Route>

                {/* 未匹配路由 → 重定向首页 */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </BrowserRouter>
        </ErrorBoundary>
      </AntApp>
    </ConfigProvider>
  );
}
