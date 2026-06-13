/**
 * 应用根组件 — React Router 路由配置
 * 使用 React.lazy + Suspense 实现路由级代码分割
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider, theme, App as AntApp } from 'antd';
import ErrorBoundary from './components/common/ErrorBoundary';
import AppLayout from './components/Layout/AppLayout';
import Loading from './components/common/Loading';

// ─── 路由懒加载 ───
const Cockpit = lazy(() => import('./pages/Cockpit'));
const AlertRecords = lazy(() => import('./pages/AlertRecords'));
const SiteManagement = lazy(() => import('./pages/SiteManagement'));
const VideoDetection = lazy(() => import('./pages/VideoDetection'));
const MapView = lazy(() => import('./pages/MapView'));
const RoiCalibration = lazy(() => import('./pages/RoiCalibration'));
const Settings = lazy(() => import('./pages/Settings'));

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
                <Route element={<AppLayout />}>
                  <Route path="/" element={<Cockpit />} />
                  <Route path="/alerts" element={<AlertRecords />} />
                  <Route path="/sites" element={<SiteManagement />} />
                  <Route path="/map" element={<MapView />} />
                  <Route path="/video-detect" element={<VideoDetection />} />
                  <Route path="/roi" element={<RoiCalibration />} />
                  <Route path="/settings" element={<Settings />} />
                </Route>
              </Routes>
            </Suspense>
          </BrowserRouter>
        </ErrorBoundary>
      </AntApp>
    </ConfigProvider>
  );
}
