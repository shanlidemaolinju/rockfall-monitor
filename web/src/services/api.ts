/**
 * Axios 实例 + API 封装
 * 开发环境下通过 Vite proxy 转发到 FastAPI 后端 (:8000)
 */

import axios from 'axios';
import { captureException } from './sentry';

const api = axios.create({
  baseURL: '/',
  timeout: 120000, // 2分钟超时 (视频上传可能较大)
  headers: { 'Content-Type': 'application/json' },
});

// ── 请求拦截器: 自动附加 API Key ──
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('rockguard_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── 响应拦截器: 统一错误处理 ──
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('rockguard_token');
    } else {
      // 非 401 错误上报 Sentry（401 是正常鉴权流程，不需要上报）
      captureException(err, {
        tags: { source: 'api' },
        extra: {
          url: err.config?.url,
          method: err.config?.method,
          status: err.response?.status,
        },
      });
    }
    return Promise.reject(err);
  },
);

// ════════════════════════════════════════════════
// 看板统计
// ════════════════════════════════════════════════

export interface DashboardStats {
  today_total: number;
  today_red: number;
  today_orange: number;
  today_yellow: number;
  today_blue: number;
  last_count: number | null;
  last_conf: number | null;
  last_alert_level: string | null;
}

export const fetchStats = () => api.get<DashboardStats>('/api/stats').then((r) => r.data);

// ════════════════════════════════════════════════
// 预警记录
// ════════════════════════════════════════════════

export interface AlertItem {
  id: number;
  time: string;
  alert_level: string;
  count: number;
  max_confidence: number;
  track_ids: number[];
  class_summary: string;
  saved_frame: string;
  push_status: string;
  review_status?: string;
}

export interface PagedAlerts {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  rows: AlertItem[];
}

export const fetchAlerts = (limit = 20) =>
  api.get<AlertItem[]>('/api/alerts', { params: { limit } }).then((r) => r.data);

export const fetchAlertsPaged = (params: {
  page?: number;
  page_size?: number;
  start_date?: string;
  end_date?: string;
  alert_level?: string;
}) => api.get<PagedAlerts>('/api/alerts/paged', { params }).then((r) => r.data);

export const reviewAlert = (alertId: number, reviewStatus: string) =>
  api.post(`/api/alerts/${alertId}/review`, { review_status: reviewStatus, note: '' });

export const fetchAlertStatistics = (days = 7) =>
  api.get('/api/statistics', { params: { days } }).then((r) => r.data);

// ════════════════════════════════════════════════
// 点位管理
// ════════════════════════════════════════════════

export interface MonitoringSite {
  site_id: string;
  name: string;
  location: string;
  region: string;
  latitude: number;
  longitude: number;
  highway: string;
  stake_mark: string;
  risk_level: string;
  camera_url: string;
  is_active: boolean;
}

export const fetchSites = () =>
  api.get<{ sites: MonitoringSite[]; active_site_id: string }>('/api/sites').then((r) => r.data);

export const switchSite = (siteId: string) => api.post('/api/sites/switch', { site_id: siteId });

// ════════════════════════════════════════════════
// 视频检测
// ════════════════════════════════════════════════

export interface TaskResponse {
  task_id: string;
  status: string;
}

export interface TaskStatus {
  task_id: string;
  status: string;
  progress: number;
  current_frame: number;
  total_frames: number;
  result?: Record<string, unknown>;
  error?: string;
}

export const uploadVideo = (file: File, options: {
  save_frames: boolean;
  push_alerts: boolean;
  camera_id: string;
}) => {
  const form = new FormData();
  form.append('file', file);
  form.append('save_frames', String(options.save_frames));
  form.append('push_alerts', String(options.push_alerts));
  form.append('camera_id', options.camera_id);
  return api
    .post<TaskResponse>('/detect/video?sync=false', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000, // 10分钟上传超时
    })
    .then((r) => r.data);
};

export const fetchTaskStatus = (taskId: string) =>
  api.get<TaskStatus>(`/api/tasks/${taskId}`).then((r) => r.data);

// ════════════════════════════════════════════════
// 运行时配置
// ════════════════════════════════════════════════

export const fetchRuntimeConfig = () =>
  api.get('/api/config/runtime').then((r) => r.data);

export const updateRuntimeConfig = (config: Record<string, unknown>) =>
  api.post('/api/config/update', config);

// ════════════════════════════════════════════════
// 认证
// ════════════════════════════════════════════════

export const login = (apiKey: string) =>
  api.post('/api/auth/login', { api_key: apiKey }).then((r) => {
    if (r.data.token) {
      localStorage.setItem('rockguard_token', r.data.token);
    }
    return r.data;
  });

// ════════════════════════════════════════════════
// 地图可视化 — 带经纬度的预警数据
// ════════════════════════════════════════════════

export interface GeoAlert {
  id: number;
  time: string;
  alert_level: string;
  count: number;
  max_confidence: number;
  class_summary: string;
  saved_frame: string;
  site_id: string;
  site_name: string;
  latitude: number;
  longitude: number;
}

export const fetchGeoAlerts = (days = 30, alertLevel = '') =>
  api
    .get<GeoAlert[]>('/api/alerts/geo', { params: { days, alert_level: alertLevel } })
    .then((r) => r.data);

// ════════════════════════════════════════════════
// ROI 多边形管理
// ════════════════════════════════════════════════

export interface RoiData {
  site_id: string;
  roi_polygon: number[][];
  frame_size: [number, number];
}

export interface RoiHeatmap {
  base64: string;
  width: number;
  height: number;
}

export const fetchRoi = (siteId = '') =>
  api.get<RoiData>('/api/roi', { params: { site_id: siteId } }).then((r) => r.data);

export const saveRoi = (siteId: string, polygon: number[][]) =>
  api.post('/api/roi', { site_id: siteId, polygon }).then((r) => r.data);

export const fetchRoiHeatmap = (siteId = '', frame = '') =>
  api
    .get<RoiHeatmap>('/api/roi/heatmap', { params: { site_id: siteId, frame } })
    .then((r) => r.data);

export default api;
