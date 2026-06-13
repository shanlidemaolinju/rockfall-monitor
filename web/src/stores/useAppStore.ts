/**
 * Zustand 全局状态 — 轻量、无 boilerplate
 */

import { create } from 'zustand';
import type { DashboardStats, MonitoringSite } from '../services/api';

interface AppState {
  // 侧边栏
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // 实时统计
  stats: DashboardStats | null;
  setStats: (s: DashboardStats) => void;

  // 当前激活点位
  activeSiteId: string;
  setActiveSiteId: (id: string) => void;

  // 点位列表缓存
  sites: MonitoringSite[];
  setSites: (sites: MonitoringSite[]) => void;

  // 报警声音
  soundEnabled: boolean;
  toggleSound: () => void;

  // 认证
  isAuthenticated: boolean;
  token: string | null;
  setAuth: (token: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  stats: null,
  setStats: (stats) => set({ stats }),

  activeSiteId: 'default',
  setActiveSiteId: (id) => set({ activeSiteId: id }),

  sites: [],
  setSites: (sites) => set({ sites }),

  soundEnabled: true,
  toggleSound: () => set((s) => ({ soundEnabled: !s.soundEnabled })),

  isAuthenticated: false,
  token: localStorage.getItem('rockguard_token'),
  setAuth: (token) => {
    if (token) {
      localStorage.setItem('rockguard_token', token);
    } else {
      localStorage.removeItem('rockguard_token');
    }
    set({ token, isAuthenticated: !!token });
  },
}));
