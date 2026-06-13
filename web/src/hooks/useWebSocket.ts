/**
 * WebSocket Hook — 自动重连 + 进度推送
 *
 * 用法:
 *   const { progress, status, connect } = useTaskWebSocket(taskId);
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import type { TaskStatus } from '../services/api';
import { captureException } from '../services/sentry';

interface WsState {
  status: string;
  progress: number;
  currentFrame: number;
  totalFrames: number;
  result?: Record<string, unknown>;
  error?: string;
}

export function useTaskWebSocket(taskId: string | null) {
  const [state, setState] = useState<WsState>({
    status: 'idle',
    progress: 0,
    currentFrame: 0,
    totalFrames: 0,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const reconnectCount = useRef<number>(0);
  // 用 ref 保存最新 state.status，避免 onclose 闭包过期
  const statusRef = useRef(state.status);
  statusRef.current = state.status;

  const connect = useCallback(() => {
    if (!taskId) return;

    // 关闭已有连接
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // 重连次数限制：最多 5 次
    if (reconnectCount.current >= 5) {
      setState((s) => ({ ...s, status: 'ws_error', error: '重连次数超限，请手动重试' }));
      return;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/tasks/${taskId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectCount.current = 0;
        setState((s) => ({ ...s, status: 'connecting' }));
      };

      ws.onmessage = (event) => {
        const data: TaskStatus = JSON.parse(event.data);

        if (data.status === 'not_found') {
          setState((s) => ({ ...s, status: 'not_found', error: '任务不存在' }));
          return;
        }

        setState({
          status: data.status,
          progress: data.progress,
          currentFrame: data.current_frame,
          totalFrames: data.total_frames,
          result: data.result,
          error: data.error,
        });

        if (data.status === 'completed' || data.status === 'failed') {
          ws.close();
        }
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, status: 'ws_error' }));
        // WebSocket 错误以 warning 级别上报，避免与 API 层重复
        captureException(new Error(`WebSocket 连接失败: ${wsUrl}`), {
          level: 'warning',
          tags: { source: 'websocket' },
          extra: { taskId, url: wsUrl },
        });
      };

      ws.onclose = (_e) => {
        wsRef.current = null;
        // 使用 ref 获取最新状态，避免闭包过期
        const currentStatus = statusRef.current;
        if (currentStatus === 'processing' || currentStatus === 'connecting') {
          reconnectCount.current += 1;
          reconnectTimer.current = setTimeout(() => connect(), 3000);
        }
      };
    } catch {
      setState((s) => ({ ...s, status: 'ws_error', error: 'WebSocket 连接失败' }));
    }
  }, [taskId]);

  // 清理
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, []);

  return { ...state, connect };
}
