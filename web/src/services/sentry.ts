/**
 * Sentry 前端错误监控初始化
 * ==========================
 * 仅在 VITE_SENTRY_DSN 环境变量存在时启用。
 * 无 DSN 时所有导出函数静默跳过（零侵入）。
 *
 * 使用:
 *   import { captureException, captureMessage, isEnabled } from '@/services/sentry';
 */

const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN as string | undefined;

let _enabled = false;

/**
 * 初始化 Sentry — 在 main.tsx 中调用一次。
 * 需要异步 import 以避免将 @sentry/react 打包进无 Sentry 的构建。
 */
export async function initSentry(): Promise<boolean> {
  if (!SENTRY_DSN) {
    return false;
  }

  try {
    const Sentry = await import('@sentry/react');
    Sentry.init({
      dsn: SENTRY_DSN,
      environment: import.meta.env.VITE_SENTRY_ENVIRONMENT || 'production',
      release: 'rockfall@2.2.0',
      integrations: [
        Sentry.browserTracingIntegration(),
        Sentry.replayIntegration(),
      ],
      tracesSampleRate: 0.1,
      replaysSessionSampleRate: 0.01,
      replaysOnErrorSampleRate: 0.1,
    });
    _enabled = true;
    console.log('[sentry] 前端错误监控已启用');
    return true;
  } catch (e) {
    console.warn('[sentry] 初始化失败:', e);
    return false;
  }
}

/**
 * 安全捕获异常 — Sentry 未启用时静默跳过。
 */
export async function captureException(
  error: unknown,
  context?: {
    tags?: Record<string, string>;
    extra?: Record<string, unknown>;
    level?: 'fatal' | 'error' | 'warning' | 'info' | 'debug';
    mechanism?: { type: string; handled: boolean };
  },
): Promise<string | null> {
  if (!_enabled) return null;

  try {
    const Sentry = await import('@sentry/react');
    if (context) {
      // 使用 withScope 附加上下文后仅上报一次
      return new Promise<string | null>((resolve) => {
        Sentry.withScope((scope) => {
          if (context.tags) scope.setTags(context.tags);
          if (context.extra) scope.setExtras(context.extra);
          if (context.level) scope.setLevel(context.level);
          resolve(Sentry.captureException(error));
        });
      });
    }
    return Sentry.captureException(error);
  } catch {
    return null;
  }
}

/**
 * 安全捕获消息 — Sentry 未启用时静默跳过。
 */
export async function captureMessage(
  message: string,
  level: 'fatal' | 'error' | 'warning' | 'info' | 'debug' = 'error',
): Promise<string | null> {
  if (!_enabled) return null;

  try {
    const Sentry = await import('@sentry/react');
    return Sentry.captureMessage(message, level);
  } catch {
    return null;
  }
}

/**
 * 返回 Sentry 是否已启用。
 */
export function isEnabled(): boolean {
  return _enabled;
}
