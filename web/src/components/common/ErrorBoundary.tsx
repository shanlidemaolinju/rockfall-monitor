/**
 * 全局错误边界 — 捕获子树中未处理的渲染异常，显示友好降级 UI。
 */

import { Component, type ReactNode } from 'react';
import { Result, Button } from 'antd';
import { captureException } from '../../services/sentry';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error('[ErrorBoundary]', error.message, info.componentStack);
    captureException(error, {
      mechanism: { type: 'react', handled: false },
      extra: { componentStack: info.componentStack },
    });
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
          <Result
            status="error"
            title="页面渲染异常"
            subTitle={this.state.error?.message || '发生了未知错误，请刷新页面重试'}
            extra={[
              <Button type="primary" key="retry" onClick={this.handleRetry}>
                重试
              </Button>,
              <Button key="reload" onClick={() => window.location.reload()}>
                刷新页面
              </Button>,
            ]}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
