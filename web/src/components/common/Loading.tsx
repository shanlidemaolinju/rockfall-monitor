/**
 * 统一加载 / 错误提示组件（React.memo 优化）
 *
 * 用法:
 *   <Loading loading={true}>内容</Loading>
 *   <Loading error="加载失败">内容</Loading>
 */

import { memo } from 'react';
import { Spin, Button, Result, Alert } from 'antd';
import type { ReactNode } from 'react';

interface Props {
  loading?: boolean;
  error?: string;
  onRetry?: () => void;
  children?: ReactNode;
  minHeight?: number;
}

const Loading = memo(function Loading({ loading, error, onRetry, children, minHeight = 300 }: Props) {
  if (error) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight }}>
        <Result
          status="error"
          title="加载失败"
          subTitle={error}
          extra={
            onRetry && (
              <Button type="primary" onClick={onRetry}>
                重试
              </Button>
            )
          }
        />
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight }}>
        <Spin size="large" tip="加载中..." />
      </div>
    );
  }

  return <>{children}</>;
});

export default Loading;

/**
 * 页面级错误提示（非阻塞，浮层 toast 风格）
 */
export function InlineError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <Alert
      type="error"
      message="加载失败"
      description={message}
      showIcon
      closable
      style={{ marginBottom: 16 }}
      action={
        onRetry ? (
          <Button size="small" danger onClick={onRetry}>
            重试
          </Button>
        ) : undefined
      }
    />
  );
}
