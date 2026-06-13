/**
 * 通用组件 — 预警等级标签（React.memo 优化纯展示场景）
 */

import { memo } from 'react';
import { Tag } from 'antd';

const LEVEL_CONFIG: Record<string, { color: string; label: string }> = {
  red: { color: '#f85149', label: 'Ⅰ级红色' },
  orange: { color: '#f0883e', label: 'Ⅱ级橙色' },
  yellow: { color: '#d29922', label: 'Ⅲ级黄色' },
  blue: { color: '#58a6ff', label: 'Ⅳ级蓝色' },
  green: { color: '#3fb950', label: '安全' },
};

interface Props {
  level: string;
  showLabel?: boolean;
}

const AlertLevelTag = memo(function AlertLevelTag({ level, showLabel = true }: Props) {
  const cfg = LEVEL_CONFIG[level] || LEVEL_CONFIG.green;
  return (
    <Tag
      color={cfg.color}
      style={{
        background: `${cfg.color}22`,
        border: `1px solid ${cfg.color}44`,
        color: cfg.color,
      }}
    >
      {showLabel ? cfg.label : '●'}
    </Tag>
  );
});

export default AlertLevelTag;
