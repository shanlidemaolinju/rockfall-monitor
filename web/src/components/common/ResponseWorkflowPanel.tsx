/**
 * 响应流程与推送配置 — 按预警等级查看详情
 *
 * Tab 切换四个预警等级，展示:
 *   - 触发条件
 *   - 处置流程
 *   - 推送渠道
 *   - 声光报警状态
 */

import { memo, useEffect, useState } from 'react';
import { Tabs, Spin, Empty, Tag, Space, List, Typography } from 'antd';
import {
  AlertOutlined,
  SoundOutlined,
  SoundFilled,
  ApiOutlined,
} from '@ant-design/icons';
import { fetchAllResponseWorkflows } from '../../services/api';
import type { ResponseWorkflow } from '../../services/api';

const { Text, Paragraph } = Typography;

// ─── 等级配置 ───
const LEVEL_TABS = [
  { key: 'red', color: '#f85149', icon: '🔴', label: 'Ⅰ 级 · 特别严重' },
  { key: 'orange', color: '#f0883e', icon: '🟠', label: 'Ⅱ 级 · 严重' },
  { key: 'yellow', color: '#d29922', icon: '🟡', label: 'Ⅲ 级 · 较重' },
  { key: 'blue', color: '#58a6ff', icon: '🔵', label: 'Ⅳ 级 · 一般' },
];

// ─── 渠道中文名 ───
const CHANNEL_LABELS: Record<string, string> = {
  pushplus: 'PushPlus 微信',
  smtp: '邮件 (SMTP)',
  wecom: '企业微信',
  dingtalk: '钉钉',
  feishu: '飞书',
  webhook: 'Webhook',
};

// ─── 单个等级的详情面板 ───
const LevelDetail = memo(function LevelDetail({
  workflow,
}: {
  workflow: ResponseWorkflow;
}) {
  return (
    <div style={{ padding: '8px 0' }}>
      {/* 触发条件 */}
      <div style={{ marginBottom: 16 }}>
        <Text
          strong
          style={{
            color: '#f85149',
            fontSize: 13,
            display: 'block',
            marginBottom: 6,
          }}
        >
          ⚡ 触发条件
        </Text>
        <div
          style={{
            fontSize: 13,
            color: '#8b949e',
            padding: '10px 14px',
            background: 'rgba(248,81,73,0.06)',
            borderRadius: 6,
            border: '1px solid rgba(248,81,73,0.15)',
            lineHeight: 1.8,
          }}
        >
          {workflow.trigger_conditions.map((cond, i) => (
            <div key={i}>• {cond}</div>
          ))}
        </div>
      </div>

      {/* 处置流程 */}
      <div style={{ marginBottom: 16 }}>
        <Text
          strong
          style={{
            color: '#3fb950',
            fontSize: 13,
            display: 'block',
            marginBottom: 6,
          }}
        >
          📋 处置流程
        </Text>
        <List
          size="small"
          dataSource={workflow.disposal_steps}
          renderItem={(item, i) => (
            <List.Item
              style={{
                borderBottom: '1px solid #21262d',
                padding: '6px 8px',
                fontSize: 13,
                color: '#c9d1d9',
              }}
            >
              <Space size={8}>
                <Text style={{ color: '#58a6ff', fontSize: 12, fontWeight: 600 }}>
                  {i + 1}.
                </Text>
                <Text style={{ color: '#c9d1d9', fontSize: 13 }}>{item}</Text>
              </Space>
            </List.Item>
          )}
          style={{
            background: '#0d1117',
            borderRadius: 6,
            border: '1px solid #30363d',
          }}
        />
      </div>

      {/* 推送渠道 + 声光报警 */}
      <div style={{ marginBottom: 16 }}>
        <Text
          strong
          style={{
            color: '#58a6ff',
            fontSize: 13,
            display: 'block',
            marginBottom: 6,
          }}
        >
          📡 推送配置
        </Text>
        <div
          style={{
            padding: '10px 14px',
            background: '#0d1117',
            borderRadius: 6,
            border: '1px solid #30363d',
          }}
        >
          {/* 推送渠道 */}
          <div style={{ marginBottom: 8 }}>
            <Space size={4} wrap>
              <ApiOutlined style={{ color: '#8b949e', fontSize: 12 }} />
              <Text style={{ color: '#8b949e', fontSize: 12 }}>推送渠道:</Text>
              {workflow.push_channels.length > 0 ? (
                workflow.push_channels.map((ch) => (
                  <Tag
                    key={ch}
                    style={{
                      background: 'rgba(88,166,255,0.1)',
                      border: '1px solid rgba(88,166,255,0.3)',
                      color: '#58a6ff',
                      fontSize: 11,
                    }}
                  >
                    {CHANNEL_LABELS[ch] || ch}
                  </Tag>
                ))
              ) : (
                <Text style={{ color: '#8b949e', fontSize: 12 }}>
                  不推送 (仅本地记录)
                </Text>
              )}
            </Space>
          </div>

          {/* 声光报警 */}
          <div>
            <Space size={4}>
              {workflow.requires_sound ? (
                <SoundFilled style={{ color: '#f85149', fontSize: 14 }} />
              ) : (
                <SoundOutlined style={{ color: '#8b949e', fontSize: 14 }} />
              )}
              <Text style={{ color: '#8b949e', fontSize: 12 }}>
                声光报警: {workflow.requires_sound ? '✅ 触发' : '❌ 不触发'}
              </Text>
            </Space>
          </div>
        </div>
      </div>
    </div>
  );
});

// ─── 主组件 ───
function ResponseWorkflowPanel() {
  const [workflows, setWorkflows] = useState<Record<
    string,
    ResponseWorkflow
  > | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeKey, setActiveKey] = useState('red');

  useEffect(() => {
    let cancelled = false;
    fetchAllResponseWorkflows()
      .then((data) => {
        if (!cancelled) setWorkflows(data);
      })
      .catch(() => { /* ignore */ })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading)
    return (
      <Spin size="small" style={{ display: 'block', margin: '24px auto' }} />
    );
  if (!workflows)
    return <Empty description="无法加载响应流程配置" />;

  return (
    <div style={{ padding: '8px 0' }}>
      <Tabs
        activeKey={activeKey}
        onChange={setActiveKey}
        size="small"
        items={LEVEL_TABS.map((tab) => ({
          key: tab.key,
          label: (
            <Space size={4}>
              <span>{tab.icon}</span>
              <span style={{ color: tab.color, fontSize: 12, fontWeight: 600 }}>
                {tab.label}
              </span>
            </Space>
          ),
          children: workflows[tab.key] ? (
            <LevelDetail workflow={workflows[tab.key]} />
          ) : (
            <Empty description="暂无配置" />
          ),
        }))}
        style={{ color: '#c9d1d9' }}
      />
    </div>
  );
}

export default ResponseWorkflowPanel;
