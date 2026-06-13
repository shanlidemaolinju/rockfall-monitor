/**
 * 点位管理 — 点位卡片 + 激活切换
 */

import { useEffect, useState } from 'react';
import { Card, Row, Col, Tag, Button, Space, Typography, Spin, message } from 'antd';
import { EnvironmentOutlined, CheckCircleFilled } from '@ant-design/icons';
import type { MonitoringSite } from '../services/api';
import { fetchSites, switchSite } from '../services/api';
import { useAppStore } from '../stores/useAppStore';

const { Text, Title } = Typography;

const RISK_LABELS: Record<string, { color: string; label: string }> = {
  high: { color: '#f85149', label: '高风险' },
  medium: { color: '#f0883e', label: '中风险' },
  low: { color: '#3fb950', label: '低风险' },
};

export default function SiteManagement() {
  const [sites, setSites] = useState<MonitoringSite[]>([]);
  const [activeId, setActiveId] = useState('');
  const [loading, setLoading] = useState(true);
  const setGlobalSites = useAppStore((s) => s.setSites);
  const setGlobalActiveId = useAppStore((s) => s.setActiveSiteId);

  useEffect(() => { loadSites(); }, []);

  async function loadSites() {
    try {
      const data = await fetchSites();
      setSites(data.sites || []);
      setActiveId(data.active_site_id || '');
      setGlobalSites(data.sites || []);
      setGlobalActiveId(data.active_site_id || '');
    } catch {
      message.error('加载点位失败');
    } finally {
      setLoading(false);
    }
  }

  async function handleSwitch(siteId: string) {
    try {
      await switchSite(siteId);
      setActiveId(siteId);
      setGlobalActiveId(siteId);
      message.success('已切换监测点位');
    } catch {
      message.error('切换失败');
    }
  }

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;

  return (
    <div>
      <Title level={4} style={{ color: '#c9d1d9', marginBottom: 16 }}>
        <EnvironmentOutlined /> 监测点位管理
      </Title>
      <Row gutter={[16, 16]}>
        {sites.map((site) => {
          const risk = RISK_LABELS[site.risk_level] || RISK_LABELS.low;
          const isActive = site.site_id === activeId;
          return (
            <Col xs={24} sm={12} lg={8} key={site.site_id}>
              <Card
                bordered={false}
                style={{
                  background: '#161b22',
                  border: isActive ? '2px solid #58a6ff' : '1px solid #30363d',
                  opacity: isActive ? 1 : 0.8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <Space>
                      <Text strong style={{ color: '#c9d1d9', fontSize: 15 }}>
                        📍 {site.name || site.site_id}
                      </Text>
                      {isActive && <CheckCircleFilled style={{ color: '#58a6ff' }} />}
                    </Space>
                    <div style={{ marginTop: 4 }}>
                      <Text style={{ color: '#8b949e', fontSize: 12 }}>{site.location || site.region || '--'}</Text>
                    </div>
                    {site.highway && (
                      <div style={{ marginTop: 4 }}>
                        <Text style={{ color: '#8b949e', fontSize: 12 }}>
                          🛣 {site.highway} {site.stake_mark || ''}
                        </Text>
                      </div>
                    )}
                    {site.latitude && site.longitude && (
                      <div style={{ marginTop: 4 }}>
                        <Text style={{ color: '#8b949e', fontSize: 11 }} copyable>
                          {site.latitude}, {site.longitude}
                        </Text>
                      </div>
                    )}
                  </div>
                  <Tag color={risk.color} style={{ background: `${risk.color}22`, border: `1px solid ${risk.color}44` }}>
                    {risk.label}
                  </Tag>
                </div>
                <div style={{ marginTop: 12 }}>
                  <Button
                    type={isActive ? 'primary' : 'default'}
                    size="small"
                    block
                    onClick={() => handleSwitch(site.site_id)}
                    disabled={isActive}
                  >
                    {isActive ? '当前激活点位' : '切换到此点位'}
                  </Button>
                </div>
              </Card>
            </Col>
          );
        })}
        {sites.length === 0 && (
          <Col span={24}>
            <div style={{ textAlign: 'center', padding: 60, color: '#8b949e' }}>
              暂无配置点位，请在 .env 中配置或通过 API 创建
            </div>
          </Col>
        )}
      </Row>
    </div>
  );
}
