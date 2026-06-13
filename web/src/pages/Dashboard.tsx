/**
 * 监测大屏 — 主页
 * 四色预警统计卡片 + 实时视频流 + 最近预警列表
 */

import { useEffect, useState } from 'react';
import { Row, Col, Card, Statistic, Table, Typography, Spin, Alert } from 'antd';
import {
  AlertOutlined,
  FireOutlined,
  WarningOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { fetchStats, fetchAlerts, type DashboardStats, type AlertItem } from '../services/api';
import { useAppStore } from '../stores/useAppStore';
import AlertLevelTag from '../components/common/AlertLevelTag';

const { Text } = Typography;

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const setGlobalStats = useAppStore((s) => s.setStats);

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 10000); // 10s 刷新
    return () => clearInterval(timer);
  }, []);

  async function loadData() {
    try {
      const [s, a] = await Promise.all([fetchStats(), fetchAlerts(10)]);
      setStats(s);
      setGlobalStats(s);
      setAlerts(a);
      setError('');
    } catch (e) {
      setError('加载失败，请检查后端服务是否运行');
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;
  if (error) return <Alert message={error} type="error" showIcon />;

  const statCards = [
    { title: 'Ⅰ级红色预警', value: stats?.today_red || 0, icon: <FireOutlined />, color: '#f85149' },
    { title: 'Ⅱ级橙色预警', value: stats?.today_orange || 0, icon: <WarningOutlined />, color: '#f0883e' },
    { title: 'Ⅲ级黄色预警', value: stats?.today_yellow || 0, icon: <AlertOutlined />, color: '#d29922' },
    { title: 'Ⅳ级蓝色预警', value: stats?.today_blue || 0, icon: <InfoCircleOutlined />, color: '#58a6ff' },
  ];

  const alertColumns = [
    { title: '时间', dataIndex: 'time', key: 'time', width: 180 },
    {
      title: '等级', dataIndex: 'alert_level', key: 'level', width: 100,
      render: (l: string) => <AlertLevelTag level={l} />,
    },
    { title: '目标数', dataIndex: 'count', key: 'count', width: 80 },
    {
      title: '置信度', dataIndex: 'max_confidence', key: 'conf', width: 100,
      render: (v: number) => v ? `${(v * 100).toFixed(1)}%` : '--',
    },
    { title: '类别', dataIndex: 'class_summary', key: 'cls', ellipsis: true },
    {
      title: '截图', dataIndex: 'saved_frame', key: 'img', width: 80,
      render: (f: string, r: AlertItem) =>
        f ? <a href={`/api/alerts/${r.id}/image`} target="_blank">🖼️ 查看</a> : '--',
    },
  ];

  return (
    <div>
      {/* ─ 统计卡片 ─ */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {statCards.map((c) => (
          <Col xs={12} sm={12} md={6} key={c.title}>
            <Card bordered={false} style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <Statistic
                title={<Text style={{ color: '#8b949e', fontSize: 13 }}>{c.title}</Text>}
                value={c.value}
                prefix={<span style={{ color: c.color }}>{c.icon}</span>}
                valueStyle={{ color: c.color, fontSize: 28, fontWeight: 700 }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      {/* ─ 实时视频流 + 最近预警 ─ */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card
            title="📹 实时监测画面"
            bordered={false}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
            bodyStyle={{ padding: 0, background: '#000', borderRadius: '0 0 8px 8px', minHeight: 360 }}
          >
            <img
              src="/api/stream.mjpeg"
              alt="MJPEG Stream"
              style={{ width: '100%', display: 'block' }}
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
            <div id="stream-offline" style={{ display: 'none', textAlign: 'center', padding: 80, color: '#8b949e' }}>
              ⏳ 视频流暂不可用 — 请在桌面端启动检测
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title="📋 最近预警"
            bordered={false}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
          >
            <Table
              dataSource={alerts}
              columns={alertColumns}
              rowKey="id"
              size="small"
              pagination={false}
              locale={{ emptyText: '暂无预警记录 🎉' }}
              style={{ background: 'transparent' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
