/**
 * 数据大屏 / 驾驶舱 — 整合地图 + ECharts 趋势 + 统计卡片 + 实时视频流
 */

import { useEffect, useState, useMemo, useCallback } from 'react';
import { Row, Col, Card, Statistic, Table, Typography, Spin, Space } from 'antd';
import ReactECharts from 'echarts-for-react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

// ─── ECharts 按需导入 (tree-shaking) ───
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

import { fetchStats, fetchAlertStatistics, fetchGeoAlerts } from '../services/api';
import type { DashboardStats, GeoAlert } from '../services/api';
import { useAppStore } from '../stores/useAppStore';
import AlertLevelTag from '../components/common/AlertLevelTag';
import { fixLeafletIcons } from '../utils/leaflet-fix';

const { Text, Title } = Typography;

// ─── Leaflet 图标修复 ───
fixLeafletIcons();

const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_ATTR = '&copy; OSM &copy; CARTO';
const MAP_CENTER: [number, number] = [23.0, 108.5];

const LEVEL_COLORS: Record<string, string> = {
  red: '#f85149', orange: '#f0883e', yellow: '#d29922', blue: '#58a6ff',
};

const ALERT_ORDER = ['blue', 'yellow', 'orange', 'red'];

// ─── 聚合站点预警 ───
function aggregateBySite(alerts: GeoAlert[]) {
  const map = new Map<string, { siteId: string; count: number; lat: number; lng: number; name: string; maxLevel: string }>();
  alerts.forEach((a) => {
    if (!a.latitude || !a.longitude) return;
    const key = a.site_id || a.site_name || 'unknown';
    const e = map.get(key) || { siteId: key, count: 0, lat: a.latitude, lng: a.longitude, name: a.site_name, maxLevel: 'blue' };
    e.count += 1;
    if (ALERT_ORDER.indexOf(a.alert_level) > ALERT_ORDER.indexOf(e.maxLevel)) {
      e.maxLevel = a.alert_level;
    }
    map.set(key, e);
  });
  return Array.from(map.values());
}

// ─── 表格列定义 (组件外避免重复创建) ───
const ALERT_COLUMNS = [
  { title: '时间', dataIndex: 'time', key: 'time', width: 150, render: (v: string) => v?.slice(5) || v },
  { title: '等级', dataIndex: 'alert_level', key: 'level', width: 90, render: (l: string) => <AlertLevelTag level={l} /> },
  { title: '站点', dataIndex: 'site_name', key: 'site', ellipsis: true },
  { title: '目标', dataIndex: 'count', key: 'count', width: 50 },
  { title: '置信度', dataIndex: 'max_confidence', key: 'conf', width: 70, render: (v: number) => v ? `${(v * 100).toFixed(0)}%` : '--' },
];

export default function Cockpit() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [trends, setTrends] = useState<{ date: string; red: number; orange: number; yellow: number; blue: number; total: number }[]>([]);
  const [alerts, setAlerts] = useState<GeoAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const setGlobalStats = useAppStore((s) => s.setStats);

  const loadAll = useCallback(async () => {
    try {
      const [s, t, a] = await Promise.all([
        fetchStats(),
        fetchAlertStatistics(7),
        fetchGeoAlerts(7),
      ]);
      setStats(s);
      setGlobalStats(s);
      setTrends(t.daily_trends || []);
      setAlerts(a);
    } catch { /* ignore */ } finally {
      setLoading(false);
    }
  }, [setGlobalStats]);

  useEffect(() => {
    loadAll();
    const timer = setInterval(loadAll, 15000); // 15s 刷新
    return () => clearInterval(timer);
  }, [loadAll]);

  const siteAgg = useMemo(() => aggregateBySite(alerts), [alerts]);

  // ─── ECharts 折线图 option ───
  const chartOption = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: {
      data: ['Ⅰ级红色', 'Ⅱ级橙色', 'Ⅲ级黄色', 'Ⅳ级蓝色'],
      textStyle: { color: '#8b949e', fontSize: 11 },
      top: 0,
    },
    grid: { left: 40, right: 16, top: 36, bottom: 24 },
    xAxis: {
      type: 'category' as const,
      data: trends.map((t) => t.date.slice(5)), // MM-DD
      axisLabel: { color: '#8b949e', fontSize: 10 },
      axisLine: { lineStyle: { color: '#30363d' } },
    },
    yAxis: {
      type: 'value' as const,
      axisLabel: { color: '#8b949e', fontSize: 10 },
      splitLine: { lineStyle: { color: '#21262d' } },
    },
    series: [
      { name: 'Ⅰ级红色', type: 'line', data: trends.map((t) => t.red), smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#f85149', width: 2 }, itemStyle: { color: '#f85149' } },
      { name: 'Ⅱ级橙色', type: 'line', data: trends.map((t) => t.orange), smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#f0883e', width: 2 }, itemStyle: { color: '#f0883e' } },
      { name: 'Ⅲ级黄色', type: 'line', data: trends.map((t) => t.yellow), smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#d29922', width: 2 }, itemStyle: { color: '#d29922' } },
      { name: 'Ⅳ级蓝色', type: 'line', data: trends.map((t) => t.blue), smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#58a6ff', width: 2 }, itemStyle: { color: '#58a6ff' } },
    ],
  }), [trends]);

  // ─── 最近预警表格 ───
  const alertTableData = useMemo(() => alerts.slice(0, 8), [alerts]);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;

  return (
    <div>
      <Title level={4} style={{ color: '#c9d1d9', marginBottom: 16 }}>
        🖥️ 数据大屏 · 驾驶舱
      </Title>

      {/* ━━━ 第一行: 统计卡片 + 视频流 ━━━ */}
      <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
        <Col xs={24} sm={12} md={6} lg={5}>
          <Card bordered={false} style={{ background: '#161b22', border: '1px solid #30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic
              title={<Text style={{ color: '#f85149', fontSize: 12 }}>🚨 Ⅰ级红色</Text>}
              value={stats?.today_red || 0}
              valueStyle={{ color: '#f85149', fontSize: 32, fontWeight: 700 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6} lg={5}>
          <Card bordered={false} style={{ background: '#161b22', border: '1px solid #30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic
              title={<Text style={{ color: '#f0883e', fontSize: 12 }}>🔶 Ⅱ级橙色</Text>}
              value={stats?.today_orange || 0}
              valueStyle={{ color: '#f0883e', fontSize: 32, fontWeight: 700 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6} lg={5}>
          <Card bordered={false} style={{ background: '#161b22', border: '1px solid #30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic
              title={<Text style={{ color: '#d29922', fontSize: 12 }}>⚠️ Ⅲ级黄色</Text>}
              value={stats?.today_yellow || 0}
              valueStyle={{ color: '#d29922', fontSize: 32, fontWeight: 700 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6} lg={5}>
          <Card bordered={false} style={{ background: '#161b22', border: '1px solid #30363d' }} bodyStyle={{ padding: '12px 16px' }}>
            <Statistic
              title={<Text style={{ color: '#58a6ff', fontSize: 12 }}>ℹ️ Ⅳ级蓝色</Text>}
              value={stats?.today_blue || 0}
              valueStyle={{ color: '#58a6ff', fontSize: 32, fontWeight: 700 }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={4}>
          <Card
            title={<Text style={{ color: '#8b949e', fontSize: 12 }}>📹 实时画面</Text>}
            bordered={false}
            bodyStyle={{ padding: 0, background: '#000', borderRadius: '0 0 6px 6px', height: 110, overflow: 'hidden' }}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
          >
            <img
              src="/api/stream.mjpeg"
              alt="MJPEG"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
            <div id="cockpit-stream-offline" style={{ display: 'none', textAlign: 'center', padding: '40px 8px', color: '#8b949e', fontSize: 11 }}>
              ⏳ 视频流暂不可用
            </div>
          </Card>
        </Col>
      </Row>

      {/* ━━━ 第二行: 迷你地图 + 趋势图 ━━━ */}
      <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
        <Col xs={24} lg={10}>
          <Card
            title={<Text style={{ color: '#c9d1d9', fontSize: 13 }}>🗺 近7天预警分布</Text>}
            bordered={false}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
            bodyStyle={{ padding: 0, borderRadius: 6, overflow: 'hidden' }}
          >
            <div style={{ height: 340 }}>
              <MapContainer
                center={MAP_CENTER}
                zoom={7}
                style={{ height: '100%', width: '100%', background: '#0d1117' }}
                scrollWheelZoom={false}
                zoomControl={false}
              >
                <TileLayer url={TILE_URL} attribution={TILE_ATTR} />
                {siteAgg.map((s) => (
                  <CircleMarker
                    key={s.siteId}
                    center={[s.lat, s.lng]}
                    radius={Math.min(6 + s.count * 3, 32)}
                    pathOptions={{ color: LEVEL_COLORS[s.maxLevel] || '#58a6ff', fillColor: LEVEL_COLORS[s.maxLevel] || '#58a6ff', fillOpacity: 0.35, weight: 2 }}
                  >
                    <Popup>
                      <div style={{ fontSize: 12 }}>
                        <strong>{s.name}</strong><br />
                        预警: {s.count} 条
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>
            <div style={{ padding: '4px 12px', background: '#0d1117', borderTop: '1px solid #30363d', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(LEVEL_COLORS).map(([k, c]) => (
                <Space key={k} size={2}><span style={{ color: c, fontSize: 12 }}>●</span><span style={{ color: '#8b949e', fontSize: 10 }}>{k === 'red' ? '红色' : k === 'orange' ? '橙色' : k === 'yellow' ? '黄色' : '蓝色'}</span></Space>
              ))}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card
            title={<Text style={{ color: '#c9d1d9', fontSize: 13 }}>📈 近7天预警趋势</Text>}
            bordered={false}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
            bodyStyle={{ padding: '8px 12px' }}
          >
            {trends.length > 0 ? (
              <ReactECharts echarts={echarts} option={chartOption} style={{ height: 300 }} />
            ) : (
              <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#8b949e', fontSize: 13 }}>
                暂无趋势数据
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <Text style={{ color: '#8b949e', fontSize: 11 }}>
                📊 近7天累计: {trends.reduce((s, t) => s + t.total, 0)} 条预警
              </Text>
              <Text style={{ color: '#8b949e', fontSize: 11 }}>
                🔄 每15秒自动刷新
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* ━━━ 第三行: 最近预警 ━━━ */}
      <Row gutter={[12, 12]}>
        <Col span={24}>
          <Card
            title={<Text style={{ color: '#c9d1d9', fontSize: 13 }}>📋 最近预警</Text>}
            bordered={false}
            style={{ background: '#161b22', border: '1px solid #30363d' }}
          >
            <Table
              dataSource={alertTableData}
              columns={ALERT_COLUMNS}
              rowKey="id"
              size="small"
              pagination={false}
              locale={{ emptyText: '暂无预警记录 🎉' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
