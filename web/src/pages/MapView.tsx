/**
 * 地图监控 — Leaflet 地图 + 监测点位 + 预警热力图 + 时间轴
 */

import { useEffect, useState, useMemo, useCallback, memo } from 'react';
import { Card, Select, Space, Typography, Spin, Tag } from 'antd';
import { MapContainer, TileLayer, CircleMarker, Marker, Popup, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';

import { fetchSites, fetchGeoAlerts, type MonitoringSite, type GeoAlert } from '../services/api';
import AlertLevelTag from '../components/common/AlertLevelTag';
import { fixLeafletIcons } from '../utils/leaflet-fix';

const { Text } = Typography;

// ─── Leaflet 默认图标修复 ───
fixLeafletIcons();

// ─── 站点自定义图标 ───
const siteIcon = L.divIcon({
  className: 'custom-site-icon',
  html: '<div style="background:#58a6ff;width:14px;height:14px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 6px #58a6ff;"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// ─── 暗色地图 tile (CartoDB dark) ───
const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>';

// ─── 广西中心坐标 ───
const DEFAULT_CENTER: [number, number] = [23.0, 108.5];
const DEFAULT_ZOOM = 7;

const DAY_OPTIONS = [
  { value: 7, label: '近 7 天' },
  { value: 14, label: '近 14 天' },
  { value: 30, label: '近 30 天' },
  { value: 90, label: '近 90 天' },
  { value: 365, label: '近 1 年' },
];

// ══════════════════════════════════════════════════════════════
// 子组件: 地图自适应视图
// ══════════════════════════════════════════════════════════════

const FitBounds = memo(function FitBounds({ points }: { points: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length > 0) {
      const bounds = L.latLngBounds(points);
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 });
      }
    }
  }, [points, map]);
  return null;
});

// ══════════════════════════════════════════════════════════════
// 颜色映射
// ══════════════════════════════════════════════════════════════

const LEVEL_COLORS: Record<string, string> = {
  red: '#f85149',
  orange: '#f0883e',
  yellow: '#d29922',
  blue: '#58a6ff',
};

// ══════════════════════════════════════════════════════════════
// 主组件
// ══════════════════════════════════════════════════════════════

export default function MapView() {
  const [sites, setSites] = useState<MonitoringSite[]>([]);
  const [alerts, setAlerts] = useState<GeoAlert[]>([]);
  const [days, setDays] = useState(30);
  const [alertLevel, setAlertLevel] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedSiteId, setSelectedSiteId] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, a] = await Promise.all([
        fetchSites(),
        fetchGeoAlerts(days, alertLevel),
      ]);
      setSites(s.sites || []);
      setAlerts(a);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [days, alertLevel]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const filteredAlerts = useMemo(() => {
    if (!selectedSiteId) return alerts;
    return alerts.filter((a) => a.site_id === selectedSiteId);
  }, [alerts, selectedSiteId]);

  // ── 去重后的坐标点 (用于 fitBounds，避免大量重复坐标) ──
  const geoPoints: [number, number][] = useMemo(() => {
    const seen = new Set<string>();
    const pts: [number, number][] = [];
    const add = (lat: number, lng: number) => {
      const key = `${lat.toFixed(4)},${lng.toFixed(4)}`;
      if (!seen.has(key)) {
        seen.add(key);
        pts.push([lat, lng]);
      }
    };
    sites.forEach((s) => {
      if (s.latitude && s.longitude) add(s.latitude, s.longitude);
    });
    alerts.forEach((a) => {
      if (a.latitude && a.longitude) add(a.latitude, a.longitude);
    });
    return pts.length > 0 ? pts : [DEFAULT_CENTER];
  }, [sites, alerts]);

  // ── 按站点聚合预警 ──
  const siteAlertMap = useMemo(() => {
    const map = new Map<string, { count: number; alerts: GeoAlert[]; maxLevel: string }>();
    alerts.forEach((a) => {
      const key = a.site_id || a.site_name || 'unknown';
      const entry = map.get(key) || { count: 0, alerts: [], maxLevel: 'blue' };
      entry.count += 1;
      entry.alerts.push(a);
      // 跟踪最高预警等级
      const order = ['blue', 'yellow', 'orange', 'red'];
      if (order.indexOf(a.alert_level) > order.indexOf(entry.maxLevel)) {
        entry.maxLevel = a.alert_level;
      }
      map.set(key, entry);
    });
    return map;
  }, [alerts]);

  // ── 站点匹配的坐标 ──
  const siteCoordMap = useMemo(() => {
    const map = new Map<string, { lat: number; lng: number; name: string }>();
    sites.forEach((s) => {
      if (s.latitude && s.longitude) {
        map.set(s.site_id, { lat: s.latitude, lng: s.longitude, name: s.name });
      }
    });
    return map;
  }, [sites]);

  if (loading && sites.length === 0) {
    return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;
  }

  const hasSites = sites.some((s) => s.latitude && s.longitude);
  const hasAlerts = alerts.length > 0;

  return (
    <div>
      {/* ── 控制栏 ── */}
      <Card
        bordered={false}
        style={{ background: '#161b22', border: '1px solid #30363d', marginBottom: 16 }}
        bodyStyle={{ padding: '12px 16px' }}
      >
        <Space wrap size="middle">
          <Space>
            <Text style={{ color: '#8b949e', fontSize: 13 }}>⏱ 时间范围:</Text>
            <Select
              value={days}
              onChange={setDays}
              style={{ width: 120 }}
              options={DAY_OPTIONS}
            />
          </Space>
          <Space>
            <Text style={{ color: '#8b949e', fontSize: 13 }}>🎚 预警等级:</Text>
            <Select
              value={alertLevel}
              onChange={setAlertLevel}
              style={{ width: 120 }}
              options={[
                { value: '', label: '全部等级' },
                { value: 'red', label: '🔴 Ⅰ级红色' },
                { value: 'orange', label: '🟠 Ⅱ级橙色' },
                { value: 'yellow', label: '🟡 Ⅲ级黄色' },
                { value: 'blue', label: '🔵 Ⅳ级蓝色' },
              ]}
            />
          </Space>
          <Tag color="blue" style={{ marginLeft: 8 }}>
            站点 {sites.length} 个
          </Tag>
          <Tag color="orange">
            预警 {alerts.length} 条 ({days}天)
          </Tag>
          {!hasSites && !hasAlerts && (
            <Tag color="default">无数据 — 请确认监测点位已配置经纬度</Tag>
          )}
        </Space>
      </Card>

      {/* ── 地图 ── */}
      <Card
        bordered={false}
        style={{ background: '#161b22', border: '1px solid #30363d' }}
        bodyStyle={{ padding: 0, borderRadius: 8, overflow: 'hidden' }}
      >
        <div style={{ height: '65vh', minHeight: 480 }}>
          <MapContainer
            center={DEFAULT_CENTER}
            zoom={DEFAULT_ZOOM}
            style={{ height: '100%', width: '100%', background: '#0d1117' }}
            scrollWheelZoom={true}
          >
            <TileLayer url={TILE_URL} attribution={TILE_ATTR} />

            {/* 自适应视图 */}
            <FitBounds points={geoPoints} />

            {/* ── 站点标记 (蓝色圆点) ── */}
            {sites
              .filter((s) => s.latitude && s.longitude)
              .map((s) => (
                <Marker
                  key={`site-${s.site_id}`}
                  position={[s.latitude, s.longitude]}
                  icon={siteIcon}
                  eventHandlers={{
                    click: () => setSelectedSiteId(
                      selectedSiteId === s.site_id ? null : s.site_id
                    ),
                  }}
                >
                  <Tooltip direction="top" offset={[0, -8]}>
                    <div style={{ fontSize: 12 }}>
                      <strong>{s.name || s.site_id}</strong>
                      <br />
                      {s.location || s.region || ''}
                      {s.highway ? ` · ${s.highway}` : ''}
                      <br />
                      风险: {s.risk_level === 'high' ? '🔴 高' : s.risk_level === 'medium' ? '🟠 中' : '🟢 低'}
                    </div>
                  </Tooltip>
                  <Popup maxWidth={280}>
                    <div style={{ fontSize: 13 }}>
                      <strong style={{ color: '#58a6ff' }}>📍 {s.name || s.site_id}</strong>
                      <div style={{ marginTop: 4, color: '#666' }}>
                        {s.location && <div>位置: {s.location}</div>}
                        {s.highway && <div>公路: {s.highway} {s.stake_mark || ''}</div>}
                        {s.latitude && s.longitude && (
                          <div>
                            坐标: {s.latitude.toFixed(4)}, {s.longitude.toFixed(4)}
                          </div>
                        )}
                        <div>
                          风险等级:{' '}
                          <Tag
                            color={
                              s.risk_level === 'high'
                                ? 'red'
                                : s.risk_level === 'medium'
                                  ? 'orange'
                                  : 'green'
                            }
                          >
                            {s.risk_level === 'high' ? '高' : s.risk_level === 'medium' ? '中' : '低'}
                          </Tag>
                        </div>
                      </div>
                    </div>
                  </Popup>
                </Marker>
              ))}

            {/* ── 预警热力圆圈 ── */}
            {Array.from(siteAlertMap.entries()).map(([siteId, entry]) => {
              const coord = siteCoordMap.get(siteId);
              if (!coord) return null;

              const radius = Math.min(8 + entry.count * 3, 40);
              const color = LEVEL_COLORS[entry.maxLevel] || '#58a6ff';
              const opacity = Math.min(0.3 + entry.count * 0.05, 0.8);

              return (
                <CircleMarker
                  key={`alert-${siteId}`}
                  center={[coord.lat, coord.lng]}
                  radius={radius}
                  pathOptions={{
                    color,
                    fillColor: color,
                    fillOpacity: opacity,
                    weight: 2,
                  }}
                >
                  <Popup maxWidth={320}>
                    <div style={{ fontSize: 13, maxHeight: 240, overflowY: 'auto' }}>
                      <strong style={{ color }}>🔥 {coord.name || siteId}</strong>
                      <div style={{ marginTop: 4, color: '#666' }}>
                        预警总数: <strong>{entry.count}</strong> 条
                      </div>
                      <div style={{ marginTop: 4, color: '#666' }}>
                        最高等级: <AlertLevelTag level={entry.maxLevel} />
                      </div>
                      <hr style={{ margin: '6px 0', borderColor: '#eee' }} />
                      {entry.alerts.slice(0, 10).map((a) => (
                        <div
                          key={a.id}
                          style={{
                            marginBottom: 6,
                            padding: '6px 8px',
                            background: '#f5f5f5',
                            borderRadius: 4,
                            fontSize: 11,
                          }}
                        >
                          <div>
                            <AlertLevelTag level={a.alert_level} />
                            <span style={{ marginLeft: 4, color: '#999' }}>{a.time}</span>
                          </div>
                          <div>
                            目标: {a.count} 个 | 置信度:{' '}
                            {a.max_confidence ? (a.max_confidence * 100).toFixed(0) + '%' : '--'}
                          </div>
                          {a.class_summary && <div>类别: {a.class_summary}</div>}
                          {a.saved_frame && (
                            <a
                              href={`/api/alerts/${a.id}/image`}
                              target="_blank"
                              style={{ fontSize: 11 }}
                            >
                              🖼️ 查看截图
                            </a>
                          )}
                        </div>
                      ))}
                      {entry.alerts.length > 10 && (
                        <div style={{ color: '#999', fontSize: 11, textAlign: 'center' }}>
                          ... 还有 {entry.alerts.length - 10} 条
                        </div>
                      )}
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>

        {/* ── 图例 ── */}
        <div
          style={{
            padding: '8px 16px',
            display: 'flex',
            gap: 16,
            flexWrap: 'wrap',
            background: '#0d1117',
            borderTop: '1px solid #30363d',
          }}
        >
          <Space size={4}><span style={{ color: '#58a6ff', fontSize: 14 }}>●</span> 监测站点</Space>
          <Space size={4}><span style={{ color: '#f85149', fontSize: 14 }}>●</span> Ⅰ级红色预警区</Space>
          <Space size={4}><span style={{ color: '#f0883e', fontSize: 14 }}>●</span> Ⅱ级橙色预警区</Space>
          <Space size={4}><span style={{ color: '#d29922', fontSize: 14 }}>●</span> Ⅲ级黄色预警区</Space>
          <Space size={4}><span style={{ color: '#58a6ff', fontSize: 14 }}>●</span> Ⅳ级蓝色预警区</Space>
          <Text style={{ color: '#8b949e', fontSize: 11 }}>
            圆圈半径 ∝ 预警数量 | 颜色 = 最高预警等级
          </Text>
        </div>
      </Card>

      {/* ── 预警列表 (底部) ── */}
      {alerts.length > 0 && (
        <Card
          title={
            <Space>
              <span>📋 预警明细</span>
              {selectedSiteId ? (
                <Tag color="blue" closable onClose={() => setSelectedSiteId(null)}>
                  已过滤: {filteredAlerts.length} 条
                </Tag>
              ) : (
                <span style={{ color: '#8b949e', fontSize: 13 }}>({alerts.length} 条)</span>
              )}
              {selectedSiteId && (
                <Text style={{ color: '#8b949e', fontSize: 11 }}>
                  点击地图站点可筛选 | 再次点击取消
                </Text>
              )}
            </Space>
          }
          bordered={false}
          style={{ background: '#161b22', border: '1px solid #30363d', marginTop: 16 }}
        >
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#8b949e', textAlign: 'left' }}>
                  <th style={thStyle}>时间</th>
                  <th style={thStyle}>等级</th>
                  <th style={thStyle}>站点</th>
                  <th style={thStyle}>目标数</th>
                  <th style={thStyle}>置信度</th>
                  <th style={thStyle}>类别</th>
                  <th style={thStyle}>坐标</th>
                </tr>
              </thead>
              <tbody>
                {filteredAlerts.slice(0, 100).map((a) => (
                  <tr key={a.id} style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={tdStyle}>{a.time}</td>
                    <td style={tdStyle}><AlertLevelTag level={a.alert_level} /></td>
                    <td style={tdStyle}>{a.site_name || '--'}</td>
                    <td style={tdStyle}>{a.count}</td>
                    <td style={tdStyle}>
                      {a.max_confidence ? (a.max_confidence * 100).toFixed(0) + '%' : '--'}
                    </td>
                    <td style={tdStyle}>{a.class_summary || '--'}</td>
                    <td style={tdStyle}>
                      {a.latitude ? `${a.latitude.toFixed(3)}, ${a.longitude.toFixed(3)}` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

const thStyle: React.CSSProperties = { padding: '6px 8px', borderBottom: '2px solid #30363d' };
const tdStyle: React.CSSProperties = { padding: '4px 8px', color: '#c9d1d9' };
