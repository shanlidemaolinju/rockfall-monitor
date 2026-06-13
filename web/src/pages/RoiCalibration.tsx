/**
 * ROI 标定 — React-Konva 交互画布
 * 鼠标点击绘制多边形顶点，实时显示热力图 overlay。
 * 支持撤销、重置、加载已有 ROI、保存到后端。
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { Card, Button, Space, Switch, Typography, message, Spin, Row, Col, Tag, Select } from 'antd';
import {
  UndoOutlined,
  SaveOutlined,
  DownloadOutlined,
  DeleteOutlined,
  AimOutlined,
} from '@ant-design/icons';
import { Stage, Layer, Line, Circle, Image as KonvaImage } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { fetchRoi, saveRoi, fetchRoiHeatmap, fetchSites, type RoiData, type RoiHeatmap } from '../services/api';
import { useAppStore } from '../stores/useAppStore';

const { Text } = Typography;

// 画布尺寸常量
const CANVAS_W = 960;
const CANVAS_H = 540;
const VERTEX_RADIUS = 6;
const STROKE_COLOR = '#58a6ff';
const FILL_COLOR = 'rgba(88, 166, 255, 0.12)';

export default function RoiCalibration() {
  const [vertices, setVertices] = useState<number[][]>([]);
  const [sites, setSites] = useState<{ id: string; name: string }[]>([]);
  const [selectedSiteId, setSelectedSiteId] = useState('');
  const [heatmap, setHeatmap] = useState<RoiHeatmap | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);
  const [hoveredVertex, setHoveredVertex] = useState<number | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const heatmapImgRef = useRef<HTMLImageElement | null>(null);
  const activeSiteId = useAppStore((s) => s.activeSiteId);

  // ── 初始化: 加载站点列表 ──
  useEffect(() => {
    loadSites();
  }, []);

  useEffect(() => {
    if (selectedSiteId) {
      loadRoi(selectedSiteId);
      loadHeatmap(selectedSiteId);
    }
  }, [selectedSiteId]);

  useEffect(() => {
    if (activeSiteId && !selectedSiteId) {
      setSelectedSiteId(activeSiteId);
    }
  }, [activeSiteId]);

  async function loadSites() {
    try {
      const data = await fetchSites();
      const list = (data.sites || []).map((s) => ({ id: s.site_id, name: s.name || s.site_id }));
      setSites(list);
      if (list.length > 0 && !selectedSiteId) {
        setSelectedSiteId(data.active_site_id || list[0].id);
      }
    } catch {
      message.error('加载站点列表失败');
    } finally {
      setLoading(false);
    }
  }

  async function loadRoi(siteId: string) {
    try {
      const data: RoiData = await fetchRoi(siteId);
      setVertices(data.roi_polygon || []);
    } catch {
      setVertices([]);
    }
  }

  async function loadHeatmap(siteId: string) {
    try {
      const data: RoiHeatmap = await fetchRoiHeatmap(siteId);
      setHeatmap(data);
    } catch {
      setHeatmap(null);
    }
  }

  // ── 处理热力图图片 ──
  useEffect(() => {
    if (heatmap?.base64) {
      const img = new window.Image();
      img.onload = () => { heatmapImgRef.current = img; };
      img.src = heatmap.base64;
    } else {
      heatmapImgRef.current = null;
    }
  }, [heatmap]);

  // ── 处理默认背景 ──
  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;
    const ctx = canvas.getContext('2d')!;
    // 渐变背景: 模拟山路场景
    const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grad.addColorStop(0, '#1a2a1a');
    grad.addColorStop(0.3, '#2a3a2a');
    grad.addColorStop(0.5, '#3a4a3a');
    grad.addColorStop(0.7, '#4a4a3a');
    grad.addColorStop(1, '#3a3a3a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    // 模拟边坡轮廓
    ctx.strokeStyle = '#4a5a3a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, CANVAS_H * 0.5);
    ctx.quadraticCurveTo(CANVAS_W * 0.4, CANVAS_H * 0.2, CANVAS_W, CANVAS_H * 0.45);
    ctx.stroke();
    // 道路区域
    ctx.fillStyle = '#3a3a3a';
    ctx.fillRect(0, CANVAS_H * 0.65, CANVAS_W, CANVAS_H * 0.35);
    ctx.strokeStyle = '#555';
    ctx.setLineDash([20, 10]);
    ctx.beginPath();
    ctx.moveTo(0, CANVAS_H * 0.72);
    ctx.lineTo(CANVAS_W, CANVAS_H * 0.72);
    ctx.stroke();
    ctx.setLineDash([]);
    // 标注文字
    ctx.fillStyle = '#666';
    ctx.font = '13px sans-serif';
    ctx.fillText('边坡区域 (建议 ROI 范围)', 20, CANVAS_H * 0.35);
    ctx.fillText('公路区域 (排除)', 20, CANVAS_H * 0.85);

    const img = new window.Image();
    img.onload = () => setBgImage(img);
    img.src = canvas.toDataURL();
  }, []);

  // ── Konva 事件处理 ──
  const handleCanvasClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;

    // 检查是否点击了已有顶点 (允许删除)
    const hitDist = 12;
    const nearIdx = vertices.findIndex(
      (v) => Math.hypot(v[0] - pos.x, v[1] - pos.y) < hitDist,
    );
    if (nearIdx >= 0) {
      // Ctrl+click 删除顶点
      if (e.evt.ctrlKey || e.evt.metaKey) {
        setVertices((prev) => prev.filter((_, i) => i !== nearIdx));
      }
      return;
    }

    // 添加新顶点
    setVertices((prev) => [...prev, [Math.round(pos.x), Math.round(pos.y)]]);
  }, [vertices]);

  const handleMouseMove = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;
    const pos = stage.getPointerPosition();
    if (pos) {
      setMousePos([pos.x, pos.y]);
      // 检测 hover 顶点
      const nearIdx = vertices.findIndex(
        (v) => Math.hypot(v[0] - pos.x, v[1] - pos.y) < 10,
      );
      setHoveredVertex(nearIdx >= 0 ? nearIdx : null);
    }
  }, [vertices]);

  const handleDragStart = useCallback((index: number) => {
    setDragIndex(index);
  }, []);

  const handleDragMove = useCallback((index: number, e: KonvaEventObject<DragEvent>) => {
    const node = e.target;
    setVertices((prev) => {
      const next = [...prev];
      next[index] = [Math.round(node.x()), Math.round(node.y())];
      return next;
    });
  }, []);

  const handleDragEnd = useCallback(() => {
    setDragIndex(null);
  }, []);

  // ── 操作按钮 ──
  function handleUndo() {
    setVertices((prev) => prev.slice(0, -1));
  }

  function handleReset() {
    setVertices([]);
  }

  async function handleSave() {
    if (vertices.length < 3) {
      message.warning('至少需要 3 个顶点才能保存 ROI 多边形');
      return;
    }
    if (!selectedSiteId) {
      message.warning('请先选择监测点位');
      return;
    }
    setSaving(true);
    try {
      const result = await saveRoi(selectedSiteId, vertices);
      message.success(`ROI 已保存 (${result.vertices} 个顶点)`);
    } catch (e: any) {
      message.error('保存失败: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setSaving(false);
    }
  }

  // ── 生成多边形闭合路径 ──
  const pointsFlat = vertices.flat();
  const isClosed = vertices.length >= 3;

  if (loading) {
    return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;
  }

  return (
    <div>
      <Card
        title={<><AimOutlined /> ROI 多边形标定</>}
        bordered={false}
        style={{ background: '#161b22', border: '1px solid #30363d' }}
        bodyStyle={{ padding: 16 }}
        extra={
          <Space>
            <Text style={{ color: '#8b949e', fontSize: 12 }}>站点:</Text>
            <Select
              value={selectedSiteId}
              onChange={setSelectedSiteId}
              style={{ width: 200 }}
              options={sites.map((s) => ({ value: s.id, label: `${s.name} (${s.id})` }))}
            />
          </Space>
        }
      >
        <Row gutter={16}>
          {/* ── 左侧: 画布 ── */}
          <Col xs={24} lg={16}>
            <div
              style={{
                border: '2px solid #30363d',
                borderRadius: 6,
                overflow: 'hidden',
                background: '#0a0f0a',
                cursor: 'crosshair',
              }}
            >
              <Stage
                width={CANVAS_W}
                height={CANVAS_H}
                onClick={handleCanvasClick}
                onMouseMove={handleMouseMove}
                style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
              >
                {/* Layer 0: 背景 */}
                <Layer>
                  {bgImage && (
                    <KonvaImage
                      image={bgImage}
                      width={CANVAS_W}
                      height={CANVAS_H}
                      opacity={0.7}
                    />
                  )}
                </Layer>

                {/* Layer 1: 热力图 overlay */}
                {showHeatmap && heatmapImgRef.current && (
                  <Layer opacity={0.5}>
                    <KonvaImage
                      image={heatmapImgRef.current}
                      width={CANVAS_W}
                      height={CANVAS_H}
                    />
                  </Layer>
                )}

                {/* Layer 2: 多边形 + 顶点 */}
                <Layer>
                  {/* 已完成区域填充 */}
                  {isClosed && (
                    <Line
                      points={[...pointsFlat, vertices[0][0], vertices[0][1]]}
                      fill={FILL_COLOR}
                      stroke={STROKE_COLOR}
                      strokeWidth={2}
                      closed
                    />
                  )}

                  {/* 已绘制边 (未闭合时) */}
                  {vertices.length >= 2 && (
                    <Line
                      points={pointsFlat}
                      stroke={STROKE_COLOR}
                      strokeWidth={2}
                      dash={isClosed ? [] : [8, 4]}
                    />
                  )}

                  {/* 鼠标跟随预览线 */}
                  {mousePos && vertices.length >= 1 && !isClosed && (
                    <Line
                      points={[
                        vertices[vertices.length - 1][0],
                        vertices[vertices.length - 1][1],
                        mousePos[0],
                        mousePos[1],
                      ]}
                      stroke="#666"
                      strokeWidth={1}
                      dash={[4, 4]}
                    />
                  )}

                  {/* 顶点圆圈 */}
                  {vertices.map((v, i) => (
                    <Circle
                      key={i}
                      x={v[0]}
                      y={v[1]}
                      radius={hoveredVertex === i || dragIndex === i ? VERTEX_RADIUS + 2 : VERTEX_RADIUS}
                      fill={dragIndex === i ? '#f0883e' : hoveredVertex === i ? '#fff' : STROKE_COLOR}
                      stroke="#fff"
                      strokeWidth={1.5}
                      draggable
                      onDragStart={() => handleDragStart(i)}
                      onDragMove={(e) => handleDragMove(i, e)}
                      onDragEnd={handleDragEnd}
                      hitStrokeWidth={20}
                    />
                  ))}
                </Layer>
              </Stage>
            </div>

            {/* 热力图开关 */}
            <div style={{ marginTop: 8, display: 'flex', gap: 16, alignItems: 'center' }}>
              <Space>
                <Switch checked={showHeatmap} onChange={setShowHeatmap} size="small" />
                <Text style={{ color: '#8b949e', fontSize: 12 }}>
                  显示道路/边坡热力图 (辅助标定)
                </Text>
              </Space>
              <Text style={{ color: '#8b949e', fontSize: 11 }}>
                {heatmap?.base64 ? '✅ 热力图已加载' : '⚠️ 热力图不可用 (FastSAM 未就绪)'}
              </Text>
            </div>
          </Col>

          {/* ── 右侧: 操作面板 ── */}
          <Col xs={24} lg={8}>
            <div
              style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: 6,
                padding: 16,
              }}
            >
              <Text strong style={{ color: '#c9d1d9', fontSize: 14, display: 'block', marginBottom: 12 }}>
                📐 ROI 顶点编辑器
              </Text>

              <div style={{ marginBottom: 12 }}>
                <Text style={{ color: '#8b949e', fontSize: 12 }}>
                  已添加 <Tag color="blue">{vertices.length}</Tag> 个顶点
                  {vertices.length >= 3 ? ' ✅ 可保存' : ' (至少需要 3 个)'}
                </Text>
              </div>

              <Space direction="vertical" style={{ width: '100%' }} size="small">
                <Button
                  icon={<UndoOutlined />}
                  onClick={handleUndo}
                  disabled={vertices.length === 0}
                  block
                >
                  撤销上一点
                </Button>
                <Button
                  icon={<DeleteOutlined />}
                  onClick={handleReset}
                  disabled={vertices.length === 0}
                  block
                >
                  清空全部
                </Button>
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => loadRoi(selectedSiteId)}
                  block
                >
                  加载已有 ROI
                </Button>
                <Button
                  type="primary"
                  icon={<SaveOutlined />}
                  onClick={handleSave}
                  loading={saving}
                  disabled={vertices.length < 3}
                  block
                  style={{ marginTop: 4 }}
                >
                  保存到后端
                </Button>
              </Space>

              <div style={{ marginTop: 16 }}>
                <Text style={{ color: '#8b949e', fontSize: 11 }}>
                  💡 操作提示:
                </Text>
                <ul style={{ color: '#8b949e', fontSize: 11, paddingLeft: 16, marginTop: 4 }}>
                  <li>点击画布添加顶点</li>
                  <li>拖拽顶点调整位置</li>
                  <li>Ctrl+点击 删除顶点</li>
                  <li>至少 3 个顶点形成闭合多边形</li>
                  <li>保存后自动重建 MOG2 背景模型</li>
                </ul>
              </div>

              {/* 坐标列表 */}
              {vertices.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <Text style={{ color: '#8b949e', fontSize: 11 }}>顶点坐标:</Text>
                  <div
                    style={{
                      maxHeight: 180,
                      overflowY: 'auto',
                      background: '#0a0f0a',
                      border: '1px solid #30363d',
                      borderRadius: 4,
                      padding: 8,
                      marginTop: 4,
                    }}
                  >
                    {vertices.map((v, i) => (
                      <div
                        key={i}
                        style={{
                          fontFamily: 'monospace',
                          fontSize: 11,
                          color: '#c9d1d9',
                          padding: '2px 0',
                        }}
                      >
                        [{i}] ({v[0]}, {v[1]})
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
}
