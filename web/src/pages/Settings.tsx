/**
 * 系统设置 — 运行时参数调节
 */

import { useEffect, useState } from 'react';
import { Card, Typography, Slider, Button, Space, message, Spin, Divider } from 'antd';
import { fetchRuntimeConfig, updateRuntimeConfig } from '../services/api';

const { Text, Title } = Typography;

export default function Settings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [confidence, setConfidence] = useState(0.3);
  const [imgSize, setImgSize] = useState(640);
  const [minArea, setMinArea] = useState(100);
  const [blueHigh, setBlueHigh] = useState(0.5);
  const [yellowHigh, setYellowHigh] = useState(0.7);
  const [orangeHigh, setOrangeHigh] = useState(0.9);

  useEffect(() => {
    loadConfig();
  }, []);

  async function loadConfig() {
    try {
      const data = await fetchRuntimeConfig();
      if (data.detection_confidence) setConfidence(data.detection_confidence);
      if (data.detection_img_size) setImgSize(data.detection_img_size);
      if (data.motion_min_area) setMinArea(data.motion_min_area);
      setLoading(false);
    } catch {
      message.error('加载配置失败');
      setLoading(false);
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      await updateRuntimeConfig({
        detection_confidence: confidence,
        detection_img_size: imgSize,
        motion_min_area: minArea,
        alert_blue_confidence_high: blueHigh,
        alert_yellow_confidence_high: yellowHigh,
        alert_orange_confidence_high: orangeHigh,
      });
      message.success('配置已更新');
    } catch {
      message.error('保存失败');
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '120px auto' }} />;

  return (
    <Card
      title="⚙️ 系统设置"
      bordered={false}
      style={{ background: '#161b22', border: '1px solid #30363d', maxWidth: 800 }}
    >
      <Title level={5} style={{ color: '#c9d1d9' }}>🎯 检测参数</Title>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#8b949e' }}>检测置信度阈值: {confidence.toFixed(2)}</Text>
        <Slider min={0.1} max={0.9} step={0.05} value={confidence} onChange={setConfidence} />
      </div>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#8b949e' }}>推理图像尺寸: {imgSize}</Text>
        <Slider min={320} max={1280} step={32} value={imgSize} onChange={setImgSize} />
      </div>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#8b949e' }}>最小运动区域 (px): {minArea}</Text>
        <Slider min={50} max={2000} step={50} value={minArea} onChange={setMinArea} />
      </div>

      <Divider style={{ borderColor: '#30363d' }} />

      <Title level={5} style={{ color: '#c9d1d9' }}>🚨 四级预警阈值</Title>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#58a6ff' }}>🔵 Ⅳ级蓝色上限: {blueHigh.toFixed(2)}</Text>
        <Slider min={0.2} max={0.7} step={0.05} value={blueHigh} onChange={setBlueHigh} />
      </div>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#d29922' }}>🟡 Ⅲ级黄色上限: {yellowHigh.toFixed(2)}</Text>
        <Slider min={0.4} max={0.85} step={0.05} value={yellowHigh} onChange={setYellowHigh} />
      </div>
      <div style={{ marginBottom: 16 }}>
        <Text style={{ color: '#f0883e' }}>🟠 Ⅱ级橙色上限: {orangeHigh.toFixed(2)}</Text>
        <Slider min={0.6} max={0.99} step={0.05} value={orangeHigh} onChange={setOrangeHigh} />
      </div>

      <Space>
        <Button type="primary" onClick={handleSave} loading={saving}>✅ 应用参数</Button>
        <Button onClick={loadConfig}>🔄 恢复默认</Button>
      </Space>
      <div style={{ marginTop: 8 }}>
        <Text style={{ color: '#8b949e', fontSize: 11 }}>
          注意：检测流水线需重启才能应用新配置
        </Text>
      </div>
    </Card>
  );
}
