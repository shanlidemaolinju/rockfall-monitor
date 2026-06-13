/**
 * 视频检测 — 异步上传 + WebSocket 进度推送 (从 dashboard.html 迁移)
 */

import { useState, useCallback, useEffect } from 'react';
import { Card, Button, Checkbox, Select, Typography, Progress, Space, Descriptions, Tag } from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import { uploadVideo, type TaskResponse } from '../services/api';
import { useTaskWebSocket } from '../hooks/useWebSocket';

const { Text } = Typography;

export default function VideoDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [saveFrames, setSaveFrames] = useState(true);
  const [pushAlerts, setPushAlerts] = useState(false);
  const [cameraId, setCameraId] = useState('default');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [taskList, setTaskList] = useState<{ id: string; name: string; status: string; time: string }[]>([]);

  const ws = useTaskWebSocket(taskId);

  // 当有 taskId 时连接 WebSocket
  useEffect(() => {
    if (taskId) ws.connect();
  }, [taskId]);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    try {
      const result: TaskResponse = await uploadVideo(file, { save_frames: saveFrames, push_alerts: pushAlerts, camera_id: cameraId });
      setTaskId(result.task_id);
      setTaskList((prev) => [
        { id: result.task_id, name: file.name, status: 'processing', time: new Date().toLocaleTimeString() },
        ...prev,
      ]);
    } catch (e: any) {
      console.error('上传失败', e);
    } finally {
      setUploading(false);
    }
  }, [file, saveFrames, pushAlerts, cameraId]);

  // 监听 WebSocket 状态更新任务列表（移到 useEffect 中避免 render 中 setState）
  useEffect(() => {
    if (taskId && ws.status) {
      setTaskList((prev) =>
        prev.map((t) => (t.id === taskId ? { ...t, status: ws.status } : t)),
      );
    }
  }, [ws.status, taskId]);

  const progressPct = ws.progress || 0;

  return (
    <div>
      <Card
        title="🎬 上传视频检测"
        bordered={false}
        style={{ background: '#161b22', border: '1px solid #30363d', marginBottom: 16 }}
      >
        {/* 上传区域 */}
        <div
          style={{
            border: '2px dashed #30363d',
            borderRadius: 8,
            padding: '40px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'border-color .2s',
            marginBottom: 16,
          }}
          onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = '#58a6ff'; }}
          onDragLeave={(e) => { e.currentTarget.style.borderColor = '#30363d'; }}
          onDrop={(e) => {
            e.preventDefault();
            e.currentTarget.style.borderColor = '#30363d';
            const f = e.dataTransfer.files[0];
            if (f && f.type.startsWith('video/')) setFile(f);
          }}
          onClick={() => document.getElementById('video-file-input')?.click()}
        >
          <InboxOutlined style={{ fontSize: 40, color: '#8b949e' }} />
          <div style={{ marginTop: 8, color: '#c9d1d9' }}>
            {file ? file.name : '点击选择视频文件 或拖拽文件到此处'}
          </div>
          <Text style={{ color: '#8b949e', fontSize: 12 }}>支持 mp4 / avi / mov / mkv，最大 2GB</Text>
          <input
            id="video-file-input"
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) setFile(f);
            }}
          />
        </div>

        {/* 选项 */}
        <Space wrap style={{ marginBottom: 16 }}>
          <Checkbox checked={saveFrames} onChange={(e) => setSaveFrames(e.target.checked)}>
            保存检测帧截图
          </Checkbox>
          <Checkbox checked={pushAlerts} onChange={(e) => setPushAlerts(e.target.checked)}>
            推送预警通知
          </Checkbox>
          <Space>
            <Text style={{ color: '#8b949e', fontSize: 12 }}>监测点位:</Text>
            <Select
              value={cameraId}
              onChange={setCameraId}
              style={{ width: 160 }}
              options={[{ value: 'default', label: '默认点位' }]}
            />
          </Space>
        </Space>

        <Button
          type="primary"
          icon={<UploadOutlined />}
          onClick={handleUpload}
          loading={uploading}
          disabled={!file}
          size="large"
        >
          {uploading ? '上传中...' : '提交检测'}
        </Button>

        {/* 进度条 */}
        {taskId && (
          <div style={{ marginTop: 20 }}>
            <Progress
              percent={Math.round(progressPct)}
              status={ws.status === 'failed' ? 'exception' : ws.status === 'completed' ? 'success' : 'active'}
              strokeColor={{ from: '#58a6ff', to: '#3fb950' }}
            />
            <Space style={{ marginTop: 4 }}>
              <Text style={{ color: '#8b949e', fontSize: 12 }}>
                帧 {ws.currentFrame} / {ws.totalFrames || '?'}
              </Text>
              <Tag color={ws.status === 'completed' ? 'green' : ws.status === 'failed' ? 'red' : 'blue'}>
                {ws.status === 'processing' ? `检测中 ${progressPct.toFixed(0)}%` :
                 ws.status === 'completed' ? '✅ 完成' :
                 ws.status === 'failed' ? '❌ 失败' :
                 ws.status}
              </Tag>
              <Text style={{ color: '#8b949e', fontSize: 11 }}>task: {taskId.slice(0, 8)}...</Text>
            </Space>

            {/* 结果摘要 */}
            {ws.status === 'completed' && ws.result && (
              <Descriptions size="small" bordered style={{ marginTop: 12, background: '#0d1117' }}>
                <Descriptions.Item label="总帧数">{String(ws.result.total_frames || '--')}</Descriptions.Item>
                <Descriptions.Item label="检出帧">{String(ws.result.frames_with_detections || '--')}</Descriptions.Item>
                <Descriptions.Item label="分辨率">{String(ws.result.resolution || '--')}</Descriptions.Item>
              </Descriptions>
            )}
            {ws.status === 'failed' && ws.error && (
              <Text type="danger" style={{ marginTop: 8, display: 'block' }}>{ws.error}</Text>
            )}
          </div>
        )}
      </Card>

      {/* 历史任务 */}
      {taskList.length > 0 && (
        <Card
          title="📋 最近任务"
          bordered={false}
          style={{ background: '#161b22', border: '1px solid #30363d' }}
        >
          {taskList.map((t) => (
            <div
              key={t.id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 12px',
                background: '#0d1117',
                borderRadius: 4,
                marginBottom: 4,
              }}
            >
              <Space>
                <Text style={{ color: '#c9d1d9', fontSize: 13 }}>{t.name}</Text>
                <Text style={{ color: '#8b949e', fontSize: 11 }}>{t.time}</Text>
              </Space>
              <Tag color={t.status === 'completed' ? 'green' : t.status === 'failed' ? 'red' : 'blue'}>
                {t.status === 'completed' ? '✅' : t.status === 'failed' ? '❌' : '⏳'}
              </Tag>
            </div>
          ))}
        </Card>
      )}
    </div>
  );
}
