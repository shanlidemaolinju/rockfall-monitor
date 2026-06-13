/**
 * 预警记录 — 分页表格 + 日期等级筛选 + 图片预览
 */

import { useEffect, useState, useCallback, useMemo } from 'react';
import { Card, Table, DatePicker, Select, Space, Modal, Tag, Button, message } from 'antd';
import { EyeOutlined } from '@ant-design/icons';
import type { AlertItem, PagedAlerts } from '../services/api';
import { fetchAlertsPaged, reviewAlert } from '../services/api';
import AlertLevelTag from '../components/common/AlertLevelTag';

const { RangePicker } = DatePicker;

export default function AlertRecords() {
  const [data, setData] = useState<PagedAlerts | null>(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [level, setLevel] = useState('');
  const [dateRange, setDateRange] = useState<[string, string]>(['', '']);
  const [previewId, setPreviewId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetchAlertsPaged({
        page, page_size: pageSize,
        start_date: dateRange[0], end_date: dateRange[1],
        alert_level: level,
      });
      setData(r);
    } catch {
      message.error('加载预警记录失败');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, level, dateRange]);

  useEffect(() => { load(); }, [load]);

  const handleReview = useCallback(async (alertId: number, status: string) => {
    try {
      await reviewAlert(alertId, status);
      message.success(status === 'confirmed' ? '已确认真实预警' : '已标记为误报');
      load();
    } catch {
      message.error('审核失败');
    }
  }, [load]);

  const columns = useMemo(() => [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
    { title: '时间', dataIndex: 'time', key: 'time', width: 170 },
    {
      title: '等级', dataIndex: 'alert_level', key: 'level', width: 110,
      render: (l: string) => <AlertLevelTag level={l} />,
    },
    { title: '目标数', dataIndex: 'count', key: 'count', width: 70 },
    {
      title: '最大置信度', dataIndex: 'max_confidence', key: 'conf', width: 100,
      render: (v: number) => v ? `${(v * 100).toFixed(1)}%` : '--',
    },
    {
      title: '跟踪 ID', dataIndex: 'track_ids', key: 'tids', width: 120,
      render: (ids: number[]) => ids?.slice(0, 5).join(', ') || '--',
    },
    { title: '类别分布', dataIndex: 'class_summary', key: 'cls', ellipsis: true },
    {
      title: '推送', dataIndex: 'push_status', key: 'push', width: 80,
      render: (s: string) => {
        const colors: Record<string, string> = { sent: 'green', pending: 'orange', failed: 'red', recorded: 'blue' };
        return <Tag color={colors[s] || 'default'}>{s || '--'}</Tag>;
      },
    },
    {
      title: '审核', key: 'review', width: 100,
      render: (_: unknown, r: AlertItem) => {
        const rv = r.review_status || '';
        if (rv === 'confirmed') return <Tag color="green">✔ 已确认</Tag>;
        if (rv === 'false_alarm') return <Tag color="red">✘ 误报</Tag>;
        return (
          <Space size={4}>
            <Button size="small" type="link" onClick={() => handleReview(r.id, 'confirmed')}>确认</Button>
            <Button size="small" type="link" danger onClick={() => handleReview(r.id, 'false_alarm')}>误报</Button>
          </Space>
        );
      },
    },
    {
      title: '截图', key: 'img', width: 60,
      render: (_: unknown, r: AlertItem) =>
        r.saved_frame ? (
          <Button size="small" type="link" icon={<EyeOutlined />} onClick={() => setPreviewId(r.id)} />
        ) : null,
    },
  ], [handleReview]);

  return (
    <Card
      title="📋 预警记录"
      bordered={false}
      style={{ background: '#161b22', border: '1px solid #30363d' }}
      extra={
        <Space wrap>
          <Select
            value={level}
            onChange={setLevel}
            style={{ width: 120 }}
            options={[
              { value: '', label: '全部等级' },
              { value: 'red', label: 'Ⅰ级红色' },
              { value: 'orange', label: 'Ⅱ级橙色' },
              { value: 'yellow', label: 'Ⅲ级黄色' },
              { value: 'blue', label: 'Ⅳ级蓝色' },
            ]}
          />
          <RangePicker
            onChange={(_, dateStrings) => setDateRange(dateStrings as [string, string])}
            placeholder={['起始日期', '结束日期']}
          />
        </Space>
      }
    >
      <Table
        virtual
        scroll={{ y: 600 }}
        dataSource={data?.rows || []}
        columns={columns}
        rowKey="id"
        loading={loading}
        size="small"
        pagination={{
          current: data?.page || 1,
          total: data?.total || 0,
          pageSize: data?.page_size || 20,
          showSizeChanger: true,
          showTotal: (t) => `共 ${t} 条`,
          onChange: (p, ps) => { setPage(p); setPageSize(ps); },
        }}
        locale={{ emptyText: '暂无预警记录' }}
      />

      {/* 图片预览 */}
      <Modal
        open={previewId !== null}
        footer={null}
        onCancel={() => setPreviewId(null)}
        width="80vw"
        title="现场截图"
      >
        {previewId && (
          <img
            src={`/api/alerts/${previewId}/image`}
            alt="现场截图"
            style={{ width: '100%', borderRadius: 8 }}
          />
        )}
      </Modal>
    </Card>
  );
}
