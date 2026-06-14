/**
 * 登录页面 — 用户名 + 密码 认证
 *
 * 将密码作为 API Key 发送到 /api/auth/login 换取 JWT，
 * 登录成功后跳转到首页。
 */

import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, Form, Input, Button, Typography, Space, theme } from 'antd';
import {
  SafetyCertificateOutlined,
  UserOutlined,
  LockOutlined,
} from '@ant-design/icons';
import { useAppStore } from '../stores/useAppStore';
import api from '../services/api';

const { Title, Text } = Typography;

export default function LoginPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const setAuth = useAppStore((s) => s.setAuth);
  const { token: themeToken } = theme.useToken();

  const handleSubmit = async (values: { username: string; password: string }) => {
    setLoading(true);
    setError('');

    try {
      // 将密码作为 API Key 发送，username 作为 client 标识
      const formData = new URLSearchParams();
      formData.append('api_key', values.password);
      formData.append('client', values.username || 'admin');
      formData.append('label', `Web Login · ${values.username || 'admin'}`);
      formData.append('expires_hours', '24');

      const res = await api.post('/api/auth/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });

      if (res.data?.access_token) {
        const token = res.data.access_token;
        localStorage.setItem('rockguard_token', token);
        setAuth(token);

        // 跳转到 redirect 参数指定的页面，或默认首页
        const redirect = searchParams.get('redirect') || '/';
        navigate(redirect, { replace: true });
      } else {
        setError('服务器返回异常，请联系管理员');
      }
    } catch (err: any) {
      if (err.response?.status === 401) {
        setError('账号或密码错误');
      } else if (err.response?.status === 400) {
        setError('请输入账号和密码');
      } else {
        setError(err.response?.data?.detail || '登录失败，请稍后重试');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%)',
        padding: 24,
      }}
    >
      <Card
        style={{
          width: 420,
          maxWidth: '100%',
          background: themeToken.colorBgContainer,
          borderColor: themeToken.colorBorder,
          boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
        }}
        styles={{ body: { padding: '40px 32px' } }}
      >
        {/* Logo + 标题 */}
        <Space direction="vertical" size={4} style={{ width: '100%', textAlign: 'center', marginBottom: 32 }}>
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #58a6ff 0%, #3fb950 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
              fontSize: 32,
            }}
          >
            🪨
          </div>
          <Title level={3} style={{ margin: 0, color: themeToken.colorText }}>
            RockGuard
          </Title>
          <Text type="secondary" style={{ fontSize: 13 }}>
            公路自然灾害监测预警平台
          </Text>
        </Space>

        {/* 登录表单 */}
        <Form
          onFinish={handleSubmit}
          size="large"
          autoComplete="off"
          initialValues={{ username: '', password: '' }}
        >
          <Form.Item
            name="username"
            rules={[{ required: true, message: '请输入账号' }]}
          >
            <Input
              prefix={<UserOutlined style={{ color: themeToken.colorTextQuaternary }} />}
              placeholder="账号"
              autoFocus
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password
              prefix={<LockOutlined style={{ color: themeToken.colorTextQuaternary }} />}
              placeholder="密码"
            />
          </Form.Item>

          {/* 错误提示 */}
          {error && (
            <div
              style={{
                color: '#f85149',
                fontSize: 13,
                marginBottom: 16,
                padding: '8px 12px',
                background: 'rgba(248,81,73,0.1)',
                borderRadius: 6,
                border: '1px solid rgba(248,81,73,0.2)',
              }}
            >
              {error}
            </div>
          )}

          <Form.Item style={{ marginBottom: 16 }}>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
              icon={<SafetyCertificateOutlined />}
              style={{ height: 44 }}
            >
              登 录
            </Button>
          </Form.Item>
        </Form>

        {/* 底部信息 */}
        <div style={{ textAlign: 'center' }}>
          <Space split={<span style={{ color: themeToken.colorBorder }}>·</span>}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              🟢 演示模式
            </Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              v2.2.0
            </Text>
          </Space>
        </div>
      </Card>
    </div>
  );
}
