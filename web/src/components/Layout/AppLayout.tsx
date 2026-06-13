/**
 * 应用主布局 — Ant Design Layout
 * 顶栏 + 可折叠侧边栏 + 内容区
 */

import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Button, Badge, Space, Typography, Tooltip } from 'antd';
import {
  DashboardOutlined,
  AlertOutlined,
  EnvironmentOutlined,
  CompassOutlined,
  AimOutlined,
  VideoCameraOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SoundOutlined,
  SoundFilled,
  BellOutlined,
} from '@ant-design/icons';
import { useAppStore } from '../../stores/useAppStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '🖥️ 数据大屏' },
  { key: '/alerts', icon: <AlertOutlined />, label: '预警记录' },
  { key: '/map', icon: <CompassOutlined />, label: '地图监控' },
  { key: '/roi', icon: <AimOutlined />, label: 'ROI 标定' },
  { key: '/sites', icon: <EnvironmentOutlined />, label: '点位管理' },
  { key: '/video-detect', icon: <VideoCameraOutlined />, label: '视频检测' },
  { key: '/settings', icon: <SettingOutlined />, label: '系统设置' },
];

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const stats = useAppStore((s) => s.stats);
  const soundEnabled = useAppStore((s) => s.soundEnabled);
  const toggleSound = useAppStore((s) => s.toggleSound);
  const sidebarCollapsed = useAppStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);

  const totalAlerts = stats
    ? stats.today_red + stats.today_orange + stats.today_yellow + stats.today_blue
    : 0;

  // 子路由匹配
  const selectedKey = menuItems.find((item) => {
    if (item.key === '/') return location.pathname === '/';
    return location.pathname.startsWith(item.key);
  })?.key || '/';

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* ━━━ 侧边栏 ━━━ */}
      <Sider
        trigger={null}
        collapsible
        collapsed={sidebarCollapsed}
        breakpoint="lg"
        collapsedWidth={0}
        onBreakpoint={(broken) => {
          if (broken !== sidebarCollapsed) toggleSidebar();
        }}
        theme="dark"
        width={220}
        style={{
          borderRight: '1px solid #30363d',
          background: '#161b22',
        }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #30363d',
          }}
        >
          {sidebarCollapsed ? (
            <Text strong style={{ color: '#58a6ff', fontSize: 18 }}>🪨</Text>
          ) : (
            <Space direction="vertical" size={0} style={{ textAlign: 'center' }}>
              <Text strong style={{ color: '#c9d1d9', fontSize: 16 }}>🪨 RockGuard</Text>
              <Text style={{ color: '#8b949e', fontSize: 11 }}>公路自然灾害监测预警平台</Text>
            </Space>
          )}
        </div>

        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{
            background: 'transparent',
            borderRight: 'none',
            marginTop: 8,
          }}
        />
      </Sider>

      {/* ━━━ 主内容区 ━━━ */}
      <Layout>
        <Header
          style={{
            background: '#161b22',
            borderBottom: '1px solid #30363d',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            height: 56,
          }}
        >
          <Space>
            <Button
              type="text"
              icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={toggleSidebar}
              style={{ color: '#8b949e', fontSize: 16 }}
            />
            <Text style={{ color: '#c9d1d9', fontSize: 14 }}>
              {menuItems.find((m) => m.key === selectedKey)?.label || '监测大屏'}
            </Text>
          </Space>

          <Space size="middle">
            {/* 今日预警数 */}
            <Tooltip title="今日预警总数">
              <Badge count={totalAlerts} overflowCount={999} size="small">
                <BellOutlined style={{ color: '#f85149', fontSize: 18 }} />
              </Badge>
            </Tooltip>

            {/* 声音开关 */}
            <Tooltip title={soundEnabled ? '关闭报警声音' : '开启报警声音'}>
              <Button
                type="text"
                icon={soundEnabled ? <SoundFilled /> : <SoundOutlined />}
                onClick={toggleSound}
                style={{ color: soundEnabled ? '#58a6ff' : '#8b949e' }}
              />
            </Tooltip>

            <Text style={{ color: '#8b949e', fontSize: 12 }}>v2.2.0</Text>
          </Space>
        </Header>

        <Content
          style={{
            padding: 24,
            background: '#0d1117',
            minHeight: 'calc(100vh - 56px)',
            overflow: 'auto',
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
