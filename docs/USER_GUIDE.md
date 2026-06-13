# RockGuard 用户手册

> 公路落石灾害监测预警系统 — Web 看板 & 移动端 H5 使用指南

---

## 目录

- [1. 系统概述](#1-系统概述)
- [2. Web 看板（React SPA）](#2-web-看板react-spa)
- [3. 经典 Web 看板（Jinja2）](#3-经典-web-看板jinja2)
- [4. 移动端 H5 看板](#4-移动端-h5-看板)
- [5. Streamlit 管理界面](#5-streamlit-管理界面)
- [6. API 接口文档](#6-api-接口文档)
- [7. 预警工单流转](#7-预警工单流转)
- [8. 常见操作指南](#8-常见操作指南)

---

## 1. 系统概述

RockGuard 是一个基于 YOLO + MOG2 + SORT 的实时落石检测预警系统，支持四级预警分级（红/橙/黄/蓝）、多监测点位管理、多端可视化。

### 系统入口

| 界面 | 地址 | 适用场景 |
|------|------|----------|
| **Web SPA 看板** | `http://your-server:8000/` | PC 端数据大屏、预警管理 |
| **经典看板** | `http://your-server:8000/classic` | PC 端（暗色主题，无需 Node） |
| **移动端 H5** | `http://your-server:8000/m` | 手机扫码查看预警 |
| **API 文档** | `http://your-server:8000/docs` | 开发调试、接口测试 |
| **Streamlit** | `http://your-server:8501/` | 演示与算法展示 |

### 预警等级说明

系统预警等级对齐《公路自然灾害监测预警系统技术指南》：

| 等级 | 图标颜色 | 含义 | 触发条件（示例） |
|------|---------|------|------------------|
| 🔴 **红色** | `#f85149` | 紧急：大型/高速落石 | 置信度 ≥ 0.6，面积比 ≥ 2% |
| 🟠 **橙色** | `#f0883e` | 重要：中型落石或多块 | 置信度 ≥ 0.4，面积比 ≥ 0.8% |
| 🟡 **黄色** | `#d29922` | 关注：小型落石 | 置信度 ≥ 0.3，运动检测触发 |
| 🔵 **蓝色** | `#58a6ff` | 提示：可疑运动 | MOG2 前景检测异常 |

---

## 2. Web 看板（React SPA）

> 访问地址：`http://your-server:8000/`
>
> 前端技术栈：React 19 + Ant Design 6 + ECharts 6 + Leaflet + Konva

### 2.1 数据大屏（Cockpit）

**路径**：`/`（首页）

**功能**：一站式监控总览

```
┌─────────────────────────────────────────────────────────┐
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ │
│  │ 当日预警  │ │ 红色预警  │ │ 在线设备  │ │ 平均置信度 │ │
│  │   23     │ │    2     │ │    4     │ │  0.72     │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ │
│                                                         │
│  ┌──────────────────────┐ ┌───────────────────────────┐ │
│  │                      │ │                           │ │
│  │  Leaflet 地图分布    │ │  ECharts 24h 预警趋势    │ │
│  │  (预警点位标注)      │ │  (柱状图+折线)           │ │
│  │                      │ │                           │ │
│  └──────────────────────┘ └───────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  实时 MJPEG 视频流 (来自摄像头/RTSP)              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**使用说明**：

- **顶部统计卡片**：实时显示当日预警总数、红色预警数、在线监测点位数、平均检测置信度
- **地图面板**：Leaflet 瓦片地图上标注各监测点位，点击标注查看该点位最近预警
- **趋势图**：24 小时预警数量分布柱状图 + 各等级趋势折线，可切换时间范围
- **视频流**：实时 MJPEG 视频画面（需配置摄像头或上传视频源）

### 2.2 预警记录（AlertRecords）

**路径**：`/alerts`

**功能**：完整预警数据管理与审核

**操作指南**：

| 操作 | 步骤 |
|------|------|
| **查看列表** | 页面加载自动获取，支持分页滚动 |
| **等级筛选** | 点击顶部 红/橙/黄/蓝 标签切换 |
| **日期筛选** | 日期选择器选择起止日期 |
| **查看截图** | 点击预警行的"截图"按钮，弹出大图预览 |
| **审核标记** | 点击"审核"按钮，选择"有效预警"/"误报" |
| **导出 Excel** | 点击"导出"按钮，下载当前筛选结果 |
| **查看详情** | 点击行展开：跟踪 ID、置信度、落石直径、工单状态 |

**筛选参数说明**：

- `alert_level`：red / orange / yellow / blue
- `start_date` / `end_date`：格式 `YYYY-MM-DD`
- `limit`：每页条数（默认 20，最大 100）

### 2.3 点位管理（SiteManagement）

**路径**：`/sites`

**功能**：管理多个监测站点的配置

**系统预设 4 个点位**：

| 点位 ID | 名称 | 地区 | 风险等级 |
|---------|------|------|----------|
| `nanning_naan_s1` | 南宁那安快速路 1 号边坡 | 广西南宁 | 高 |
| `chongzuo_hena_s2` | 崇左合那高速 2 号边坡 | 广西崇左 | 中 |
| `fangchenggang_lanhai_s3` | 防城港兰海高速 3 号边坡 | 广西防城港 | 高 |
| `pingxiang_crossborder_s4` | 凭祥跨境公路 4 号边坡 | 广西凭祥 | 中 |

**操作指南**：

| 操作 | 步骤 |
|------|------|
| **切换激活点位** | 点击目标点位的"激活"按钮，后续检测将使用该点位的摄像头和参数 |
| **新增点位** | 点击"新增"，填写名称/位置/摄像头地址/经纬度/ROI 多边形 |
| **编辑点位** | 点击"编辑"，修改参数后保存 |
| **删除点位** | 点击"删除"，确认后移除（不可恢复） |

**点位字段说明**：

| 字段 | 必填 | 说明 | 示例 |
|------|------|------|------|
| site_id | 是 | 唯一标识 | `nanning_naan_s1` |
| name | 是 | 点位名称 | `南宁那安快速路 1 号边坡` |
| location | 否 | 详细地址 | `南宁市兴宁区那安快速路 K32+500` |
| region | 否 | 所属地区 | `广西南宁` |
| camera_url | 否 | RTSP / 文件路径 | `rtsp://admin:pass@192.168.1.100:554/stream` |
| latitude | 否 | 纬度 | `22.817` |
| longitude | 否 | 经度 | `108.366` |
| highway | 否 | 公路编号 | `G75` |
| stake_mark | 否 | 桩号 | `K32+500` |
| risk_level | 否 | 风险等级 | `high` / `medium` / `low` |
| roi_polygon | 否 | ROI 多边形坐标 | JSON 数组 |
| alert_contacts | 否 | 预警联系人 | JSON 对象 |

### 2.4 地图视图（MapView）

**路径**：`/map`

**功能**：预警地理分布可视化

**使用说明**：

- 地图上以不同颜色标记不同等级的预警事件
- 点击标记查看该预警的详细信息（时间、等级、截图缩略图）
- 支持缩放和平移，可切换卫星图/街道图
- 右上角图例说明颜色含义
- 支持按时间和等级过滤显示

### 2.5 视频检测（VideoDetection）

**路径**：`/video-detect`

**功能**：上传视频文件进行离线检测

**操作流程**：

1. 点击或拖拽上传视频文件（支持 .mp4, .avi, .mov, .mkv）
2. 选择检测参数（可选）：
   - 置信度阈值（默认 0.3）
   - 是否启用边缘增强
   - 是否启用 SAHI 切片
3. 点击"开始检测"
4. **实时进度**：WebSocket 连接显示进度条和当前帧号
5. 检测完成后显示结果汇总（总帧数、检测到的落石数、各等级预警数）
6. 可下载标注视频或查看截图集

**进度信息示例**：

```json
{
  "status": "processing",
  "progress": 45.2,
  "current_frame": 452,
  "total_frames": 1000
}
```

**异步模式**：大文件检测支持异步执行，提交后生成 task_id，可在任务列表中查看进度，完成后通知。

### 2.6 ROI 标定（RoiCalibration）

**路径**：`/roi`

**功能**：框选监测区域（Region of Interest），排除无关区域

**操作指南**：

1. 上传或加载监测场景截图
2. 使用 **Konva Canvas** 工具：
   - **多边形框选**：点击添加顶点，双击闭合多边形
   - **矩形框选**：拖拽绘制矩形
   - **编辑顶点**：拖拽已有顶点调整形状
   - **删除区域**：选中后按 Delete 键
3. 点击"保存"将 ROI 多边形关联到当前激活点位
4. 可选：点击"FastSAM 辅助"自动分割道路/边坡区域

**ROI 作用**：

- 只检测 ROI 区域内的落石，减少误报
- 树木晃动、车辆经过等无关运动被忽略
- 支持多个 ROI 多边形同时生效

**ROI 热力图**（`/api/roi/heatmap`）：

- 显示历史预警在画面中的空间分布热力图
- 辅助判断高风险区域，优化 ROI 多边形

### 2.7 系统设置（Settings）

**路径**：`/settings`

**功能**：系统参数实时调整

**可配置项**：

| 分类 | 参数 | 说明 |
|------|------|------|
| 检测 | `DETECTION_CONFIDENCE` | YOLO 置信度阈值 |
| 检测 | `DETECTION_IMG_SIZE` | 推理分辨率 |
| 预警 | `ALERT_RED/YELLOW/ORANGE_CONFIDENCE` | 各等级置信度阈值 |
| 跳帧 | `SKIP_IDLE/ACTIVE/CRITICAL` | 三级自适应跳帧间隔 |
| MOG2 | `MOG2_HISTORY` | 背景建模历史帧数 |
| 跟踪 | `TRACK_MIN_CONFIRM` | 跟踪确认帧数 |
| 推送 | `ALERT_CHANNEL_MAP` | 推送通道分配 |
| 模型 | `MODEL_SLOT_MAP` | 时段模型映射 |

**操作**：修改参数 → 点击"保存" → 立即生效（热更新，无需重启）

**注意**：部分参数（如 `MYSQL_*`）需要重启服务才能生效。

---

## 3. 经典 Web 看板（Jinja2）

> 访问地址：`http://your-server:8000/classic`
>
> 无需 Node.js，纯 HTML + CSS + JS

### 3.1 页面布局

- **深色主题**（dark mode），护眼低亮度
- **顶栏**：系统标题 + 当前时间和激活点位
- **左侧**：四级预警统计卡片 + 最近预警列表
- **中央**：实时视频画面（MJPEG）
- **底部**：预警记录表格（支持等级筛选、日期筛选）

### 3.2 功能

| 功能 | 操作 |
|------|------|
| 查看预警列表 | 页面自动刷新（SSE 实时推送） |
| 筛选预警 | 顶部下拉框选择等级 |
| 查看截图 | 点击"查看截图"链接 |
| 切换点位 | 顶部点位下拉选择 |

---

## 4. 移动端 H5 看板

> 访问地址：`http://your-server:8000/m` 或 `http://your-server:8000/mobile`
>
> 技术栈：HTML + Tailwind CSS + Vanilla JS

### 4.1 扫码访问

**方式一：生成二维码**

```bash
# 在服务器上生成二维码（终端）
pip install qrcode
qr "http://your-server-ip:8000/m" > rockguard_qr.png
```

**方式二：浏览器访问**

手机浏览器直接输入 `http://your-server:8000/m`。

### 4.2 页面功能

```
┌──────────────────────────┐
│  RockGuard 移动端        │
│  当前点位: 南宁1号边坡    │
├──────────────────────────┤
│  [红] [橙] [黄] [蓝]     │  ← 四级预警标签筛选
│  [刷新]                  │
├──────────────────────────┤
│  ┌────────────────────┐  │
│  │ 🔴 红色预警  2次    │  │  ← 预警卡片
│  │ 2026-06-14 15:23   │  │
│  │ 置信度: 0.87       │  │
│  │ [查看截图]          │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ 🟠 橙色预警  5次    │  │
│  │ ...                │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ 🟡 黄色预警  8次    │  │
│  │ ...                │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ 🔵 蓝色预警  12次   │  │
│  │ ...                │  │
│  └────────────────────┘  │
│                          │
│  共 27 条预警记录        │  ← 底部统计
└──────────────────────────┘
```

### 4.3 操作说明

| 操作 | 方法 |
|------|------|
| **筛选等级** | 点击顶部红/橙/黄/蓝标签（支持多选） |
| **刷新数据** | 下拉页面或点击"刷新"按钮（触发旋转动画） |
| **查看截图** | 点击预警卡片 → 弹出大图弹窗（可双指缩放） |
| **返回列表** | 点击弹窗外部或按返回键 |
| **切换点位** | 暂不支持，需在 PC 端 /settings 切换 |

### 4.4 PWA 特性

移动端页面支持添加到手机主屏幕：

- **iOS Safari**：点击分享按钮 → "添加到主屏幕"
- **Android Chrome**：点击菜单 → "添加到主屏幕"
- 添加后以独立应用形式打开（无浏览器地址栏）
- 启动画面颜色：`#0d1117`（深色）

---

## 5. Streamlit 管理界面

> 访问地址：`http://your-server:8501/`
>
> 10 个页面，通过左侧侧边栏导航

### 5.1 页面列表

| # | 页面 | 功能 |
|---|------|------|
| 1 | **Preset Demo** | 预计算结果展示，零等待加载 |
| 2 | **Live Detection** | 实时检测（上传视频 / RTSP / USB） |
| 3 | **Multi-Camera** | 多路摄像头同时监控 |
| 4 | **Algorithm** | 算法亮点展示：流水线可视化 + FPS 对比 + Kalman 轨迹 |
| 5 | **Extreme Scenarios** | 极端场景验证（6 种条件） |
| 6 | **Alert Standards** | 四级预警标准详细说明 |
| 7 | **Alert Records** | 预警记录查询和管理 |
| 8 | **Site Manager** | 监测点位管理（CRUD） |
| 9 | **Settings** | 系统参数设置 |
| 10 | **System** | 系统管理（日志、健康检查） |

### 5.2 Live Detection 使用流程

1. 选择输入源：上传视频文件 / RTSP 地址 / USB 摄像头（`0`）
2. 配置检测参数（可选展开"高级设置"）
3. 点击"开始检测"
4. 实时查看：视频画面 + 检测框 + 预警列表
5. 检测完成自动保存截图和日志

### 5.3 Algorithm 页面

展示核心技术原理，适合比赛展示：

- 流水线可视化：输入 → MOG2 → YOLO → SORT → 分级
- FPS 实时对比：各模块耗时拆解
- Kalman 跟踪轨迹：落石运动路径可视化
- 四级分级决策树

---

## 6. API 接口文档

### 6.1 自动生成文档

系统启动后自动生成交互式 API 文档：

| 文档 | 地址 |
|------|------|
| **Swagger UI** | `http://your-server:8000/docs` |
| **ReDoc** | `http://your-server:8000/redoc` |
| **OpenAPI JSON** | `http://your-server:8000/openapi.json` |

### 6.2 Swagger UI 使用

1. 浏览器打开 `http://your-server:8000/docs`
2. 点击展开任意接口
3. 点击 "Try it out" 按钮
4. 填入参数
5. 点击 "Execute" 测试接口
6. 查看返回结果和响应状态码

### 6.3 API 认证

系统支持两种认证方式（`/api/auth/login` 接口换取 JWT）：

```bash
# 方式一：API Key 认证（推荐）
curl -H "X-Api-Key: your_api_key" http://localhost:8000/api/alerts?limit=10

# 方式二：Bearer Token 认证
curl -H "Authorization: Bearer <jwt_token>" http://localhost:8000/api/alerts?limit=10
```

**获取 JWT Token**：

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your_api_key"}'
# 返回: {"access_token": "...", "token_type": "bearer"}
```

### 6.4 核心 API 速查

#### 预警查询

```bash
# 最近预警
GET /api/alerts?limit=10&alert_level=red

# 分页查询
GET /api/alerts/paged?page=1&page_size=20&start_date=2026-06-01&end_date=2026-06-14

# SSE 实时推送
GET /api/alerts/stream

# 获取截图
GET /api/alerts/{id}/image
```

#### 检测接口

```bash
# 图片检测
POST /detect/image
Content-Type: multipart/form-data
file: @test.jpg

# 视频检测（同步，小文件）
POST /detect/video
Content-Type: multipart/form-data
file: @test.mp4

# 视频检测（异步，大文件）
POST /detect/video
Content-Type: multipart/form-data
file: @test.mp4
async: true
# 返回: {"task_id": "abc123"}

# 查询任务进度
GET /api/tasks/{task_id}
# 或 WebSocket
WS /ws/tasks/{task_id}
```

#### 点位管理

```bash
# 获取全部点位
GET /api/sites

# 切换激活点位
POST /api/sites/switch
{"site_id": "nanning_naan_s1"}
```

#### 配置管理

```bash
# 查看当前配置
GET /api/config/current

# 热更新参数
POST /api/config/update
{"DETECTION_CONFIDENCE": 0.4, "SKIP_IDLE": 10}

# 热重载 .env
POST /api/config/reload
```

#### 系统健康

```bash
# 基础健康
GET /health

# 完整诊断
GET /api/health/full

# Prometheus 指标
GET /metrics
```

### 6.5 SSE 实时预警流

```javascript
// JavaScript 客户端示例
const eventSource = new EventSource('http://your-server:8000/api/alerts/stream');

eventSource.addEventListener('alert', (event) => {
  const alert = JSON.parse(event.data);
  console.log('新预警:', alert.alert_level, alert.count);
});

eventSource.addEventListener('heartbeat', () => {
  // 每 30 秒心跳，保持连接
});

eventSource.onerror = () => {
  console.log('SSE 连接断开，将自动重连');
};
```

---

## 7. 预警工单流转

### 7.1 工单状态机

```
触发预警 ──▶ [new] ──▶ [reviewed] ──▶ [confirmed] ──▶ [dispatched] ──▶ [resolved]
                 │          │              │               │
                 ▼          ▼              ▼               ▼
              [dismissed] [escalated]   [false_alarm]   [archived]
```

### 7.2 流转操作

```bash
# 查看工单状态
GET /api/alerts/{id}/workflow

# 流转工单状态
POST /api/alerts/{id}/workflow
{
  "action": "confirm",     // dismiss | confirm | escalate | dispatch | resolve | archive
  "operator": "张三",
  "comment": "已确认落石，已通知养护班组"
}
```

### 7.3 工单统计

```bash
GET /api/workflow/stats
# 返回各状态的工单数量统计
```

---

## 8. 常见操作指南

### 8.1 新监测点位上线流程

1. **PC 端** → 点位管理 → 新增点位 → 填写信息
2. 配置该点位的摄像头 RTSP 地址
3. **ROI 标定** → 上传场景截图 → 框选监测区域 → 保存
4. **激活点位** → 在点位列表中点击"激活"
5. 验证：观察实时视频流和检测结果
6. 可选：配置该点位专用的预警推送联系人和模型

### 8.2 预警误报处理

1. **查看详情**：点击预警记录，查看截图和检测信息
2. **判断类型**：
   - 非落石物体（车辆、行人、飞鸟）→ 标记为"误报"
   - 实际落石但等级不对 → 标记审核意见
   - 检测正确 → 标记为"有效预警"
3. **调整参数**（如需减少误报）：
   - 提高置信度阈值：`DETECTION_CONFIDENCE`
   - 缩小 ROI：排除道路、树木等干扰区域
   - 启用运动滤波：`TFD_ENABLED=true`
   - 启用时序确认：`TEMPORAL_ENABLED=true`

### 8.3 大批量数据导出

```bash
# API 导出（Excel 格式）
curl -o alerts_202606.xlsx \
  "http://localhost:8000/api/alerts/export?start_date=2026-06-01&end_date=2026-06-14&alert_level=red"

# 先预览再导出
curl "http://localhost:8000/api/alerts/export/summary?start_date=2026-06-01&end_date=2026-06-14"
```

### 8.4 切换模型版本

```bash
# 查看可用模型
GET /api/models
# 返回: {"models": ["rock_best.pt", "rock_v2.pt", "rock_night.pt"], "current": "rock_best.pt"}

# 切换模型
POST /api/models/switch
{"model_path": "models/rock_v2.pt"}

# 按时段自动切换
# 在 .env 中配置：
# MODEL_SLOT_MAP=0-6=models/rock_night.pt;19-23=models/rock_night.pt
```

---

## 附录 A：键盘快捷键（Web SPA）

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + P` | 切换点位 |
| `Ctrl + R` | 刷新数据 |
| `Esc` | 关闭弹窗/模态框 |

## 附录 B：浏览器兼容性

| 浏览器 | 版本 | 支持 |
|--------|------|------|
| Chrome | 90+ | ✅ 完全支持 |
| Edge | 90+ | ✅ 完全支持 |
| Firefox | 90+ | ✅ 完全支持 |
| Safari | 15+ | ✅ 完全支持（含 iOS） |
| 微信内置浏览器 | 最新版 | ✅ 支持（移动端 H5） |

## 附录 C：比赛展示建议

1. **数据大屏**：使用 `Cockpit` 页面，全屏模式（F11），适合投影展示
2. **算法演示**：使用 Streamlit `Algorithm` 页面，展示流水线动画和 FPS 对比
3. **极端场景**：使用 Streamlit `Extreme Scenarios` 页面，展示 6 种恶劣条件下的检测鲁棒性
4. **移动端**：评委用手机扫描二维码体验 H5 看板
5. **实时检测**：现场接入 USB 摄像头或播放落石视频，展示实时检测效果
