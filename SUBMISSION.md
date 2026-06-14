# RockGuard · 公路落石监测预警平台

## 比赛提交信息

| 项目 | 内容 |
|------|------|
| **项目网址** | （启动后填入 ngrok/cpolar 公网地址） |
| **登录账号** | `admin` |
| **登录密码** | `rockfall2024` |
| **展示视频** | （B站/YouTube 链接待填入） |

---

## 项目简介

RockGuard 是一套基于深度学习的公路边坡落石实时监测预警系统，已对齐交通运输部《公路自然灾害监测预警系统技术指南》四级预警标准。

### 核心功能

1. **实时检测大屏** — ECharts 趋势图 + Leaflet 地图 + MJPEG 视频流 + 实时预警弹窗
2. **预警记录管理** — 分页查询、日期/等级筛选、审核标记（确认/误报）、Excel 导出
3. **地图监控** — 多站点经纬度标注 + 预警热力分布
4. **ROI 标定** — Konva 画布绘制检测区域多边形
5. **点位管理** — 5 个预设监测站点（广西+东盟跨境），支持增删改查
6. **视频检测** — 上传视频 + WebSocket 进度推送 + 异步任务管理

### 技术架构

```
React SPA (Ant Design) → Nginx → FastAPI → RockDetector (YOLO + SORT + MOG2)
                                          → AlertStore (SQLite/MySQL + 哈希链)
                                          → PushPlus (微信推送)
```

### 算法亮点

- **YOLOv8 + SORT 多目标跟踪**：落石检测与轨迹预测
- **MOG2 背景建模**：运动区域初筛，减少 YOLO 误检
- **四级预警分级**：对齐交通部标准（Ⅰ红/Ⅱ橙/Ⅲ黄/Ⅳ蓝）
- **哈希链防篡改**：SHA256 链式摘要，满足审计合规
- **隐私脱敏**：人脸/车牌自动模糊，符合《个人信息保护法》

---

## 本地启动

```bash
# Docker 方式（推荐）
docker compose -f docker-compose.demo.yml up -d

# 或手动方式
pip install -r requirements-lock.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

访问 http://localhost 即可打开看板。
