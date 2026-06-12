"""
server — FastAPI Web 服务
=========================
本包是落石检测系统的 HTTP API 服务端。

分层:
  main.py    — 入口 + 路由定义(API层)
  service.py — 业务逻辑层, 封装对 rockfall.detector 的调用
  schemas.py — 请求/响应数据模型(可选, 随接口增多添加)

依赖: rockfall (核心库)

运行: uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
