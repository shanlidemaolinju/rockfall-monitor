@echo off
chcp 65001 >nul
title 落石检测系统 - Web 服务
echo ========================================
echo   落石检测系统 - Web 监控看板
echo   钦州监测点
echo ========================================
echo.
echo   启动后访问:
echo   仪表盘:  http://localhost:8000/
echo   API文档: http://localhost:8000/docs
echo.
cd /d "%~dp0"
uvicorn server.main:app --host 0.0.0.0 --port 8000
pause
