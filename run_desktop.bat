@echo off
chcp 65001 >nul
title 落石检测系统 - 桌面监控
echo ========================================
echo   落石检测系统 - 桌面实时监控
echo   钦州监测点
echo ========================================
echo.
cd /d "%~dp0"
python -m desktop.main
pause
