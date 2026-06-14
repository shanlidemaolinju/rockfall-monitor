@echo off
title RockGuard Demo

echo.
echo ================================================
echo   RockGuard - Highway Rockfall Monitoring
echo ================================================
echo.

REM ---- Check Docker ----
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM ---- Check .env ----
if not exist .env (
    echo [ERROR] .env file not found. Copy .env.example to .env and configure.
    pause
    exit /b 1
)

REM ---- Build and start ----
echo [1/3] Building Docker image (first time may take 5-15 min)...
docker compose -f docker-compose.demo.yml build
if errorlevel 1 (
    echo [ERROR] Docker build failed.
    pause
    exit /b 1
)

echo [2/3] Starting services...
docker compose -f docker-compose.demo.yml up -d
if errorlevel 1 (
    echo [ERROR] Service startup failed.
    pause
    exit /b 1
)

echo [3/3] Waiting for services to be ready...
timeout /t 5 /nobreak >nul

echo.
echo ================================================
echo   Services started successfully!
echo ================================================
echo.
echo   Local URL:  http://localhost
echo   API Docs:   http://localhost:8000/docs
echo   Health:     http://localhost:8000/health/ready
echo.
echo   Login credentials:
echo     Username: admin
echo     Password: rockfall2024
echo.
echo ================================================
echo.
echo   Stop:    docker compose -f docker-compose.demo.yml down
echo   Logs:    docker compose -f docker-compose.demo.yml logs -f rockfall_api
echo.
pause
