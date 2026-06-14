#!/bin/bash
# ============================================================
# RockGuard 容器启动脚本
# ============================================================
# 1. 等待 MySQL 就绪（若配置了 MYSQL_HOST）
# 2. 运行 Alembic 数据库迁移
# 3. 种子演示凭据（API Key）
# 4. 启动 uvicorn
# ============================================================
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🪨 RockGuard · 公路落石监测预警平台"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. 等待 MySQL（可选）──
if [ -n "${MYSQL_HOST:-}" ]; then
    echo "[entrypoint] 等待 MySQL ${MYSQL_HOST}:${MYSQL_PORT:-3306} ..."
    RETRIES=30
    while [ $RETRIES -gt 0 ]; do
        if python -c "
import pymysql, os
try:
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT', 3306)),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE', 'rock'),
        connect_timeout=3,
    )
    conn.close()
    exit(0)
except Exception:
    exit(1)
" 2>/dev/null; then
            echo "[entrypoint] ✓ MySQL 已就绪"
            break
        fi
        RETRIES=$((RETRIES - 1))
        echo "[entrypoint] 等待中... (剩余 $RETRIES 次)"
        sleep 2
    done
    if [ $RETRIES -le 0 ]; then
        echo "[entrypoint] ⚠ MySQL 未就绪，将使用 SQLite 降级"
    fi
else
    echo "[entrypoint] 未配置 MySQL，使用 SQLite"
fi

# ── 2. Alembic 迁移 ──
echo "[entrypoint] 运行数据库迁移..."
python -m alembic upgrade head
echo "[entrypoint] ✓ 数据库迁移完成"

# ── 3. 演示凭据提示 ──
if [ -n "${API_KEY:-}" ] && [ "${API_KEY}" != "your_token_here" ]; then
    echo "[entrypoint] ──────────────────────────────"
    echo "[entrypoint] 🔑 演示登录凭据:"
    echo "[entrypoint]    账号: admin"
    echo "[entrypoint]    密码: ${API_KEY}"
    echo "[entrypoint] ──────────────────────────────"
fi

# ── 4. 启动 uvicorn ──
echo "[entrypoint] 启动 FastAPI 服务..."
exec uvicorn server.main:app \
    --proxy-headers \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
