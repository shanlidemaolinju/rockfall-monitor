#!/bin/bash
# ============================================================
# 落石监测系统 — 腾讯云服务器首次部署初始化
# ============================================================
# 功能:
#   1. 创建 rockfall 系统用户和目录结构
#   2. 安装 Python 虚拟环境和依赖
#   3. 配置 systemd 服务
#   4. 安装定时备份 crontab
#   5. 配置防火墙 (可选)
#
# 用法 (在服务器上以 root 执行):
#   curl -O https://raw.githubusercontent.com/your-org/rockfall-system/main/scripts/setup_server.sh
#   sudo bash setup_server.sh
#
# 或通过 CI/CD 部署后:
#   ssh root@server "bash /opt/rockfall/scripts/setup_server.sh --init-only"
# ============================================================

set -euo pipefail

# ── 默认配置 ────────────────────────────────────────────────
DEPLOY_PATH="${DEPLOY_PATH:-/opt/rockfall}"
SERVICE_USER="${SERVICE_USER:-rockfall}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
DATA_DIR="${DEPLOY_PATH}/data"
BACKUP_DIR="/data/backups"
LOG_DIR="/var/log/rockfall"
VENV_DIR="${DEPLOY_PATH}/.venv"
SERVICE_NAME="rockfall"

# ── 颜色 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
err()  { echo -e "  ${RED}✗${NC} $*" >&2; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }
step() { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

# ── 检查权限 ──────────────────────────────────────────────
check_root() {
    if [[ "$(id -u)" -ne 0 ]]; then
        err "此脚本需要 root 权限执行"
        echo "  请使用: sudo bash $0"
        exit 1
    fi
}

# ── 步骤 1: 创建系统用户 ──────────────────────────────────
create_user() {
    step "步骤 1/7: 创建系统用户"

    if id "$SERVICE_USER" &>/dev/null; then
        ok "用户 $SERVICE_USER 已存在"
    else
        info "创建用户: $SERVICE_USER"
        useradd --system --shell /bin/bash --create-home "$SERVICE_USER"
        ok "用户 $SERVICE_USER 已创建"
    fi

    # 将当前管理员用户加入 rockfall 组 (方便管理)
    local admin_user="${SUDO_USER:-}"
    if [[ -n "$admin_user" ]] && [[ "$admin_user" != "root" ]]; then
        usermod -a -G "$SERVICE_USER" "$admin_user" 2>/dev/null || true
        ok "已将 $admin_user 加入 $SERVICE_USER 组"
    fi
}

# ── 步骤 2: 创建目录结构 ──────────────────────────────────
create_directories() {
    step "步骤 2/7: 创建目录结构"

    local dirs=(
        "$DEPLOY_PATH"
        "$DATA_DIR"
        "$DATA_DIR/results"
        "$DATA_DIR/clips"
        "$DATA_DIR/uploads"
        "$DATA_DIR/masks"
        "$DATA_DIR/quarantine"
        "$BACKUP_DIR"
        "$BACKUP_DIR/pre_deploy"
        "$LOG_DIR"
    )

    for d in "${dirs[@]}"; do
        if [[ -d "$d" ]]; then
            ok "目录已存在: $d"
        else
            mkdir -p "$d"
            ok "目录已创建: $d"
        fi
    done

    # 设置所有权
    info "设置目录所有权: $SERVICE_USER:$SERVICE_USER"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$DEPLOY_PATH"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$BACKUP_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

    # 设置权限
    chmod 755 "$DEPLOY_PATH" "$DATA_DIR" "$BACKUP_DIR" "$LOG_DIR"
    chmod 775 "$DATA_DIR/results" "$DATA_DIR/uploads" "$DATA_DIR/clips"
    chmod 700 "$DATA_DIR/quarantine"  # 隔离目录严格权限
}

# ── 步骤 3: 检查系统依赖 ──────────────────────────────────
check_system_deps() {
    step "步骤 3/7: 检查系统依赖"

    local missing=()

    for cmd in python3 pip3 sqlite3 curl rsync; do
        if command -v "$cmd" &>/dev/null; then
            ok "$cmd: $(command -v "$cmd")"
        else
            err "$cmd: 未安装"
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo ""
        warn "缺少系统依赖: ${missing[*]}"
        info "正在安装..."

        if command -v apt-get &>/dev/null; then
            apt-get update -qq
            apt-get install -y -qq python3 python3-pip python3-venv sqlite3 curl rsync
        elif command -v yum &>/dev/null; then
            yum install -y python3 python3-pip sqlite curl rsync
        else
            err "无法自动安装, 请手动安装: ${missing[*]}"
            return 1
        fi
        ok "系统依赖安装完成"
    fi

    # 检查 Python 版本
    local py_ver
    py_ver=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    info "Python 版本: $py_ver"

    if [[ "${py_ver%.*}" -lt 3 ]] || [[ "${py_ver%.*}" -eq 3 && "${py_ver#*.}" -lt 10 ]]; then
        err "需要 Python >= 3.10, 当前: $py_ver"
        return 1
    fi
    ok "Python 版本满足要求 (>= 3.10)"

    # 检查 ffmpeg (用于视频处理)
    if command -v ffmpeg &>/dev/null; then
        ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
    else
        warn "ffmpeg 未安装 (视频处理需要, 非必须)"
        echo "  安装: apt-get install ffmpeg"
    fi
}

# ── 步骤 4: 创建 Python 虚拟环境 ──────────────────────────
setup_venv() {
    step "步骤 4/7: 创建 Python 虚拟环境"

    if [[ -d "$VENV_DIR" ]]; then
        warn "虚拟环境已存在: $VENV_DIR"
        read -r -p "  是否重新创建? [y/N] " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            ok "保留现有虚拟环境"
            return 0
        fi
    fi

    info "创建虚拟环境: $VENV_DIR"
    sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
    ok "虚拟环境已创建"

    # 升级 pip
    sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    ok "pip 已升级"
}

# ── 步骤 5: 安装 Python 依赖 ──────────────────────────────
install_deps() {
    step "步骤 5/7: 安装 Python 依赖"

    if [[ ! -f "$DEPLOY_PATH/requirements-lock.txt" ]]; then
        warn "requirements-lock.txt 不存在, 尝试从 pyproject.toml 安装"
        if [[ -f "$DEPLOY_PATH/pyproject.toml" ]]; then
            sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --quiet -e "$DEPLOY_PATH"
        else
            err "找不到依赖文件, 请先部署代码到 $DEPLOY_PATH"
            return 1
        fi
    else
        sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --quiet -r "$DEPLOY_PATH/requirements-lock.txt"
    fi

    # 安装 alembic (如果不在依赖中)
    if ! "$VENV_DIR/bin/pip" show alembic &>/dev/null; then
        sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --quiet alembic
    fi

    ok "Python 依赖安装完成"
}

# ── 步骤 6: 配置 systemd 服务 ─────────────────────────────
setup_systemd() {
    step "步骤 6/7: 配置 systemd 服务"

    local service_file="$DEPLOY_PATH/scripts/rockfall.service"
    local target="/etc/systemd/system/${SERVICE_NAME}.service"

    if [[ ! -f "$service_file" ]]; then
        err "服务文件不存在: $service_file"
        return 1
    fi

    # 复制服务文件
    cp "$service_file" "$target"
    ok "服务文件已复制: $target"

    # 重新加载 systemd
    systemctl daemon-reload
    ok "systemd 已重新加载"

    # 启用开机自启
    systemctl enable "$SERVICE_NAME"
    ok "服务已启用开机自启"

    # 启动服务
    info "启动服务..."
    if systemctl start "$SERVICE_NAME"; then
        ok "服务已启动"
    else
        err "服务启动失败, 查看日志: journalctl -u $SERVICE_NAME -n 50"
        return 1
    fi

    # 检查状态
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        ok "服务状态: 运行中"
    else
        warn "服务未运行, 查看日志: journalctl -u $SERVICE_NAME -n 30"
    fi

    systemctl status "$SERVICE_NAME" --no-pager -l 2>&1 | head -15
}

# ── 步骤 7: 配置 cron 定时备份 ────────────────────────────
setup_cron() {
    step "步骤 7/7: 配置定时备份"

    local cron_script="$DEPLOY_PATH/scripts/setup_cron.sh"

    if [[ -f "$cron_script" ]]; then
        chmod +x "$cron_script"
        sudo -u "$SERVICE_USER" bash "$cron_script" install
    else
        warn "setup_cron.sh 不存在, 跳过 crontab 配置"
        echo "  稍后手动执行: bash $DEPLOY_PATH/scripts/setup_cron.sh install"
    fi
}

# ── 完成总结 ──────────────────────────────────────────────
print_summary() {
    echo ""
    echo "=============================================="
    echo -e "  ${GREEN}服务器初始化完成 ✓${NC}"
    echo "=============================================="
    echo ""
    echo "部署路径:   $DEPLOY_PATH"
    echo "服务用户:   $SERVICE_USER"
    echo "虚拟环境:   $VENV_DIR"
    echo "数据目录:   $DATA_DIR"
    echo "备份目录:   $BACKUP_DIR"
    echo "日志目录:   $LOG_DIR"
    echo "服务名称:   $SERVICE_NAME"
    echo ""
    echo "常用命令:"
    echo "  systemctl status $SERVICE_NAME    # 查看服务状态"
    echo "  journalctl -u $SERVICE_NAME -f    # 实时日志"
    echo "  systemctl restart $SERVICE_NAME   # 重启服务"
    echo ""
    echo "健康检查:"
    echo "  curl http://localhost:8000/health"
    echo ""
    echo "手动备份:"
    echo "  sudo -u $SERVICE_USER bash $DEPLOY_PATH/scripts/backup_db.sh"
    echo ""
    echo "回滚:"
    echo "  sudo -u $SERVICE_USER bash $DEPLOY_PATH/scripts/rollback.sh --list-backups"
    echo ""
}

# ── 仅初始化模式 (CI/CD 部署后使用) ──────────────────────
init_only() {
    echo "仅初始化模式 (跳过用户创建和系统依赖安装)"
    setup_venv
    install_deps
    setup_systemd
    print_summary
}

# ── 主入口 ────────────────────────────────────────────────
main() {
    local mode="${1:-full}"

    echo ""
    echo "=============================================="
    echo "  落石监测系统 — 服务器初始化"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
    echo ""

    case "$mode" in
        --init-only)
            init_only
            ;;
        full|*)
            check_root
            create_user
            create_directories
            check_system_deps
            setup_venv
            install_deps
            setup_systemd
            setup_cron
            print_summary
            ;;
    esac
}

main "$@"
