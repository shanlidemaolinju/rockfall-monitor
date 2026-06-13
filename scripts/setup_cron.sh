#!/bin/bash
# ============================================================
# 落石监测系统 — Crontab 自动配置脚本
# ============================================================
# 功能:
#   1. 将 backup_db.sh 注册到 crontab (每天凌晨 2:00)
#   2. 将日志清理任务注册到 crontab (每周日 3:00)
#   3. 支持安装、卸载、查看状态
#
# 用法:
#   ./scripts/setup_cron.sh install     # 安装 crontab 任务
#   ./scripts/setup_cron.sh uninstall   # 移除 crontab 任务
#   ./scripts/setup_cron.sh status      # 查看当前 crontab 状态
#   ./scripts/setup_cron.sh --help      # 显示帮助
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_db.sh"
LOG_DIR="/var/log/rockfall"
CRON_MARKER="# rockfall-system"

# ── 颜色 ──────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── 确保脚本可执行 ──────────────────────────────────────
chmod +x "$BACKUP_SCRIPT" 2>/dev/null || true

# ── 准备日志目录 ────────────────────────────────────────
prepare() {
    sudo mkdir -p "$LOG_DIR"
    # 将日志目录所有权赋予当前用户，确保 cron 任务可以写入
    local current_user
    current_user="$(whoami)"
    sudo chown "$current_user:$(id -gn "$current_user")" "$LOG_DIR" 2>/dev/null || true
    sudo chmod 755 "$LOG_DIR"
    echo -e "${GREEN}✓${NC} 日志目录就绪: $LOG_DIR (owner: $current_user)"
}

# ── 安装 crontab ────────────────────────────────────────
install_cron() {
    echo ""
    echo -e "${CYAN}==============================================${NC}"
    echo -e "${CYAN}  安装 crontab 定时任务${NC}"
    echo -e "${CYAN}==============================================${NC}"
    echo ""

    prepare

    # 检查备份脚本
    if [[ ! -f "$BACKUP_SCRIPT" ]]; then
        echo -e "${RED}✗${NC} 备份脚本不存在: $BACKUP_SCRIPT"
        exit 1
    fi

    # 先移除已有任务 (避免重复)
    uninstall_cron_silent

    # 构建新的 crontab 条目
    local new_entries=(
        "# ${CRON_MARKER} — 数据库定时备份 (每天凌晨 2:00)"
        "0 2 * * * ${BACKUP_SCRIPT} >> ${LOG_DIR}/backup.log 2>&1"
        ""
        "# ${CRON_MARKER} — 每周日清理检测日志 (3:00)"
        "0 3 * * 0 find ${PROJECT_ROOT}/data -name 'detection_log.*.jsonl' -mtime +30 -delete 2>/dev/null"
        ""
        "# ${CRON_MARKER} — 每周日清理过期结果文件 (3:30)"
        "30 3 * * 0 find ${PROJECT_ROOT}/data/results -type f -mtime +90 -delete 2>/dev/null"
    )

    # 读取现有 crontab
    local current_cron
    current_cron=$(crontab -l 2>/dev/null || echo "")

    # 合并
    local new_cron
    if [[ -z "$current_cron" ]] || [[ "$current_cron" =~ ^[[:space:]]*$ ]]; then
        new_cron=$(printf '%s\n' "${new_entries[@]}")
    else
        # 确保以换行结尾
        if [[ ! "$current_cron" =~ $'\n'$ ]]; then
            current_cron="${current_cron}"$'\n'
        fi
        new_cron="${current_cron}"$'\n'$(printf '%s\n' "${new_entries[@]}")
    fi

    # 安装
    echo "$new_cron" | crontab -

    echo ""
    echo -e "${GREEN}✓${NC} crontab 安装完成"
    echo ""
    echo "已注册任务:"
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │ 每天 02:00  数据库备份 → ${LOG_DIR}/backup.log     │"
    echo "  │ 每周日 03:00  清理过期检测日志 (30天)               │"
    echo "  │ 每周日 03:30  清理过期结果文件 (90天)               │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo ""

    show_status
}

# ── 静默卸载 ────────────────────────────────────────────
uninstall_cron_silent() {
    local current_cron
    current_cron=$(crontab -l 2>/dev/null || echo "")
    if [[ -z "$current_cron" ]]; then
        return
    fi

    # 移除包含 rockfall-system 标记的行
    local filtered
    filtered=$(echo "$current_cron" | grep -v "$CRON_MARKER" || echo "")

    # 移除孤立空行
    filtered=$(echo "$filtered" | sed '/^$/N;/^\n$/D' || echo "")

    if [[ -z "$(echo "$filtered" | tr -d '[:space:]')" ]]; then
        # 全部删完 → 清空 crontab
        crontab -r 2>/dev/null || true
    else
        echo "$filtered" | crontab -
    fi
}

# ── 卸载 crontab ────────────────────────────────────────
uninstall_cron() {
    echo ""
    echo -e "${CYAN}卸载 crontab 任务...${NC}"

    local current_cron
    current_cron=$(crontab -l 2>/dev/null || echo "")

    if [[ -z "$current_cron" ]] || ! echo "$current_cron" | grep -q "$CRON_MARKER"; then
        echo -e "${YELLOW}⚠${NC} 未找到 rockfall-system 相关 crontab 任务"
        return 0
    fi

    uninstall_cron_silent
    echo -e "${GREEN}✓${NC} crontab 任务已移除"
}

# ── 查看状态 ────────────────────────────────────────────
show_status() {
    echo ""
    echo -e "${CYAN}当前 crontab 状态:${NC}"
    echo "  ─────────────────────────────────────────────"

    local current_cron
    current_cron=$(crontab -l 2>/dev/null || echo "")

    if [[ -z "$current_cron" ]] || [[ -z "$(echo "$current_cron" | tr -d '[:space:]')" ]]; then
        echo "  (空 — 没有定时任务)"
        return
    fi

    echo "$current_cron" | while IFS= read -r line; do
        if echo "$line" | grep -q "$CRON_MARKER"; then
            echo -e "  ${GREEN}$line${NC}"
        elif [[ -n "${line:-}" ]]; then
            echo "  $line"
        fi
    done
    echo ""
}

# ── 手动测试备份 ────────────────────────────────────────
test_backup() {
    echo ""
    echo -e "${CYAN}手动执行一次备份 (测试)...${NC}"
    echo ""

    if [[ ! -f "$BACKUP_SCRIPT" ]]; then
        echo -e "${RED}✗${NC} 备份脚本不存在: $BACKUP_SCRIPT"
        exit 1
    fi

    bash "$BACKUP_SCRIPT" --dry-run
    echo ""

    read -r -p "确认执行实际备份? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        bash "$BACKUP_SCRIPT"
    fi
}

# ── 帮助 ────────────────────────────────────────────────
show_help() {
    cat << EOF
用法: $0 [命令]

命令:
  install    安装 crontab 定时任务 (数据库备份 + 日志清理)
  uninstall  移除 crontab 任务
  status     查看当前 crontab 状态
  test       手动测试一次备份 (--dry-run + 确认后实际执行)
  help       显示此帮助

安装后的 crontab 任务:
  每天 02:00  数据库备份到 /data/backups (保留 30 天)
  每周日 03:00  清理超过 30 天的检测日志
  每周日 03:30  清理超过 90 天的结果文件

日志文件:
  /var/log/rockfall/backup.log — 备份执行日志

手动备份:
  $SCRIPT_DIR/backup_db.sh

手动回滚:
  $SCRIPT_DIR/rollback.sh
EOF
}

# ── 主入口 ────────────────────────────────────────────────
case "${1:-help}" in
    install)
        install_cron
        ;;
    uninstall|remove)
        uninstall_cron
        ;;
    status|show)
        show_status
        ;;
    test)
        test_backup
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo -e "${RED}未知命令: ${1:-}${NC}"
        show_help
        exit 1
        ;;
esac
