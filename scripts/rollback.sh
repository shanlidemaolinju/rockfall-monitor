#!/bin/bash
# ============================================================
# 落石监测系统 — 一键回滚脚本
# ============================================================
# 功能:
#   1. 恢复上一个版本的 dist/ 目录 (前端)
#   2. 恢复上一个版本的 rockfall/ 目录 (核心库)
#   3. 恢复上一个版本的 server/ 目录 (后端 API)
#   4. 回滚数据库迁移 (alembic downgrade -1)
#   5. 重启服务
#
# 使用前提:
#   - 每次部署时自动创建 .prev 备份 (由 CI/CD 流水线完成)
#   - 或手动执行 backup_current.sh 创建备份点
#
# 用法:
#   ./scripts/rollback.sh                    # 交互式回滚 (需要确认)
#   ./scripts/rollback.sh --force            # 强制回滚 (跳过确认)
#   ./scripts/rollback.sh --dry-run          # 仅显示将回滚的内容
#   ./scripts/rollback.sh --list-backups     # 列出可用备份
#   ./scripts/rollback.sh --backup-now       # 手动创建当前版本备份
#   ./scripts/rollback.sh --restore YYYY-MM-DD  # 从指定日期备份恢复
#
# 部署路径 (可通过环境变量覆盖):
#   DEPLOY_PATH=/opt/rockfall  ./scripts/rollback.sh
# ============================================================

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_PATH="${DEPLOY_PATH:-$PROJECT_ROOT}"
BACKUP_DIR="${DEPLOY_PATH}/.rollback_snapshots"
SERVICE_NAME="${SERVICE_NAME:-rockfall}"

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

# ── 帮助信息 ──────────────────────────────────────────────
show_help() {
    cat << 'EOF'
落石监测系统 — 一键回滚脚本
============================

用法:
  rollback.sh [选项]

选项:
  --force           强制回滚，不询问确认
  --dry-run         仅显示将要执行的操作，不实际修改文件
  --list-backups    列出所有可用备份
  --backup-now      立即创建当前版本的备份快照
  --restore DATE    从指定日期的备份恢复 (格式: YYYY-MM-DD)
  -h, --help        显示此帮助

环境变量:
  DEPLOY_PATH       部署路径 (默认: 项目根目录)
  SERVICE_NAME      系统服务名 (默认: rockfall)

示例:
  # 查看可用备份
  ./scripts/rollback.sh --list-backups

  # 创建当前版本的备份
  ./scripts/rollback.sh --backup-now

  # 干运行 — 查看回滚将执行什么
  ./scripts/rollback.sh --dry-run

  # 交互式回滚到上一个版本
  ./scripts/rollback.sh

  # 强制回滚
  ./scripts/rollback.sh --force

  # 恢复到指定日期的备份
  ./scripts/rollback.sh --restore 2026-06-10
EOF
}

# ── 检查是否为 root ───────────────────────────────────────
check_sudo() {
    if ! command -v sudo &>/dev/null; then
        return
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        return  # 已是 root
    fi
    # 检查是否可 sudo (不需要密码)
    if sudo -n true 2>/dev/null; then
        return
    fi
    warn "服务管理需要 sudo 权限"
}

# ── 备份当前版本 ─────────────────────────────────────────
backup_current() {
    local label="${1:-manual}"
    local timestamp
    timestamp="$(date '+%Y-%m-%d_%H%M%S')"
    local snapshot_dir="$BACKUP_DIR/${timestamp}_${label}"

    echo ""
    info "创建当前版本快照: $snapshot_dir"
    mkdir -p "$snapshot_dir"

    local backup_count=0

    # 备份目录 (使用 cp -a 保留权限和时间戳)
    for component in dist rockfall server; do
        local src="$DEPLOY_PATH/$component"
        if [[ -d "$src" ]]; then
            cp -a "$src" "$snapshot_dir/$component"
            ok "备份: $component/"
            backup_count=$((backup_count + 1))
        else
            warn "跳过 (目录不存在): $component/"
        fi
    done

    # 备份 alembic 相关
    for item in alembic alembic.ini; do
        local src="$DEPLOY_PATH/$item"
        if [[ -d "$src" || -f "$src" ]]; then
            cp -a "$src" "$snapshot_dir/$item"
            ok "备份: $item"
            backup_count=$((backup_count + 1))
        fi
    done

    # 部署信息
    {
        echo "backup_time=$(date -Iseconds)"
        echo "label=$label"
        echo "deploy_path=$DEPLOY_PATH"
        echo "hostname=$(hostname)"
        echo "components_backed_up=$backup_count"
        if [[ -f "$DEPLOY_PATH/.deploy_backup_info" ]]; then
            echo "--- deploy_backup_info ---"
            cat "$DEPLOY_PATH/.deploy_backup_info"
        fi
    } > "$snapshot_dir/backup_info.txt"

    ok "快照创建完成: $snapshot_dir ($backup_count 个组件)"
    echo ""

    # 清理旧快照 (保留最近 10 个)
    _cleanup_snapshots 10
}

# ── 清理过期快照 ─────────────────────────────────────────
_cleanup_snapshots() {
    local keep="${1:-10}"
    local total
    total=$(find "$BACKUP_DIR" -maxdepth 1 -type d -not -path "$BACKUP_DIR" 2>/dev/null | wc -l)

    if [[ $total -le $keep ]]; then
        return
    fi

    local to_delete=$((total - keep))
    info "清理 $to_delete 个旧快照 (保留最近 $keep 个)..."

    find "$BACKUP_DIR" -maxdepth 1 -type d -not -path "$BACKUP_DIR" \
        | sort | head -n "$to_delete" | while read -r d; do
        rm -rf "$d"
        ok "已删除: $(basename "$d")"
    done
}

# ── 列出可用备份 ─────────────────────────────────────────
list_backups() {
    echo ""
    echo "可用备份:"
    echo "=========="

    # 1. 快照备份 (手动创建的)
    if [[ -d "$BACKUP_DIR" ]] && [[ -n "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]]; then
        echo ""
        echo -e "${CYAN}[快照备份] ${BACKUP_DIR}/${NC}"
        find "$BACKUP_DIR" -maxdepth 1 -type d -not -path "$BACKUP_DIR" \
            | sort -r | while read -r d; do
            local info_file="$d/backup_info.txt"
            local time_str=""
            if [[ -f "$info_file" ]]; then
                time_str=$(grep "^backup_time=" "$info_file" | cut -d= -f2- || echo "?")
            fi
            echo -e "  ${GREEN}$(basename "$d")${NC}  $time_str"
        done
    else
        warn "无快照备份 (使用 --backup-now 创建)"
    fi

    # 2. .prev 自动备份 (CI/CD 创建)
    echo ""
    echo -e "${CYAN}[自动备份] (.prev 目录)${NC}"
    for component in dist rockfall server alembic; do
        if [[ -d "$DEPLOY_PATH/$component.prev" ]]; then
            local size
            size=$(du -sh "$DEPLOY_PATH/$component.prev" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}$component.prev${NC}  $size"
        fi
    done

    # 3. 部署信息
    if [[ -f "$DEPLOY_PATH/.deploy_backup_info" ]]; then
        echo ""
        echo -e "${CYAN}[上次部署信息]${NC}"
        cat "$DEPLOY_PATH/.deploy_backup_info" | while read -r line; do
            echo "  $line"
        done
    fi

    echo ""
}

# ── 从 .prev 回滚 (快速回滚) ─────────────────────────────
rollback_from_prev() {
    local force="$1"

    echo ""
    echo "=============================================="
    echo "  回滚到上一个版本 (使用 .prev 备份)"
    echo "=============================================="
    echo ""

    local rollback_components=()

    for component in dist rockfall server; do
        local prev_dir="$DEPLOY_PATH/$component.prev"
        local current_dir="$DEPLOY_PATH/$component"

        if [[ ! -d "$prev_dir" ]]; then
            warn "没有 $component.prev 备份，跳过 $component 回滚"
            continue
        fi
        rollback_components+=("$component")
    done

    if [[ ${#rollback_components[@]} -eq 0 ]]; then
        err "没有可用的 .prev 备份文件"
        err "请先部署一次以创建 .prev 备份，或使用 --backup-now 创建快照"
        return 1
    fi

    echo "将回滚以下组件:"
    for c in "${rollback_components[@]}"; do
        echo "  - $c/  ($DEPLOY_PATH/$c.prev → $DEPLOY_PATH/$c)"
    done
    echo ""

    # 确认
    if [[ "$force" != "true" ]]; then
        read -r -p "确认回滚? [y/N] " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            warn "已取消回滚"
            return 0
        fi
    fi

    # 执行回滚
    for component in "${rollback_components[@]}"; do
        local prev_dir="$DEPLOY_PATH/$component.prev"
        local current_dir="$DEPLOY_PATH/$component"

        info "回滚 $component/ ..."
        rm -rf "$current_dir"
        cp -a "$prev_dir" "$current_dir"
        rm -rf "$prev_dir"  # 回滚后清除 .prev 防止重复回滚
        ok "回滚完成: $component/"
    done

    # 可选的数据库回滚
    if [[ -d "$DEPLOY_PATH/alembic" ]]; then
        echo ""
        warn "数据库迁移回滚需要手动执行:"
        echo "  cd $DEPLOY_PATH && python -m alembic downgrade -1"
        echo ""
        if [[ "$force" != "true" ]]; then
            read -r -p "是否同时回滚数据库迁移? [y/N] " db_response
            if [[ "$db_response" =~ ^[Yy]$ ]]; then
                _downgrade_database
            fi
        fi
    fi

    # 重启服务
    _restart_service

    ok "回滚完成 ✓"
}

# ── 从快照恢复 ───────────────────────────────────────────
rollback_from_snapshot() {
    local snapshot_date="$1"
    local force="$2"

    # 查找匹配的快照
    local snapshot_dir
    snapshot_dir=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "${snapshot_date}*" 2>/dev/null | sort | tail -1)

    if [[ -z "$snapshot_dir" ]]; then
        err "未找到匹配的快照: $snapshot_date"
        echo ""
        echo "可用快照:"
        find "$BACKUP_DIR" -maxdepth 1 -type d -not -path "$BACKUP_DIR" \
            | sort -r | while read -r d; do
            echo "  $(basename "$d")"
        done
        return 1
    fi

    echo ""
    echo "=============================================="
    echo "  恢复到快照: $(basename "$snapshot_dir")"
    echo "=============================================="
    echo ""

    # 先备份当前版本
    backup_current "pre_restore_$(basename "$snapshot_dir")"

    # 确认
    if [[ "$force" != "true" ]]; then
        read -r -p "确认恢复? 当前版本已备份 [y/N] " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            warn "已取消恢复 (当前版本已保存)"
            return 0
        fi
    fi

    # 恢复
    local restored=0
    for component in dist rockfall server; do
        local src="$snapshot_dir/$component"
        local dst="$DEPLOY_PATH/$component"
        if [[ -d "$src" ]]; then
            rm -rf "$dst"
            cp -a "$src" "$dst"
            ok "恢复: $component/"
            restored=$((restored + 1))
        fi
    done

    if [[ $restored -eq 0 ]]; then
        err "快照中没有可恢复的组件"
        return 1
    fi

    _restart_service
    ok "恢复完成 ✓ ($restored 个组件)"
}

# ── 数据库降级 ───────────────────────────────────────────
_downgrade_database() {
    info "执行数据库降级 (downgrade -1) ..."

    if ! command -v python &>/dev/null; then
        err "python 不可用"
        return 1
    fi

    cd "$DEPLOY_PATH"
    if python -m alembic downgrade -1 2>&1; then
        ok "数据库降级成功"
    else
        err "数据库降级失败 (可能需要手动处理)"
        return 1
    fi
}

# ── 服务重启 ─────────────────────────────────────────────
_restart_service() {
    echo ""
    info "重启服务: $SERVICE_NAME"

    # 检查 systemd
    if command -v systemctl &>/dev/null && systemctl is-enabled "$SERVICE_NAME" &>/dev/null; then
        if sudo systemctl restart "$SERVICE_NAME" 2>&1; then
            ok "服务已重启"
            sudo systemctl status "$SERVICE_NAME" --no-pager -l 2>&1 | head -10
        else
            err "服务重启失败"
        fi
    elif [[ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]]; then
        sudo systemctl daemon-reload
        sudo systemctl restart "$SERVICE_NAME"
        ok "服务已重启"
    else
        warn "未检测到 systemd 服务 '$SERVICE_NAME', 请手动重启"
        echo "  sudo systemctl restart $SERVICE_NAME"
        echo "  或"
        echo "  cd $DEPLOY_PATH && uvicorn server.main:app --host 0.0.0.0 --port 8000 &"
    fi
}

# ── 主入口 ────────────────────────────────────────────────
main() {
    local force=false
    local dry_run=false
    local action="rollback"

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force)
                force=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --list-backups)
                list_backups
                return 0
                ;;
            --backup-now)
                backup_current "manual_$(date '+%H%M%S')"
                return 0
                ;;
            --restore)
                action="restore"
                SNAPSHOT_DATE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                return 0
                ;;
            *)
                err "未知参数: $1"
                show_help
                return 1
                ;;
        esac
    done

    # 干运行
    if $dry_run; then
        echo ""
        info "[DRY-RUN 模式] 将执行以下操作:"
        info "  1. 停止服务: sudo systemctl stop $SERVICE_NAME"
        info "  2. 恢复 .prev 备份到当前目录"
        info "  3. 启动服务: sudo systemctl start $SERVICE_NAME"
        echo ""
        list_backups
        return 0
    fi

    check_sudo

    case "$action" in
        rollback)
            rollback_from_prev "$force"
            ;;
        restore)
            rollback_from_snapshot "${SNAPSHOT_DATE:-}" "$force"
            ;;
    esac
}

main "$@"
