#!/bin/bash
# ============================================================
# 落石监测系统 — 数据库定时备份脚本
# ============================================================
# 功能:
#   1. 备份 data/ 下所有 SQLite 数据库到 /data/backups
#   2. 保留近 30 天的备份，自动清理过期文件
#   3. 备份前后执行完整性检查 (sqlite3 integrity_check)
#   4. 支持自定义保留天数和备份目录
#
# 用法:
#   ./scripts/backup_db.sh                    # 使用默认配置
#   ./scripts/backup_db.sh --retain 60        # 保留 60 天
#   ./scripts/backup_db.sh --backup-dir /mnt/nas/backups  # 指定备份目录
#   ./scripts/backup_db.sh --dry-run          # 仅显示将要执行的操作
#
# Crontab 配置 (每天凌晨 2:00 执行):
#   0 2 * * * /opt/rockfall/scripts/backup_db.sh >> /var/log/rockfall/backup.log 2>&1
# ============================================================

set -euo pipefail

# ── 默认配置 ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
BACKUP_ROOT="/data/backups"
RETENTION_DAYS=30
DRY_RUN=false
VERIFY_BACKUP=false
LOG_FILE=""  # 为空时输出到 stdout

# ── 颜色 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()   { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
ok()    { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}✓${NC} $*"; }
err()   { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}✗${NC} $*" >&2; }
warn()  { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}⚠${NC} $*"; }
info()  { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${CYAN}→${NC} $*"; }

# ── 参数解析 ────────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --retain)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --backup-dir)
                BACKUP_ROOT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verify)
                VERIFY_BACKUP=true
                shift
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --retain N       保留天数 (默认: 30)"
                echo "  --backup-dir DIR 备份目录 (默认: /data/backups)"
                echo "  --dry-run        仅显示将要执行的操作"
                echo "  --verify         验证最近一次备份可恢复性"
                echo "  --log-file FILE  日志文件路径"
                echo "  -h, --help       显示帮助"
                exit 0
                ;;
            *)
                err "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# ── 日志重定向 ──────────────────────────────────────────────
redirect_log() {
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        exec >> "$LOG_FILE" 2>&1
    fi
}

# ── 备份目录准备 ────────────────────────────────────────────
prepare_backup_dir() {
    local today_dir="$BACKUP_ROOT/$(date '+%Y-%m-%d')"

    if $DRY_RUN; then
        info "[DRY-RUN] 将创建备份目录: $today_dir"
        return
    fi

    mkdir -p "$today_dir"
    echo "$today_dir"
}

# ── 数据库完整性检查 ────────────────────────────────────────
check_integrity() {
    local db_file="$1"

    if [[ ! -f "$db_file" ]]; then
        return 1  # 文件不存在, 跳过
    fi

    local result
    result=$(sqlite3 "$db_file" "PRAGMA integrity_check;" 2>&1)

    if [[ "$result" == "ok" ]]; then
        return 0
    else
        err "数据库完整性检查失败: $db_file → $result"
        return 1
    fi
}

# ── VACUUM 数据库 ────────────────────────────────────────────
vacuum_db() {
    local db_file="$1"

    if $DRY_RUN; then
        info "[DRY-RUN] 将 VACUUM: $db_file"
        return 0
    fi

    # VACUUM 会重建数据库文件, 释放已删除数据占用的空间
    # 同时也复制到临时文件, 是一种安全的操作
    if sqlite3 "$db_file" "VACUUM;" 2>&1; then
        return 0
    else
        warn "VACUUM 失败 (非致命): $db_file"
        return 1
    fi
}

# ── 执行备份 ────────────────────────────────────────────────
do_backup() {
    local today_dir="$1"
    local total_size=0
    local success_count=0
    local fail_count=0

    info "开始备份 data/ 目录下的 SQLite 数据库..."
    info "目标: $today_dir"

    # 查找所有 SQLite 数据库文件
    # 排除 WAL/SHM 临时文件, 只备份主数据库文件
    local db_files
    db_files=$(find "$DATA_DIR" -maxdepth 1 -name "*.db" -not -name "*-wal" -not -name "*-shm" 2>/dev/null || true)

    if [[ -z "$db_files" ]]; then
        warn "未找到 SQLite 数据库文件 ($DATA_DIR/*.db)"
        return 0
    fi

    for db_file in $db_files; do
        local db_name
        db_name="$(basename "$db_file")"

        info "处理: $db_name"

        # 1. 完整性检查 (备份前)
        if ! check_integrity "$db_file"; then
            err "跳过损坏的数据库: $db_name"
            fail_count=$((fail_count + 1))
            continue
        fi
        ok "  完整性检查通过: $db_name"

        # 2. VACUUM (可选, 优化存储)
        if [[ "${SKIP_VACUUM:-false}" != "true" ]]; then
            vacuum_db "$db_file"
        fi

        # 3. 使用 sqlite3 .backup 命令安全备份 (在线备份, 不阻塞读写)
        local backup_file="$today_dir/$db_name"

        if $DRY_RUN; then
            info "[DRY-RUN] 将备份: $db_file → $backup_file"
            success_count=$((success_count + 1))
            continue
        fi

        if sqlite3 "$db_file" ".backup '$backup_file'" 2>&1; then
            local size
            size=$(stat -c%s "$backup_file" 2>/dev/null || stat -f%z "$backup_file" 2>/dev/null || echo 0)
            total_size=$((total_size + size))
            ok "  备份成功: $db_name ($(echo "scale=1; $size/1024" | bc -l 2>/dev/null || echo "?") KB)"
            success_count=$((success_count + 1))
        else
            err "  备份失败: $db_name"
            fail_count=$((fail_count + 1))
        fi
    done

    # 汇总
    if $DRY_RUN; then
        info "[DRY-RUN] 将备份 $success_count 个数据库"
        return 0
    fi

    if [[ $total_size -gt 0 ]]; then
        local total_kb
        total_kb=$(echo "scale=1; $total_size/1024" | bc -l 2>/dev/null || echo "?")
        ok "备份完成: ${success_count} 成功, ${fail_count} 失败, 总计 ${total_kb} KB"
    fi

    # 记录备份元数据
    if [[ ! $DRY_RUN ]]; then
        {
            echo "backup_time=$(date -Iseconds)"
            echo "project_root=$PROJECT_ROOT"
            echo "success_count=$success_count"
            echo "fail_count=$fail_count"
            echo "total_size_bytes=$total_size"
            echo "retention_days=$RETENTION_DAYS"
            echo "databases=$(echo "$db_files" | tr '\n' ',' | sed 's/,$//')"
        } > "$today_dir/backup_meta.txt"
    fi

    return $fail_count
}

# ── 清理过期备份 ────────────────────────────────────────────
cleanup_old_backups() {
    info "清理超过 ${RETENTION_DAYS} 天的旧备份..."

    if $DRY_RUN; then
        local old_dirs
        old_dirs=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name "????-??-??" \
            -mtime "+${RETENTION_DAYS}" 2>/dev/null || true)
        if [[ -n "$old_dirs" ]]; then
            echo "$old_dirs" | while read -r d; do
                info "[DRY-RUN] 将删除: $d"
            done
        else
            info "[DRY-RUN] 无过期备份需要清理"
        fi
        return 0
    fi

    local deleted_count=0
    local deleted_size=0

    # 使用 find 查找超过指定天数的备份目录
    local old_dirs
    old_dirs=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name "????-??-??" \
        -mtime "+${RETENTION_DAYS}" 2>/dev/null || true)

    if [[ -z "$old_dirs" ]]; then
        ok "无过期备份需要清理"
        return 0
    fi

    for d in $old_dirs; do
        local dir_size
        dir_size=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
        deleted_size=$((deleted_size + dir_size))

        info "删除过期备份: $(basename "$d")"
        rm -rf "$d"
        deleted_count=$((deleted_count + 1))
    done

    local deleted_mb
    deleted_mb=$(echo "scale=1; $deleted_size/1048576" | bc -l 2>/dev/null || echo "?")
    ok "清理完成: 删除 ${deleted_count} 个过期备份, 释放 ${deleted_mb} MB"
}

# ── 磁盘使用统计 ────────────────────────────────────────────
show_disk_usage() {
    if $DRY_RUN; then
        return
    fi

    info "备份目录磁盘使用:"
    if [[ -d "$BACKUP_ROOT" ]]; then
        du -sh "$BACKUP_ROOT" 2>/dev/null | while read -r line; do
            echo "  $line"
        done

        # 各子目录概览 (最近 5 天)
        echo "  最近备份:"
        find "$BACKUP_ROOT" -maxdepth 1 -type d -name "????-??-??" \
            | sort -r | head -5 | while read -r d; do
            local dir_size
            dir_size=$(du -sh "$d" 2>/dev/null | cut -f1)
            echo "    $(basename "$d")  $dir_size"
        done
    fi
}

# ── 记录完成标记 (用于监控) ──────────────────────────────────
write_checkpoint() {
    local status="${1:-ok}"
    if $DRY_RUN; then
        return
    fi
    local checkpoint_file="$BACKUP_ROOT/.last_backup"
    echo "time=$(date -Iseconds)" > "$checkpoint_file"
    echo "status=$status" >> "$checkpoint_file"
    echo "retention=$RETENTION_DAYS" >> "$checkpoint_file"
}

# ── 验证备份可恢复性 ──────────────────────────────────────────
verify_latest_backup() {
    info "验证最近一次备份的可恢复性..."

    # 找到最近的备份目录
    local latest
    latest=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name "????-??-??" 2>/dev/null | sort -r | head -1)

    if [[ -z "$latest" ]]; then
        err "没有找到备份目录 ($BACKUP_ROOT)"
        return 1
    fi

    info "最近备份: $(basename "$latest")"
    local verified=0
    local failed=0

    for backup_file in "$latest"/*.db; do
        [[ -f "$backup_file" ]] || continue
        local db_name
        db_name="$(basename "$backup_file")"

        # 使用 sqlite3 打开备份文件并检查完整性
        local result
        result=$(sqlite3 "$backup_file" "PRAGMA integrity_check;" 2>&1)

        if [[ "$result" == "ok" ]]; then
            # 进一步验证：检查关键表是否存在
            local table_count
            table_count=$(sqlite3 "$backup_file" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>&1)
            ok "  验证通过: $db_name (${table_count} 张表)"
            verified=$((verified + 1))
        else
            err "  验证失败: $db_name → $result"
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -eq 0 ]]; then
        ok "备份可恢复性验证通过 ✓ ($verified 个数据库)"
        return 0
    else
        err "备份可恢复性验证失败: $failed 个数据库损坏"
        return 1
    fi
}

# ── 主流程 ──────────────────────────────────────────────────
main() {
    parse_args "$@"
    redirect_log

    log "=============================================="
    log "落石监测系统 — 数据库定时备份"
    log "=============================================="
    log "项目目录: $PROJECT_ROOT"
    log "数据目录: $DATA_DIR"
    log "备份目录: $BACKUP_ROOT"
    log "保留天数: $RETENTION_DAYS 天"
    log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "=============================================="

    # 检查 sqlite3 是否可用
    if ! command -v sqlite3 &>/dev/null; then
        err "sqlite3 命令不可用, 请安装 sqlite3"
        write_checkpoint "error: sqlite3 not found"
        exit 1
    fi

    # 检查数据目录
    if [[ ! -d "$DATA_DIR" ]]; then
        err "数据目录不存在: $DATA_DIR"
        write_checkpoint "error: DATA_DIR not found"
        exit 1
    fi

    # 准备备份目录
    local today_dir
    today_dir=$(prepare_backup_dir)

    # 执行备份
    local fail_count=0
    if ! do_backup "$today_dir"; then
        fail_count=$?
        warn "备份过程有 ${fail_count} 个失败 (将继续执行)"
    fi

    # 清理过期备份
    cleanup_old_backups

    # 验证备份 (如果指定了 --verify)
    if $VERIFY_BACKUP; then
        verify_latest_backup || warn "备份验证有失败项 (备份文件本身可能仍可用)"
    fi

    # 磁盘统计
    show_disk_usage

    # 完成
    if [[ $fail_count -eq 0 ]]; then
        ok "备份任务完成 ✓"
        write_checkpoint "ok"
    else
        warn "备份任务完成 (有 ${fail_count} 个警告)"
        write_checkpoint "partial: ${fail_count} failures"
    fi

    log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

    return $fail_count
}

main "$@"
