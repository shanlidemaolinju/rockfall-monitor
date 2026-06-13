# Changelog

## [2.2.0] — 2026-06-14

### 修复 (17 项代码审查问题)

**阶段 1 — 基础一致性**
- 统一版本号：`rockfall/__init__.py` 定义 `__version__ = "2.2.0"`，5 个引用文件全部从单一来源读取
- 删除 4 个废弃 requirements 文件，仅保留 `requirements.in` + `requirements-lock.txt`
- 修复 Python 版本不匹配：CI 与 lock 文件统一为 Python 3.11
- `pyproject.toml` 补全 `sentry-sdk[fastapi]>=2.0.0`
- 解决 opencv-python / opencv-python-headless 冲突：新增 `constraints.txt`

**阶段 2 — 依赖与 CI**
- `requirements.in` 添加 `cryptography>=42.0.0`，lock 文件重新生成
- 修复 CI rsync 规则：移除冗余 `--exclude='alembic/'`
- 删除误提交的 `web/bundle-analysis.yml` (240KB)
- 修复 `config.py` 中 `__import__('threading')` 反模式 → 正常 `import threading`
- 消除 pymysql 可用性检测重复：新建 `rockfall/db_utils.py` (`is_mysql_available()` + `get_pymysql()`)

**阶段 3 — 代码整洁**
- 19 张调试图片移入 `data/debug/`
- CI 新增 pyright 类型检查步骤
- 删除 `server/main.py` 未使用的 `import os as _os` 和 `_Path` 别名
- 确认 `.env` 未被 Git 追踪
- `download_sam.py` 移入 `scripts/`
- 根目录 `deploy.md` 改为跳转链接，消除文档重复

**阶段 4 — 验收与文档**
- 回归测试：303 个用例成功收集
- README 更新：正确安装指令、目录结构、开发指南
- `__all__` 补齐 v2.2+ 模块导出（auth/sentry/db_utils/performance 等）

### 新增
- `rockfall/db_utils.py` — 共享数据库工具
- `constraints.txt` — pip 约束文件
- `CHANGELOG.md` — 本文档
- 开发指南（版本管理 + 依赖流程）

### 修改
- `pyproject.toml`：版本 2.1.0 → 2.2.0，添加 sentry-sdk
- `requirements.in`：添加 cryptography，opencv 冲突说明
- `requirements-lock.txt`：Python 3.11 重新生成
- `rockfall/__init__.py`：新增 `__version__`，更新 `__all__`
- `rockfall/config.py`：`import threading` 规范化
- `rockfall/sentry_init.py`：版本号动态读取
- `server/main.py`：版本号动态读取，清理无用导入
- `rockfall/metrics.py`：默认版本从 `__version__` 读取
- `app.py`：版本号动态读取
- `rockfall/site_config.py`：MySQL 配置导入提升到模块顶部
- `rockfall/alert_store.py`：使用 `db_utils` 共享函数
- `.gitignore`：新增 `data/debug/`
- `deploy.md`：改为文档索引跳转页
- `README.md`：更新安装指令、目录结构、配置默认值

### 删除
- `requirements.txt`、`requirements-base.txt`、`requirements-gpu.txt`、`requirements-dev.txt`
- `web/bundle-analysis.yml`
