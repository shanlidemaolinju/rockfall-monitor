"""
共享测试 fixtures — pytest 插件机制自动发现
===========================================
提供:
  - tmp_data_dir      函数级临时 DATA_DIR (隔离测试数据)
  - mysql_container   会话级 Docker MySQL 容器 (需要 Docker 环境)
  - mysql_backend     函数级 MySQL AlertStore + SiteStore
  - sqlite_store      函数级 SQLite AlertStore (独立数据库)
  - sqlite_site_store 函数级 SQLite SiteStore
  - client            FastAPI TestClient
  - mock_detector     函数级 Mock RockDetector (不加载真实 YOLO 模型)
  - clean_config      自动重置 rockfall.config 模块级常量

运行:
    # 仅 SQLite 测试 (无需 Docker, 快速)
    pytest tests/ -v --ignore-glob='*mysql*' -k "not mysql"

    # 全部测试 (需要 Docker)
    pytest tests/ -v

    # 仅 MySQL 测试
    pytest tests/ -v -k "mysql"
"""

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# 确保项目根目录可导入
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


# ================================================================
# 工具函数
# ================================================================

def _find_free_port() -> int:
    """在 localhost 上查找空闲 TCP 端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_available() -> bool:
    """检测 Docker 是否可用"""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# 会话级 Docker 可用性标记 (只检测一次)
_DOCKER_OK = None


def is_docker_ok() -> bool:
    global _DOCKER_OK
    if _DOCKER_OK is None:
        _DOCKER_OK = _docker_available()
    return _DOCKER_OK


# ================================================================
# 会话级: Docker MySQL 容器
# ================================================================

@pytest.fixture(scope="session")
def mysql_container():
    """
    会话级 Docker MySQL 8.0 容器。

    自动探测空闲端口 → 启动容器 → 等待就绪 → 返回连接参数。
    全部测试结束后停止并删除容器。

    如果 Docker 不可用, 返回 None (MySQL 测试应检查并 skip)。
    """
    if not is_docker_ok():
        return None

    port = _find_free_port()
    root_password = "test_pwd_123"
    database = "rockfall_test"
    container_name = f"rockfall_test_mysql_{port}"

    # 清理可能残留的同名容器
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, timeout=10,
    )

    # 启动 MySQL 容器
    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-e", f"MYSQL_ROOT_PASSWORD={root_password}",
                "-e", f"MYSQL_DATABASE={database}",
                "-p", f"127.0.0.1:{port}:3306",
                "mysql:8.0",
                "--default-authentication-plugin=mysql_native_password",
                "--max_connections=20",
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        # Docker run 失败 (可能镜像未拉取、端口冲突等)
        print(f"[conftest] Docker MySQL 启动失败: {e.stderr}")
        return None

    # 等待 MySQL 就绪 (最多 60 秒)
    _wait_mysql_ready(port, root_password, database, timeout=60)

    connection_params = {
        "host": "127.0.0.1",
        "port": port,
        "user": "root",
        "password": root_password,
        "database": database,
    }

    yield connection_params

    # 清理
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, timeout=10,
    )


def _wait_mysql_ready(port: int, password: str, database: str, timeout: int = 60):
    """轮询等待 MySQL 接受连接"""
    import pymysql
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = pymysql.connect(
                host="127.0.0.1", port=port,
                user="root", password=password,
                database=database, charset="utf8mb4",
                connect_timeout=3,
            )
            conn.close()
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"MySQL 容器在 {timeout}s 内未就绪")


# ================================================================
# 函数级: 临时数据目录
# ================================================================

@pytest.fixture
def tmp_data_dir(monkeypatch):
    """
    函数级临时 DATA_DIR — 每个测试函数独立的 alert.db / sites.db。

    通过 monkeypatch 覆盖 rockfall.config.DATA_DIR,
    确保测试不会污染用户真实数据。
    """
    import gc
    import rockfall.config as cfg

    td = tempfile.mkdtemp(prefix="rockfall_test_")
    tmp_path = Path(td)
    # 创建必要子目录
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "uploads").mkdir(exist_ok=True)

    # 覆盖模块级常量
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cfg, "UPLOADS_DIR", tmp_path / "uploads")

    yield tmp_path

    # 显式清理: 强制 gc 释放 SQLite 连接, 然后删除临时目录
    gc.collect()
    try:
        import shutil
        shutil.rmtree(str(tmp_path), ignore_errors=True)
    except Exception:
        pass


# ================================================================
# 函数级: SQLite AlertStore (每次测试独立 DB)
# ================================================================

@pytest.fixture
def sqlite_store(tmp_data_dir, monkeypatch):
    """
    函数级 SQLite AlertStore — 数据库位于独立的临时文件中。

    强制使用 SQLite, 并确保所有增量列 (review_status 等) 通过 _run_migrations 添加。
    """
    import rockfall.alert_store as als
    import rockfall.config as cfg

    # 强制 SQLite: 覆盖 alert_store 模块 + config 模块的 MYSQL_HOST
    old_mysql_als = als.MYSQL_HOST
    old_mysql_cfg = cfg.MYSQL_HOST
    als.MYSQL_HOST = ""
    monkeypatch.setattr(cfg, "MYSQL_HOST", "")

    # 阻止 Alembic 以 MySQL 模式运行 (否则会打印 "Context impl MySQLImpl" 且不应用 SQLite 迁移)
    # run_migrations 在 alert_store._init_db 中通过 from rockfall.migration import run_migrations 延迟导入
    import rockfall.migration as _mig
    def _raise_skip(*args, **kwargs):
        raise RuntimeError("Alembic skipped in test — use _run_migrations instead")
    monkeypatch.setattr(_mig, "run_migrations", _raise_skip)

    db_path = str(tmp_data_dir / "alerts.db")
    store = als.AlertStore(db_path=db_path)
    store._stop_retry.set()  # 阻止重试线程

    # 手动运行增量迁移 (Alembic 已被 monkeypatch 跳过)
    store._run_migrations()

    yield store

    als.MYSQL_HOST = old_mysql_als
    cfg.MYSQL_HOST = old_mysql_cfg


# ================================================================
# 函数级: SQLite SiteStore
# ================================================================

@pytest.fixture
def sqlite_site_store(tmp_data_dir, monkeypatch):
    """
    函数级 SQLite SiteStore。

    覆盖 rockfall.site_config 的 MYSQL 配置, 强制使用 SQLite 后端。
    """
    import rockfall.site_config as sc

    # 绕过单例缓存, 每次创建新实例
    old_store = sc._site_store
    sc._site_store = None

    # 强制 SQLite
    monkeypatch.setattr(sc, "_MYSQL_AVAILABLE", False)

    # 临时 site_config 路径
    site_db = tmp_data_dir / "sites.db"
    monkeypatch.setattr(sc, "DATA_DIR", tmp_data_dir)
    monkeypatch.setattr(sc, "SITE_CONFIG_PATH", tmp_data_dir / "site_config.json")
    monkeypatch.setattr(sc, "SITE_STATE_PATH", tmp_data_dir / "site_state.json")
    monkeypatch.setattr(sc, "ROI_CONFIG_PATH", tmp_data_dir / "site_config.json")
    monkeypatch.setattr(sc, "CONFIG_PATH", tmp_data_dir / "site_config.json")

    store = sc.SiteStore()
    # 重新绑定模块级单例 (供 get_site_store 使用)
    sc._site_store = store

    yield store

    # 清理
    sc._site_store = old_store


# ================================================================
# 函数级: MySQL 后端 AlertStore + SiteStore
# ================================================================

@pytest.fixture
def mysql_backend(mysql_container, tmp_data_dir, monkeypatch):
    """
    函数级 MySQL 后端 AlertStore。

    如果 mysql_container 为 None (Docker 不可用), 自动 pytest.skip。
    每次测试前清空 alerts 表, 确保隔离。
    """
    if mysql_container is None:
        pytest.skip("Docker 不可用, 跳过 MySQL 测试")

    import pymysql
    import rockfall.alert_store as als

    # 注入 MySQL 连接参数
    monkeypatch.setattr(als, "MYSQL_HOST", mysql_container["host"])
    monkeypatch.setattr(als, "MYSQL_PORT", mysql_container["port"])
    monkeypatch.setattr(als, "MYSQL_USER", mysql_container["user"])
    monkeypatch.setattr(als, "MYSQL_PASSWORD", mysql_container["password"])
    monkeypatch.setattr(als, "MYSQL_DATABASE", mysql_container["database"])
    monkeypatch.setattr(als, "_MYSQL_AVAILABLE", True)

    store = als.AlertStore(db_path=str(tmp_data_dir / "alerts.db"))
    store._stop_retry.set()

    # 清理旧数据
    _truncate_mysql(mysql_container, "alerts")

    yield store

    _truncate_mysql(mysql_container, "alerts")


@pytest.fixture
def mysql_site_store(mysql_container, tmp_data_dir, monkeypatch):
    """
    函数级 MySQL 后端 SiteStore。

    如果 mysql_container 为 None, 自动 pytest.skip。
    """
    if mysql_container is None:
        pytest.skip("Docker 不可用, 跳过 MySQL 测试")

    import rockfall.site_config as sc

    # 清除单例缓存
    sc._site_store = None

    # 强制 MySQL 可用 + 注入连接参数
    monkeypatch.setattr(sc, "_MYSQL_AVAILABLE", True)
    monkeypatch.setattr(sc, "DATA_DIR", tmp_data_dir)

    # 临时覆盖 config 中的 MySQL 参数
    import rockfall.config as cfg
    monkeypatch.setattr(cfg, "MYSQL_HOST", mysql_container["host"])
    monkeypatch.setattr(cfg, "MYSQL_PORT", mysql_container["port"])
    monkeypatch.setattr(cfg, "MYSQL_USER", mysql_container["user"])
    monkeypatch.setattr(cfg, "MYSQL_PASSWORD", mysql_container["password"])
    monkeypatch.setattr(cfg, "MYSQL_DATABASE", mysql_container["database"])

    store = sc.SiteStore()
    sc._site_store = store

    # 清理旧数据
    _truncate_mysql(mysql_container, "monitoring_sites")

    yield store

    _truncate_mysql(mysql_container, "monitoring_sites")
    sc._site_store = None


def _truncate_mysql(conn_params: dict, table: str):
    """清空 MySQL 表"""
    import pymysql
    try:
        conn = pymysql.connect(
            host=conn_params["host"], port=conn_params["port"],
            user=conn_params["user"], password=conn_params["password"],
            database=conn_params["database"], charset="utf8mb4",
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table}")
        conn.commit()
        conn.close()
    except Exception:
        pass


# ================================================================
# 函数级: FastAPI TestClient
# ================================================================

@pytest.fixture
def client(tmp_data_dir, monkeypatch):
    """
    FastAPI TestClient — 用于测试 API 端点。

    自动覆盖配置以使用临时目录 (不污染真实数据)。
    """
    import rockfall.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_data_dir)
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_data_dir / "results")
    monkeypatch.setattr(cfg, "UPLOADS_DIR", tmp_data_dir / "uploads")

    # 延迟导入 app, 让 monkeypatch 先生效
    from server.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as tc:
        yield tc


# ================================================================
# 函数级: Mock RockDetector
# ================================================================

class _MockDetector:
    """
    Mock RockDetector — 不加载 YOLO 模型, 返回预设检测结果。

    通过 patch 掉 RockDetector.__init__ 来绕过模型加载,
    同时提供可控的 detect_image / detect_video 返回值。
    """

    # 模拟 RockDetector._model_cache 类属性 (YOLO 模型缓存)
    _model_cache: dict = {}

    def __init__(self, site_id=""):
        self.site_id = site_id
        self.confidence = 0.35
        self.img_size = 640
        self.min_area = 500
        self.alert_blue_conf_high = 0.5
        self.alert_yellow_conf_high = 0.7
        self.alert_orange_conf_high = 0.9
        self._active_model_path = "models/mock.pt"
        self._device_str = "cpu"
        self._device_name = "CPU (Mock)"

    def detect_image(self, image_path: str, push_alert: bool = True) -> dict:
        return {
            "detection": "落石检测到",
            "time": "2026-06-14 12:00:00",
            "count": 2,
            "max_confidence": 0.85,
            "saved_to": str(Path(image_path).parent / "result.jpg"),
            "push_status": {"code": 200, "msg": "ok"},
        }

    def detect_video(self, video_path, save_frames=True, push_alerts=True,
                     track=True, confirm_frames=3, polygon=None,
                     max_frames=None, stride=1, progress_callback=None) -> dict:
        # 模拟进度回调
        if progress_callback:
            for i in range(1, 11):
                progress_callback(i, 10)
        return {
            "source": Path(video_path).name,
            "resolution": "1920x1080",
            "total_frames": 10,
            "fps": 25.0,
            "frames_with_detections": 3,
            "detections": [
                {
                    "frame": 5,
                    "time_sec": 0.2,
                    "alert_level": "orange",
                    "boxes": [
                        {
                            "track_id": 1,
                            "bbox": [100, 200, 180, 300],
                            "confidence": 0.85,
                            "speed": 5.0,
                            "motion_state": "运动",
                            "confirmed": True,
                            "class_id": 0,
                            "class_name": "落石",
                        }
                    ],
                }
            ],
        }

    def init_stream_state(self, fw, fh, roi_mask=None):
        self._stream_ready = True
        self.frame_w = fw
        self.frame_h = fh

    def preprocess_frame(self, frame):
        """模拟 MOG2 预处理, 返回静止帧结果"""
        import numpy as np
        fh, fw = frame.shape[:2]
        return {
            'fg': np.zeros((fh, fw), dtype=np.uint8),
            'motion_score': 0.0, 'has_motion': False,
            'box_mask': np.zeros((fh, fw), dtype=np.uint8),
            'skip': 8,
        }

    def detect_frame(self, frame, box_mask=None, fg_mask=None):
        """模拟 YOLO 推理, 返回空检测"""
        return []

    @staticmethod
    def draw_tracks(frame, tracks, polygon=None, fw=0, fh=0, alert_level="",
                    show_panel=False, show_border=False):
        """静态绘制方法 (兼容桌面端调用)"""
        pass

    @staticmethod
    def build_alert_context(tracks, frame_w=0, frame_h=0):
        """静态方法 — 委托给真实实现"""
        # 直接使用真实 RockDetector 的静态方法 (纯函数, 不依赖模型)
        from rockfall.detector import RockDetector as _Real
        return _Real.build_alert_context(tracks, frame_w, frame_h)


@pytest.fixture
def mock_detector(monkeypatch):
    """
    替换 RockDetector 为 Mock, 避免加载 YOLO 模型。

    同时替换 rockfall.detector 和 server.service 中的引用
    (因为 server.service 使用 from X import Y, 持有独立引用)。

    用法:
        def test_detect(mock_detector):
            det = RockDetector()
            result = det.detect_image("test.jpg")
            assert result["count"] == 2
    """
    import rockfall.detector as det_mod

    # 缓存原始类
    _orig = det_mod.RockDetector

    det_mod.RockDetector = _MockDetector

    # 同时更新 server.service 中的导入 (from rockfall.detector import RockDetector)
    import server.service as svc_mod
    _svc_orig = svc_mod.RockDetector
    svc_mod.RockDetector = _MockDetector

    yield _MockDetector

    det_mod.RockDetector = _orig
    svc_mod.RockDetector = _svc_orig


# ================================================================
# 函数级: 配置隔离 (防止模块级常量状态泄漏)
# ================================================================

@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    """
    每个测试函数自动执行, 确保测试隔离。

    - 禁用 PushPlus 推送 (避免测试中真实发送 HTTP 请求)
    - monkeypatch 在测试结束后自动恢复原始值

    autouse=True: 所有测试自动应用。
    """
    import rockfall.config as cfg

    # 禁用推送 (避免测试中真实发送)
    monkeypatch.setattr(cfg, "PUSHPLUS_TOKEN", "")
    monkeypatch.setattr(cfg, "PUSHPLUS_URL", "http://localhost:9999/push")

    yield
    # monkeypatch 自动恢复, 无需手动处理


# ================================================================
# pytest 配置
# ================================================================

def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line(
        "markers", "mysql: 需要 Docker MySQL 容器的测试 (默认跳过)"
    )


def pytest_collection_modifyitems(config, items):
    """
    自动为包含 'mysql' 关键字的测试添加 mysql 标记,
    并在 Docker 不可用时自动跳过。
    """
    if is_docker_ok():
        return  # Docker OK, 所有测试正常运行

    skip_mysql = pytest.mark.skip(reason="Docker 不可用, 跳过 MySQL 测试")
    for item in items:
        # 按函数名或类名自动识别 MySQL 测试
        name = item.name.lower()
        cls_name = item.cls.__name__.lower() if item.cls else ""
        if "mysql" in name or "mysql" in cls_name:
            item.add_marker(skip_mysql)
