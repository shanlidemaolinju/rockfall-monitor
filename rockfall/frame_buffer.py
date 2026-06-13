"""
帧环形缓冲 — 内存缓存 + 异步压缩 + 告警触发落盘
==================================================
将检测帧保存在内存环形缓冲中（默认 150 帧），异步 JPEG 压缩，
仅在告警触发时才 flush 到磁盘，大幅减少磁盘 IO。

双缓冲策略:
  - 原始帧: 480p 缩略图 (节省内存)
  - 标注帧: 全分辨率 (用于告警时落盘)

用法:
    buf = FrameRingBuffer(maxlen=150)
    # 每帧调用: buf.push(frame_idx, frame_bgr, annotated_bgr)
    # 告警时: buf.flush_alert(RESULTS_DIR, frame_indices=[...])
"""

import io
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FrameRingBuffer:
    """内存环形帧缓冲 — 异步 JPEG 压缩, 告警触发落盘。"""

    def __init__(self, maxlen: int = 150, jpeg_quality: int = 70,
                 thumbnail_height: int = 480, compress_workers: int = 2):
        """
        参数:
            maxlen:            最大缓存帧数 (150 帧 ≈ 12s @ 12.5fps)
            jpeg_quality:      JPEG 压缩质量 0-100
            thumbnail_height:  原始帧缩略图高度 (480p)
            compress_workers:  异步压缩线程数
        """
        self._maxlen = maxlen
        self._quality = jpeg_quality
        self._thumb_h = thumbnail_height

        # 标注帧 (全分辨率 BGR) + 原始帧 (缩略图 BGR)
        self._annotated: deque[tuple[int, np.ndarray]] = deque(maxlen=maxlen)
        self._raw_thumb: deque[tuple[int, np.ndarray]] = deque(maxlen=maxlen)

        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=compress_workers, thread_name_prefix="frame-compress"
        )
        self._compressed_cache: dict[int, bytes] = {}  # frame_idx → JPEG bytes
        self._futures: dict[int, Any] = {}

    # ---- 推入 ----

    def push(self, frame_idx: int, frame_bgr: np.ndarray,
             annotated_bgr: np.ndarray | None = None):
        """推入一帧到环形缓冲。

        参数:
            frame_idx:     帧序号
            frame_bgr:     原始帧 (BGR, 全分辨率)
            annotated_bgr: 标注帧 (BGR, 全分辨率)，若为 None 则等于原始帧
        """
        if annotated_bgr is None:
            annotated_bgr = frame_bgr

        # 生成缩略图
        h, w = frame_bgr.shape[:2]
        thumb_w = int(w * self._thumb_h / h)
        raw_thumb = cv2.resize(frame_bgr, (thumb_w, self._thumb_h))

        with self._lock:
            self._annotated.append((frame_idx, annotated_bgr.copy()))
            self._raw_thumb.append((frame_idx, raw_thumb))

        # 异步压缩标注帧为 JPEG（不超过 50 个并发 future）
        if len(self._futures) < 50:
            future = self._executor.submit(
                self._compress, frame_idx, annotated_bgr
            )
            self._futures[frame_idx] = future
            # 清理已完成的 future
            self._cleanup_futures()

    def _compress(self, frame_idx: int, frame_bgr: np.ndarray) -> bytes | None:
        """压缩单帧为 JPEG 字节。"""
        ok, jpg = cv2.imencode(
            ".jpg", frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self._quality],
        )
        if ok:
            data = jpg.tobytes()
            with self._lock:
                self._compressed_cache[frame_idx] = data
            return data
        return None

    def _cleanup_futures(self):
        """移除已完成的 future（防止 _futures 无限增长）。"""
        done = [idx for idx, f in self._futures.items() if f.done()]
        for idx in done:
            del self._futures[idx]

    # ---- 告警落盘 ----

    def flush_alert(self, directory: Path | str,
                    alert_frame_idx: int,
                    context_frames: int = 30) -> list[str]:
        """告警触发时将标注帧写入磁盘。

        写入: alert_frame_idx 前后各 context_frames 帧。

        返回: 已写入的文件路径列表。
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # 等待压缩完成（最多 5 秒）
        for idx, future in list(self._futures.items()):
            if abs(idx - alert_frame_idx) <= context_frames:
                try:
                    future.result(timeout=5.0)
                except Exception:
                    pass

        saved = []
        with self._lock:
            # 查找告警帧及其上下文
            indices_to_save = set()
            start_idx = max(0, alert_frame_idx - context_frames)
            end_idx = alert_frame_idx + context_frames

            for idx in range(start_idx, end_idx + 1):
                if idx in self._compressed_cache:
                    indices_to_save.add(idx)

            for idx in sorted(indices_to_save):
                jpg_data = self._compressed_cache.get(idx)
                if jpg_data is None:
                    continue
                filename = directory / f"stream_{idx:06d}.jpg"
                try:
                    with open(filename, "wb") as f:
                        f.write(jpg_data)
                    saved.append(str(filename))
                except Exception:
                    pass

        return saved

    # ---- 查询 ----

    def get_recent_annotated(self, n: int = 10) -> list[tuple[int, np.ndarray]]:
        """获取最近 N 帧标注帧 (frame_idx, bgr_array)。"""
        with self._lock:
            return list(self._annotated)[-n:]

    def get_recent_jpegs(self, n: int = 10) -> list[dict]:
        """获取最近 N 帧 JPEG 数据 (用于 API 返回)。"""
        import base64
        result = []
        with self._lock:
            recent = list(self._annotated)[-n:]
        for idx, bgr in recent:
            if idx in self._compressed_cache:
                jpg_b64 = base64.b64encode(
                    self._compressed_cache[idx]
                ).decode("ascii")
            else:
                ok, jpg = cv2.imencode(
                    ".jpg", bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self._quality],
                )
                jpg_b64 = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""
            result.append({"frame_idx": idx, "jpeg_base64": jpg_b64})
        return result

    def clear(self):
        """清空缓冲和缓存。"""
        with self._lock:
            self._annotated.clear()
            self._raw_thumb.clear()
            self._compressed_cache.clear()
            self._futures.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._annotated)

    def shutdown(self):
        """关闭异步压缩线程池。"""
        self._executor.shutdown(wait=False)
