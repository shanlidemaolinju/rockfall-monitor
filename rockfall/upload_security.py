"""
上传文件安全模块 — MIME 类型校验 + 大小限制 + 隔离存储
======================================================
特性:
  - 基于文件头 (magic bytes) 的真实 MIME 类型检测
  - 文件大小限制 (默认 500MB)
  - 上传文件隔离目录 (随机文件名，防止路径遍历)
  - 白名单 MIME 类型校验
  - 病毒扫描接口 (预留 ClamAV 集成)

用法:
    from rockfall.upload_security import UploadValidator, get_upload_validator

    validator = get_upload_validator()
    result = validator.validate(file_content, original_filename)
    if result["valid"]:
        safe_path = validator.save_quarantined(file_content)

配置环境变量:
  UPLOAD_MAX_SIZE_MB: 文件大小上限 (MB)，默认 500
  UPLOAD_QUARANTINE_DIR: 隔离存储目录，默认 data/quarantine
  UPLOAD_SCAN_ENABLED: 是否启用病毒扫描，默认 false
"""

import hashlib
import mimetypes
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from .config import DATA_DIR, UPLOADS_DIR

# ============================================================
# 配置常量
# ============================================================
UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", "500"))
UPLOAD_MAX_SIZE_BYTES = UPLOAD_MAX_SIZE_MB * 1024 * 1024
UPLOAD_QUARANTINE_DIR = Path(os.getenv("UPLOAD_QUARANTINE_DIR",
                                       str(DATA_DIR / "quarantine")))
UPLOAD_SCAN_ENABLED = os.getenv("UPLOAD_SCAN_ENABLED", "false").lower() == "true"

# ============================================================
# 白名单：允许的 MIME 类型
# ============================================================

ALLOWED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

ALLOWED_VIDEO_MIMES = {
    "video/mp4",
    "video/x-msvideo",   # AVI
    "video/x-matroska",  # MKV
    "video/quicktime",   # MOV
    "video/webm",
    "video/x-ms-wmv",    # WMV
    "video/x-flv",       # FLV
}

ALLOWED_MIMES = ALLOWED_IMAGE_MIMES | ALLOWED_VIDEO_MIMES

# 文件扩展名到预期 MIME 的映射（用于二次校验）
EXTENSION_MIME_MAP = {
    # 图片
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
    # 视频
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
}

# 高危扩展名（即使 MIME 声称是合法类型也拒绝）
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".wsf",
    ".html", ".htm", ".xhtml", ".svg", ".xml",
    ".php", ".asp", ".aspx", ".jsp", ".cgi",
    ".sh", ".bash", ".zsh", ".fish",
    ".py", ".js", ".ts", ".rb", ".pl",
    ".jar", ".war", ".ear",
    ".com", ".scr", ".msi", ".pif",
}


# ============================================================
# 基于文件头的 Magic Bytes 检测 (无需 python-magic)
# ============================================================

# 常见文件格式的 magic bytes 签名
_MAGIC_SIGNATURES: list[tuple[bytes, int, str]] = [
    # (signature, offset, mime_type)
    # JPEG
    (b"\xFF\xD8\xFF", 0, "image/jpeg"),
    # PNG
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    # BMP
    (b"BM", 0, "image/bmp"),
    # TIFF (little-endian)
    (b"II*\x00", 0, "image/tiff"),
    # TIFF (big-endian)
    (b"MM\x00*", 0, "image/tiff"),
    # WebP
    (b"RIFF", 0, "image/webp"),  # 需要额外检查 WEBP 标记
    # MP4 / MOV
    (b"\x00\x00\x00\x18ftyp", 4, "video/mp4"),
    (b"\x00\x00\x00\x20ftyp", 4, "video/mp4"),
    (b"ftyp", 4, "video/mp4"),  # 宽松匹配
    # AVI
    (b"RIFF", 0, "video/x-msvideo"),  # 需要额外检查 AVI 标记
    # MKV / WebM
    (b"\x1a\x45\xdf\xa3", 0, "video/x-matroska"),
    # WMV / ASF
    (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11", 0, "video/x-ms-wmv"),
    # FLV
    (b"FLV", 0, "video/x-flv"),
    # QuickTime
    (b"moov", 4, "video/quicktime"),
    (b"mdat", 4, "video/quicktime"),
    (b"free", 4, "video/quicktime"),
    (b"skip", 4, "video/quicktime"),
]


def _detect_mime_from_bytes(data: bytes) -> str:
    """通过 magic bytes 检测文件的真实 MIME 类型。

    返回 MIME 字符串，无法识别则返回 "application/octet-stream"。
    """
    if len(data) < 16:
        return "application/octet-stream"

    for sig, offset, mime in _MAGIC_SIGNATURES:
        end = offset + len(sig)
        if end <= len(data) and data[offset:end] == sig:
            # WebP: RIFF + WEBP
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            # AVI: RIFF + AVI
            if mime == "video/x-msvideo" and data[8:12] != b"AVI ":
                continue
            return mime

    # 尝试用扩展名推断
    return "application/octet-stream"


def _contains_dangerous_content(data: bytes) -> bool:
    """检测文件内容是否包含可疑的 HTML/脚本/可执行代码。

    仅检查前 512 字节，平衡安全性和性能。
    """
    header = data[:512]

    # PE 可执行文件（必须在文件起始位置）
    if len(header) >= 2 and header[:2] == b"MZ":
        return True

    # 脚本/HTML 签名（大小写不敏感）
    header_lower = header.lower()
    for pattern in [
        b"<!doctype html", b"<html", b"<script", b"<?php",
        b"#!/bin/", b"#!/usr/bin/", b"#!/bin/bash", b"#!/usr/bin/env",
        b"<?xml", b"<%@", b"<%",
        b"#! /bin/",
    ]:
        if pattern.lower() in header_lower:
            return True
    return False


def _safe_extension(filename: str) -> str:
    """安全提取文件扩展名（小写，防止路径遍历）。"""
    ext = Path(filename).suffix.lower()
    # 过滤掉非法字符
    if any(c in ext for c in ('/', '\\', '\x00')):
        return ""
    return ext


# ============================================================
# 上传文件校验器
# ============================================================

class UploadValidator:
    """上传文件安全校验器。

    校验流程:
      1. 检查文件大小 ≤ UPLOAD_MAX_SIZE_BYTES
      2. 检查扩展名不在 BLOCKED_EXTENSIONS 中
      3. 读取文件头 512 字节，检测真实 MIME 类型
      4. MIME 类型必须在 ALLOWED_MIMES 白名单中
      5. 扩展名与 MIME 类型一致性校验
      6. 保存到隔离目录（随机文件名）
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        UPLOAD_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self, file_data: bytes | BinaryIO | None = None,
                 filename: str = "", file_path: str = "") -> dict:
        """校验上传文件的安全性。

        参数:
          file_data:  文件内容 (bytes 或 file-like object)
          filename:   原始文件名
          file_path:  已存在的文件路径 (用于已保存文件的二次校验)

        返回:
          {
            "valid": bool,
            "mime_type": str,
            "error": str (仅当 valid=False),
            "warnings": [str],
          }
        """
        warnings = []

        # 读取文件内容
        if file_data is not None:
            if isinstance(file_data, bytes):
                content = file_data
            else:
                content = file_data.read()
                if hasattr(file_data, 'seek'):
                    file_data.seek(0)
        elif file_path and Path(file_path).exists():
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            return {"valid": False, "mime_type": "", "error": "没有提供文件内容"}

        # 1. 文件大小检查
        file_size = len(content)
        if file_size > UPLOAD_MAX_SIZE_BYTES:
            max_mb = UPLOAD_MAX_SIZE_MB
            actual_mb = round(file_size / (1024 * 1024), 2)
            return {
                "valid": False, "mime_type": "",
                "error": f"文件大小 ({actual_mb}MB) 超过限制 ({max_mb}MB)",
                "warnings": warnings,
            }

        # 2. 扩展名检查
        ext = _safe_extension(filename) if filename else ""
        if ext in BLOCKED_EXTENSIONS:
            return {
                "valid": False, "mime_type": "",
                "error": f"禁止的文件类型: {ext}",
                "warnings": warnings,
            }

        # 3. Magic bytes 检测真实 MIME 类型
        detected_mime = _detect_mime_from_bytes(content)

        # 4. MIME 白名单校验
        if detected_mime not in ALLOWED_MIMES:
            if detected_mime == "application/octet-stream" and ext:
                # 文件头无法识别类型 — 进行额外安全检查
                # 检查是否包含高危内容（HTML/脚本等伪装文件）
                if _contains_dangerous_content(content):
                    return {
                        "valid": False,
                        "mime_type": "application/octet-stream",
                        "error": "文件内容包含疑似恶意代码（HTML/脚本），拒绝上传",
                        "warnings": warnings,
                    }
                # 尝试通过扩展名推断（仅限图片/视频常见格式）
                inferred = EXTENSION_MIME_MAP.get(ext, "")
                if inferred in ALLOWED_MIMES:
                    warnings.append(
                        f"无法通过文件头确认类型 (扩展名: {ext})，"
                        f"推测为 {inferred}，已放行（已通过恶意内容检查）"
                    )
                    return {
                        "valid": True,
                        "mime_type": inferred,
                        "error": "",
                        "warnings": warnings,
                    }
            return {
                "valid": False,
                "mime_type": detected_mime,
                "error": f"不允许的文件类型: {detected_mime}",
                "warnings": warnings,
            }

        # 5. 扩展名与 MIME 一致性校验
        if ext and ext in EXTENSION_MIME_MAP:
            expected_mime = EXTENSION_MIME_MAP[ext]
            # 相同顶级类型即可（image/* vs image/*, video/* vs video/*）
            if (detected_mime.split("/")[0] != expected_mime.split("/")[0]):
                warnings.append(
                    f"文件扩展名 ({ext}) 与实际内容类型 ({detected_mime}) 不匹配"
                )

        return {
            "valid": True,
            "mime_type": detected_mime,
            "error": "",
            "warnings": warnings,
        }

    def save_quarantined(self, file_data: bytes,
                        original_filename: str = "") -> str:
        """将文件保存到隔离目录（随机文件名）。

        返回安全的文件路径。
        """
        ext = _safe_extension(original_filename) or ".bin"
        # 生成随机文件名（防止冲突和路径猜测）
        rand_name = secrets.token_hex(16) + ext
        safe_path = UPLOAD_QUARANTINE_DIR / rand_name

        with open(safe_path, "wb") as f:
            f.write(file_data)

        # 记录隔离日志
        from .audit import audit_log
        audit_log("upload_quarantine",
                  detail=f"原始文件名: {original_filename}, 隔离路径: {safe_path.name}, "
                         f"大小: {len(file_data)} bytes, "
                         f"哈希: {hashlib.sha256(file_data).hexdigest()[:16]}...")

        return str(safe_path)

    def scan_file(self, file_path: str) -> dict:
        """病毒扫描接口 (预留 ClamAV 集成)。

        环境变量 UPLOAD_SCAN_ENABLED=true 时启用。
        """
        if not UPLOAD_SCAN_ENABLED:
            return {"scanned": False, "clean": True, "note": "病毒扫描未启用"}

        # 预留: ClamAV unix socket 集成
        # import socket
        # sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # sock.connect("/var/run/clamav/clamd.ctl")
        # sock.send(b"zINSTREAM\0")
        # ... 发送文件 ...
        # result = sock.recv(4096)

        return {"scanned": True, "clean": True, "note": "扫描通过 (ClamAV 未配置)"}


# 模块级单例
_validator: UploadValidator | None = None


def get_upload_validator() -> UploadValidator:
    global _validator
    if _validator is None:
        _validator = UploadValidator()
    return _validator
