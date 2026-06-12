"""
API 数据模型 — Pydantic schemas
================================
为所有 API 端点提供请求/响应类型定义，
FastAPI 自动生成精确的 OpenAPI (Swagger) 文档。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 通用
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    error: str


# ============================================================
# 看板统计
# ============================================================

class DashboardStats(BaseModel):
    today_total: int = Field(0, description="今日检测事件总数")
    today_red: int = Field(0, description="今日Ⅰ级(红色)预警次数")
    today_orange: int = Field(0, description="今日Ⅱ级(橙色)预警次数")
    today_yellow: int = Field(0, description="今日Ⅲ级(黄色)预警次数")
    today_blue: int = Field(0, description="今日Ⅳ级(蓝色)预警次数")
    last_count: Optional[int] = Field(None, description="最近一次检出目标数")
    last_conf: Optional[float] = Field(None, description="最近一次最大置信度")
    last_alert_level: Optional[str] = Field(None, description="最近一次预警等级")


class AlertItem(BaseModel):
    id: int = Field(0, description="预警记录 ID")
    time: str = Field("", description="预警时间")
    alert_level: str = Field("green", description="预警等级: red(Ⅰ级)/orange(Ⅱ级)/yellow(Ⅲ级)/blue(Ⅳ级)")
    count: int = Field(0, description="目标数量")
    max_confidence: float = Field(0, description="最大置信度")
    track_ids: list[int] = Field(default_factory=list, description="跟踪目标ID")
    class_summary: str = Field("", description="类别分布 (如: 落石:2, 滑坡:1)")
    saved_frame: str = Field("", description="保存的帧截图路径")
    push_status: str = Field("pending", description="推送状态: pending/sent/failed/recorded/popup")


# ============================================================
# 检测结果
# ============================================================

class DetectionBox(BaseModel):
    track_id: int = Field(..., description="SORT 跟踪 ID")
    bbox: list[float] = Field(..., description="边界框 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    speed: float = Field(0, description="运动速度 (px/frame)")
    motion_state: str = Field("未知", description="运动状态")
    confirmed: bool = Field(False, description="轨迹是否已确认")
    class_id: int = Field(0, description="检测类别 ID (0=落石, 1=滑坡)")
    class_name: str = Field("落石", description="检测类别名称")


class FrameDetection(BaseModel):
    frame: int = Field(..., description="帧序号")
    time_sec: float = Field(..., description="视频时间 (秒)")
    alert_level: str = Field("green", description="该帧预警等级")
    boxes: list[DetectionBox] = Field(default_factory=list)


class ImageDetectResponse(BaseModel):
    detection: str = Field("", description="检测结果摘要")
    time: str = Field("", description="检测时间")
    count: int = Field(0, description="检测目标数量")
    max_confidence: float = Field(0, description="最大置信度")
    saved_to: str = Field("", description="结果图片路径")
    push_status: Optional[dict] = Field(None, description="推送结果")


class VideoDetectResponse(BaseModel):
    source: str = Field("", description="视频来源")
    resolution: str = Field("", description="分辨率 WxH")
    total_frames: int = Field(0, description="总帧数")
    fps: float = Field(0, description="视频帧率")
    frames_with_detections: int = Field(0, description="检出帧数")
    detections: list[FrameDetection] = Field(default_factory=list)


class TaskResponse(BaseModel):
    """异步任务提交响应"""
    task_id: str = Field(..., description="任务 ID, 用于轮询结果")
    status: str = Field("processing", description="任务状态")


class TaskStatusResponse(BaseModel):
    """异步任务状态查询响应"""
    task_id: str
    status: str = Field(..., description="processing / completed / failed")
    result: Optional[dict] = Field(None, description="检测结果 (completed 时有效)")
    error: Optional[str] = Field(None, description="错误信息 (failed 时有效)")
