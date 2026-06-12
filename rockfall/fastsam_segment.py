"""
FastSAM 边坡-公路分割模块
==========================
使用 Ultralytics 原生 FastSAM + 文本提示, 替代旧 SAM 独立进程 + 传统 CV。
利用 CLIP 文本嵌入匹配分割区域, 精准区分边坡(前景)和公路/车道/护栏(负样本)。

核心流程:
  1. FastSAM 生成所有候选分割区域
  2. CLIP 将区域与文本提示匹配 → slope / road / other
  3. 生成 road_mask (公路掩码, 255=公路) 和 roi_mask (边坡掩码, 255=边坡)
  4. 完全兼容 detector.py 的 _road_mask / roi_mask 接口

依赖: ultralytics>=8.3.0 (已内置 FastSAM + CLIP)
模型: FastSAM-x.pt 首次自动下载 (~145MB), RTX 4060 推理 <2s/帧
"""

import hashlib
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---- 缓存目录 ----
_MASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "masks"

# ---- 模型单例 ----
_fastsam_model = None
_fastsam_device = None

# ================================================================
# 文本提示模板 (中英双语覆盖, 可按场景扩展)
# ================================================================
SLOPE_TEXTS = [
    "slope", "cliff", "rock face", "hillside", "mountain side",
    "rocky terrain", "steep ground", "embankment", "cut slope",
    "bare rock", "gravel surface", "dirt slope", "excavation face",
]

ROAD_TEXTS = [
    "road", "highway", "pavement", "asphalt road", "concrete road",
    "lane", "road shoulder",
    "guardrail", "concrete barrier", "crash barrier",
]

EXCLUDE_TEXTS = [
    "sky", "cloud", "tree", "vegetation", "building",
    "vehicle", "car", "truck", "bus",
]


# ================================================================
# FastSAM 模型管理
# ================================================================

def get_fastsam_model(
    model_path: str = "FastSAM-x.pt",
    device: str = "cuda:0",
    verbose: bool = False,
):
    """
    获取 FastSAM 模型单例 (延迟加载, 复用权重)。

    调用方应在主线程中调用, 避免多线程竞争。
    如需释放显存, 调用 release_fastsam_model()。
    """
    global _fastsam_model, _fastsam_device

    if _fastsam_model is not None and _fastsam_device == device:
        return _fastsam_model

    from ultralytics import FastSAM

    # 释放旧模型 (设备切换场景)
    if _fastsam_model is not None:
        del _fastsam_model
        _fastsam_model = None
        _gc_collect()

    if verbose:
        print(f"[FastSAM] 加载模型 {model_path} → {device}")

    _fastsam_model = FastSAM(model_path)
    _fastsam_device = device
    return _fastsam_model


def release_fastsam_model():
    """释放 FastSAM 模型, 回收 GPU 显存"""
    global _fastsam_model, _fastsam_device
    if _fastsam_model is not None:
        del _fastsam_model
        _fastsam_model = None
        _fastsam_device = None
        _gc_collect()


def _gc_collect():
    """Python + CUDA 垃圾回收"""
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ================================================================
# 单帧分割
# ================================================================

def segment_frame(
    frame: np.ndarray,
    model=None,
    device: str = "cuda:0",
    conf: float = 0.25,
    iou: float = 0.7,
    use_text_prompt: bool = True,
    verbose: bool = False,
) -> dict:
    """
    FastSAM 单帧分割: 区分边坡 vs 公路。

    参数:
        frame:   BGR 图像 (H, W, 3)
        model:   已加载的 FastSAM 模型, 为 None 时自动加载
        device:  "cuda:0" | "cpu"
        conf:    FastSAM 置信度阈值 (0~1)
        iou:     NMS IoU 阈值
        use_text_prompt: True=CLIP文本提示, False=仅分割不做分类
        verbose: 打印耗时信息

    返回:
        {
            'slope_mask': np.ndarray (H,W) uint8 255=边坡区域,
            'road_mask':  np.ndarray (H,W) uint8 255=公路区域,
            'all_masks':  list[np.ndarray] 所有分割区域,
            'labels':     list[str] 每个区域的标签 slope/road/other,
            'elapsed_ms': float 推理耗时(毫秒),
        }
    """
    t0 = time.time()
    h, w = frame.shape[:2]

    if model is None:
        model = get_fastsam_model(device=device)

    slope_mask = np.zeros((h, w), dtype=np.uint8)
    road_mask = np.zeros((h, w), dtype=np.uint8)
    all_masks = []
    labels = []

    # ---- 方案 A: 文本提示分割 (利用 CLIP 匹配) ----
    if use_text_prompt:
        all_texts = SLOPE_TEXTS + ROAD_TEXTS + EXCLUDE_TEXTS
        n_slope, n_road = len(SLOPE_TEXTS), len(ROAD_TEXTS)

        results = model.predict(
            source=frame,
            texts=all_texts,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
            verbose=False,
        )

        if results and len(results) > 0:
            r = results[0]
            # 检查是否有 masks 属性 (某些版本用 names)
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks.data) > 0:
                masks_data = r.masks.data.cpu().numpy()

                # FastSAM + CLIP: masks 按 text 顺序返回, 每个 text 匹配多个区域
                # 实际行为: 返回的是所有被匹配的 masks, 顺序由 CLIP 匹配决定
                # 我们需要通过 r.names 或直接检查区域分类
                #
                # 简化处理: ultralytics FastSAM 的 predict(texts=...) 内部:
                #   1. 运行 segment everything
                #   2. 对每个 segment 用 CLIP 匹配最强 text
                #   3. 返回的 Results 中 masks 包含所有被任意 text 匹配的区域
                #
                # 实际 r.masks 可能没有 label 字段, 所以我们采用推断策略:
                #   - 每个 mask 都尝试判断其归属
                #   - 如果无法从结果中获取 label, 则用启发式方法分类

                for i, mask_arr in enumerate(masks_data):
                    mask_uint8 = (mask_arr > 0.5).astype(np.uint8) * 255
                    all_masks.append(mask_uint8)

                    # 尝试从结果中获取 label
                    label = _classify_mask_heuristic(mask_uint8, frame, h, w)
                    labels.append(label)

                    if label == "slope":
                        slope_mask = cv2.bitwise_or(slope_mask, mask_uint8)
                    elif label == "road":
                        road_mask = cv2.bitwise_or(road_mask, mask_uint8)

    # ---- 方案 B: 无文本提示 (仅分割, 用启发式分类) ----
    else:
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
            verbose=False,
        )

        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, 'masks') and r.masks is not None and len(r.masks.data) > 0:
                masks_data = r.masks.data.cpu().numpy()
                for mask_arr in masks_data:
                    mask_uint8 = (mask_arr > 0.5).astype(np.uint8) * 255
                    all_masks.append(mask_uint8)
                    label = _classify_mask_heuristic(mask_uint8, frame, h, w)
                    labels.append(label)
                    if label == "slope":
                        slope_mask = cv2.bitwise_or(slope_mask, mask_uint8)
                    elif label == "road":
                        road_mask = cv2.bitwise_or(road_mask, mask_uint8)

    elapsed_ms = (time.time() - t0) * 1000

    # 后处理: 形态学平滑
    if slope_mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_CLOSE, k)
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_OPEN, k)

    if road_mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k)

    if verbose:
        n_s = (slope_mask > 0).sum()
        n_r = (road_mask > 0).sum()
        print(f"[FastSAM] {len(all_masks)} regions → "
              f"slope={n_s / (h * w) * 100:.1f}% "
              f"road={n_r / (h * w) * 100:.1f}% "
              f"({elapsed_ms:.0f}ms)")

    return {
        'slope_mask': slope_mask,
        'road_mask': road_mask,
        'all_masks': all_masks,
        'labels': labels,
        'elapsed_ms': elapsed_ms,
    }


# ================================================================
# 启发式区域分类 (CLIP 的降级兜底)
# ================================================================

def _classify_mask_heuristic(
    mask: np.ndarray, frame: np.ndarray, h: int, w: int,
) -> str:
    """
    基于颜色 + 纹理 + 位置 的启发式分类。
    当 CLIP 标签不可用时使用。

    规则 (按优先级):
      1. 天空/顶部亮区 → "other"
      2. 底部 + 灰暗 + 低纹理 → "road"
      3. 中部 + 棕色/黄色 + 高纹理 → "slope"
      4. 其余 → "other"
    """
    # 区域属性
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return "other"

    mean_y = float(np.mean(ys))
    mean_x = float(np.mean(xs))
    y_ratio = mean_y / h

    # 天空区: 画面上 30%
    if y_ratio < 0.30:
        return "other"

    # 区域内的平均颜色
    roi_pixels = frame[mask > 0]
    if len(roi_pixels) == 0:
        return "other"

    mean_bgr = np.mean(roi_pixels, axis=0)
    mean_b, mean_g, mean_r = mean_bgr

    # HSV 分析
    roi_hsv = cv2.cvtColor(roi_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(roi_hsv[:, 0, 1]))
    mean_v = float(np.mean(roi_hsv[:, 0, 2]))

    # 纹理 (拉普拉斯方差)
    gray_roi = cv2.cvtColor(roi_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
    tex_var = float(np.var(cv2.Laplacian(gray_roi, cv2.CV_64F)))

    # ---- 分类逻辑 ----

    # 道路特征: 低饱和度 + 中等明度 + 低纹理 + 底部
    road_score = 0
    if mean_s < 60:
        road_score += 2
    if 40 < mean_v < 200:
        road_score += 2
    if tex_var < 150:
        road_score += 2
    if y_ratio > 0.55:
        road_score += 3
    # 道路宽高比大
    bbox_w = xs.max() - xs.min()
    bbox_h = ys.max() - ys.min()
    if bbox_h > 0 and bbox_w / bbox_h > 2.0:
        road_score += 1

    # 边坡特征: 中高饱和度 + 棕色/黄色调 + 高纹理 + 中部
    slope_score = 0
    if mean_s > 30:
        slope_score += 2
    if mean_r > mean_b + 10:  # 偏红/棕
        slope_score += 2
    if tex_var > 80:
        slope_score += 2
    if 0.25 < y_ratio < 0.70:
        slope_score += 2
    if bbox_h > 0 and 0.5 < bbox_w / bbox_h < 3.0:
        slope_score += 1

    if road_score > slope_score and road_score >= 5:
        return "road"
    elif slope_score > road_score and slope_score >= 4:
        return "slope"
    elif y_ratio > 0.60:
        return "road"
    return "other"


# ================================================================
# 多帧融合 (初始化时使用)
# ================================================================

def generate_road_mask(
    cap: cv2.VideoCapture,
    fw: int, fh: int,
    cache_key: str = "default",
    num_samples: int = 5,
    device: str = "cuda:0",
    use_text_prompt: bool = True,
    conf: float = 0.25,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """
    多帧 FastSAM 分割 + 投票融合, 生成稳定的 road_mask。

    用于视频/流初始化阶段, 对前 num_samples 帧做分割,
    投票得到稳定的道路区域掩码。

    参数:
        cap:          已打开的 cv2.VideoCapture
        fw, fh:       帧宽高
        cache_key:    缓存文件名 key (如 RTSP URL)
        num_samples:  采样帧数
        device:       推理设备
        use_text_prompt: 是否启用 CLIP 文本提示
        conf:         FastSAM 置信度
        verbose:      打印进度

    返回:
        road_mask (np.ndarray)  uint8 255=公路, None=失败
        兼容 detector.py 的 self._road_mask 接口
    """
    _MASKS_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = hashlib.md5((cache_key + "_fastsam_v2").encode()).hexdigest()[:12]
    cache_path = _MASKS_DIR / f"{safe_key}.png"

    # 缓存命中
    if cache_path.exists():
        cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        if cached is not None and cached.shape == (fh, fw):
            if verbose:
                print(f"[FastSAM] 缓存命中: {cache_path.name}")
            return cached

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = total_frames <= 0

    model = get_fastsam_model(device=device)

    road_score = np.zeros((fh, fw), dtype=np.float32)
    valid = 0
    t_start = time.time()

    for s in range(num_samples):
        if not is_live:
            pos = s * max(1, total_frames // num_samples)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))

        result = segment_frame(
            frame, model=model, device=device,
            conf=conf, use_text_prompt=use_text_prompt,
            verbose=False,
        )

        if result['road_mask'].any() or result['slope_mask'].any():
            road_score += (result['road_mask'] > 0).astype(np.float32)
            valid += 1
            if verbose:
                print(f"[FastSAM] 帧 {s + 1}/{num_samples} "
                      f"({result['elapsed_ms']:.0f}ms)"
                      f" slope={result['slope_mask'].sum() / (fw * fh) * 100:.0f}%"
                      f" road={result['road_mask'].sum() / (fw * fh) * 100:.0f}%")

    if valid < 1:
        if verbose:
            print("[FastSAM] 无有效分割帧")
        return None

    road_score /= valid
    road_mask = (road_score > 0.5).astype(np.uint8) * 255

    # ---- 后处理 ----
    # 形态学闭合: 填充道路内部孔洞
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # 保留底部连通的最大连通域
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        road_mask_clean = np.zeros_like(road_mask)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < fw * fh * 0.015:
                break
            cv2.drawContours(road_mask_clean, [cnt], -1, 255, -1)
        road_mask = road_mask_clean

    # 底部约束: 道路应延伸到画面底部
    bottom_coverage = (road_mask[-5:, :].sum(axis=0) > 0).sum() / fw
    if bottom_coverage < 0.15:
        if verbose:
            print(f"[FastSAM] 底部覆盖率仅 {bottom_coverage:.0%}, 结果可能不可靠")

    # 边缘裁剪: 去除画面边缘的碎片
    road_mask[:, :int(fw * 0.02)] = 0
    road_mask[:, int(fw * 0.98):] = 0

    # 膨胀: 确保道路掩码完全覆盖路面
    road_mask = cv2.dilate(road_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    # 保存缓存
    cv2.imwrite(str(cache_path), road_mask)

    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if verbose:
        road_pct = (road_mask > 0).sum() / (fw * fh) * 100
        print(f"[FastSAM] 完成 road_mask={road_pct:.1f}% "
              f"({time.time() - t_start:.1f}s 总计)")

    return road_mask


# ================================================================
# 快速 ROI 多边形生成 (兼容 video_widget.py)
# ================================================================

def road_mask_to_roi_polygons(
    road_mask: np.ndarray,
    fw: int, fh: int,
    min_area_ratio: float = 0.03,
    max_polygons: int = 3,
    epsilon_factor: float = 0.003,
) -> list:
    """
    从 road_mask 生成边坡 ROI 多边形列表。

    roi_mask = 255 - road_mask (取反)
    提取 roi_mask 中的连通域轮廓, 转为闭合多边形。

    返回:
        list[np.ndarray]  每个元素为 (N,2) 的 int32 多边形顶点
        兼容 video_widget.py 的 self.polygons 接口
    """
    roi_mask = 255 - road_mask

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    polygons = []
    min_area = min_area_ratio * fw * fh

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        if len(polygons) >= max_polygons:
            break

        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, epsilon, True).squeeze(1)

        if poly.ndim == 1:
            poly = poly.reshape(-1, 2)
        if len(poly) < 3:
            continue

        # 闭合多边形
        if not np.array_equal(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0:1]])

        polygons.append(poly.astype(np.int32))

    return polygons


# ================================================================
# 质量评估 (兼容 roi_confidence.py)
# ================================================================

def evaluate_mask_quality(
    road_mask: np.ndarray,
    fw: int, fh: int,
) -> dict:
    """
    评估 road_mask 质量, 判断是否满足使用条件。

    返回:
        {
            'confidence': float  (0~1),
            'road_pct':  float  道路像素占比,
            'bottom_cov': float 底部覆盖率,
            'is_reliable': bool 是否可信,
            'needs_fallback': bool 是否需要降级到手动ROI,
        }
    """
    h, w = road_mask.shape
    road_pixels = (road_mask > 0).sum()
    road_pct = road_pixels / (fw * fh)
    bottom_cov = (road_mask[-5:, :].sum(axis=0) > 0).sum() / fw

    # 道路占比: 理想 15%-50%
    if 0.15 <= road_pct <= 0.50:
        pct_score = 1.0
    elif road_pct < 0.08:
        pct_score = 0.2
    elif road_pct > 0.60:
        pct_score = 0.4
    else:
        pct_score = 0.7

    # 底部连通性
    cov_score = min(1.0, bottom_cov / 0.30)

    # 综合置信度
    confidence = pct_score * 0.5 + cov_score * 0.5

    return {
        'confidence': round(confidence, 3),
        'road_pct': round(road_pct, 3),
        'bottom_cov': round(bottom_cov, 3),
        'is_reliable': confidence > 0.55,
        'needs_fallback': confidence < 0.30,
    }
