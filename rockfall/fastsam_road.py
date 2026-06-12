"""
FastSAM 道路/边坡分割模块（替代原SAM独立进程）
==================================================
策略: FastSAM 全图分割(边界精准) + 传统CV颜色/纹理分类(领域可靠)
      CLIP文本提示在灰岩/灰路场景区分度低，已弃用。

输出: road_mask(255=公路) 、roi_mask(255=边坡)
兼容原有视频流/图片调用逻辑
"""

import cv2
import numpy as np
from ultralytics import FastSAM

# ---- 全局模型单例 ----
_SAM_MODEL = None
_DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"


def _get_model() -> FastSAM:
    global _SAM_MODEL
    if _SAM_MODEL is None:
        _SAM_MODEL = FastSAM("FastSAM-x.pt")
    return _SAM_MODEL


def generate_road_slope_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    FastSAM 全图分割 → CV颜色/纹理/位置 分类 → road_mask + slope_mask。

    不使用 CLIP 文本提示 — 实测灰岩/灰路 embedding 距离太近，无效。
    """
    h, w = frame.shape[:2]
    model = _get_model()

    # ================================================================
    # Step 1: FastSAM 全图分割 (segment everything, 不限定文本)
    # ================================================================
    results = model.predict(
        source=frame,
        conf=0.20,
        iou=0.65,
        retina_masks=True,
        device=_DEVICE,
        verbose=False,
    )

    # ================================================================
    # Step 2: 提取所有 segment masks + 计算 CV 特征 + 分类
    # ================================================================
    slope_mask = np.zeros((h, w), dtype=np.uint8)
    road_mask = np.zeros((h, w), dtype=np.uint8)

    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        # FastSAM 无输出 → 降级：整帧按像素级CV分类
        return _pixel_level_cv_fallback(frame)

    masks_data = r.masks.data.cpu().numpy()

    # 预计算全帧特征（供每个segment使用）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    # 全帧统计量（对标原 road_detector 的自适应阈值）
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_a = float(np.mean(lab[:, :, 1]))
    lap_median = float(np.percentile(lap, 50))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_median = float(np.percentile(mag, 55))

    for mask_arr in masks_data:
        mask_uint8 = (mask_arr > 0.5).astype(np.uint8) * 255
        area = (mask_uint8 > 0).sum()

        # 跳过太小/太大的片段
        if area < h * w * 0.005 or area > h * w * 0.80:
            continue

        # 计算该segment的特征
        score = _segment_slope_score(
            mask_uint8, frame, hsv, gray, lab, lap, h, w,
            mean_s, mean_a, lap_median, mag_median,
        )

        # score > 0 → slope, score <= 0 → road/exclude
        if score > 0:
            slope_mask = cv2.bitwise_or(slope_mask, mask_uint8)
        else:
            road_mask = cv2.bitwise_or(road_mask, mask_uint8)

    # ================================================================
    # Step 3: 后处理
    # ================================================================
    # 安全网：如果 slope 还是太少，用像素级CV补充
    slope_pct = (slope_mask > 0).sum() / (h * w)
    if slope_pct < 0.15:
        _, slope_cv = _pixel_level_cv_fallback(frame)
        slope_mask = cv2.bitwise_or(slope_mask, slope_cv)

    # 形态学
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    if slope_mask.any():
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_CLOSE, k_close)
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_OPEN, k_open)
    if road_mask.any():
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k_close)

    # 重叠消解: 重叠区域按下部归路
    overlap = cv2.bitwise_and(slope_mask, road_mask)
    if overlap.any():
        ys, xs = np.where(overlap > 0)
        for y, x in zip(ys, xs):
            if y > h * 0.40:
                slope_mask[y, x] = 0
            else:
                road_mask[y, x] = 0

    # road_mask 至少覆盖非边坡
    if not road_mask.any():
        road_mask = cv2.bitwise_not(slope_mask)

    # 硬约束: road 最多占 50%
    road_pct = (road_mask > 0).sum() / (h * w)
    if road_pct > 0.50:
        road_mask = _keep_bottom_road(road_mask, h, w)

    return road_mask, slope_mask


def _segment_slope_score(
    mask_uint8, frame, hsv, gray, lab, lap, h, w,
    mean_s, mean_a, lap_median, mag_median,
) -> float:
    """
    对单个 FastSAM segment 做边坡 vs 公路的打分。

    返回: >0 → slope, <=0 → road/exclude

    特征对标原 road_detector.py generate_roi:
      color: s>mean_s*0.30, a>mean_a*0.7, v>20
      texture: lap_var > P50(lap)
      edge: mag > P55(mag)
      position: 路在底部, 坡在中上部
    """
    m = mask_uint8 > 0
    ys, xs = np.where(m)
    if len(ys) < 100:
        return -1.0

    mean_y = float(np.mean(ys))
    y_ratio = mean_y / h
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    # ---- 特征1: 颜色 (对标 road_detector slope_color) ----
    s_vals = hsv[:, :, 1][m]
    a_vals = lab[:, :, 1][m]
    v_vals = hsv[:, :, 2][m]

    s_ok = float(np.mean(s_vals > mean_s * 0.30))
    a_ok = float(np.mean(a_vals > mean_a * 0.7))
    v_ok = float(np.mean(v_vals > 20))
    color_score = (s_ok * 0.4 + a_ok * 0.3 + v_ok * 0.3)

    # ---- 特征2: 纹理 (对标 road_detector slope_texture) ----
    lap_vals = lap[m]
    tex_ok = float(np.mean(lap_vals > lap_median))
    tex_score = tex_ok

    # ---- 特征3: 边缘 (对标 road_detector slope_edge) ----
    # 复用已计算的 mag
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_vals = mag[m]
    edge_ok = float(np.mean(mag_vals > mag_median))
    edge_score = edge_ok

    # ---- 特征4: 饱和度 (道路=低饱和灰色, 边坡=有色) ----
    mean_s_seg = float(np.mean(s_vals))
    sat_score = min(1.0, mean_s_seg / max(mean_s, 1.0))

    # ---- 特征5: 位置 ----
    # 路在底部，坡在中上部 — 但必须验证底部确实"像公路"
    # 公路特征: 低饱和度(灰色沥青) + 低纹理(平整路面)
    # 如果该segment饱和度 >= 全帧均值，说明有色(植被/岩石)，不是公路
    # sat_score 已经计算了 mean_s_seg / mean_s，>1.0 说明比平均更"艳"
    bottom_looks_like_road = (
        y_ratio > 0.55
        and mean_s_seg < mean_s * 1.1     # 不比平均值更艳 → 灰调
        and tex_score < 0.55              # 纹理不过高 → 平整
    )
    if y_ratio > 0.70 and y_max > h * 0.85:
        pos_score = -1.0 if bottom_looks_like_road else 0.0
    elif y_ratio > 0.55:
        pos_score = -0.3 if bottom_looks_like_road else 0.2  # 中下部但不像路 → 中性偏坡
    elif y_ratio < 0.25:
        pos_score = -0.5   # 顶部 → 天空/远景，排除
    else:
        pos_score = 0.5    # 中部 → 边坡

    # ---- 特征6: 形状 ----
    bbox_h = y_max - y_min
    bbox_w = float(xs.max() - xs.min())
    if bbox_h > 0:
        aspect = bbox_w / bbox_h
        if aspect > 4.0 and y_ratio > 0.50:
            shape_score = -1.0  # 极宽扁 + 底部 → 公路
        else:
            shape_score = 0.0
    else:
        shape_score = 0.0

    # ---- 加权综合 ----
    total = (
        color_score * 0.30 +
        tex_score   * 0.20 +
        edge_score  * 0.15 +
        sat_score   * 0.10 +
        pos_score   * 0.20 +
        shape_score * 0.05
    )
    return total - 0.35  # 阈值偏移: >0 → slope


def _pixel_level_cv_fallback(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    FastSAM 完全失败时的降级：像素级 CV 分类。
    直接复用原 road_detector.py 的逻辑。
    """
    from rockfall.road_detector import generate_roi
    slope = generate_roi(frame)
    road = 255 - slope
    return road, slope


def _keep_bottom_road(road_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """硬约束：只保留与底部连通且不超过画面35%的道路"""
    result = np.zeros_like(road_mask)
    cutoff = int(h * 0.65)

    # 从底部向上 flood-fill
    visited = np.zeros((h, w), dtype=bool)
    for x in range(w):
        if road_mask[h - 1, x] > 0 and not visited[h - 1, x]:
            # BFS 向上扩展
            stack = [(h - 1, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < cutoff or visited[cy, cx] or road_mask[cy, cx] == 0:
                    continue
                visited[cy, cx] = True
                result[cy, cx] = 255
                for ny, nx in [(cy - 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny, nx))

    return result


def auto_segment_from_cap(
    cap: cv2.VideoCapture,
    fw: int, fh: int,
    sample_num: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从视频流采样多帧取平均掩码（提升稳定性）。

    参数:
        cap:     cv2.VideoCapture 视频流
        fw, fh:  帧宽高
        sample_num: 采样帧数

    返回:
        road_mask, roi_mask  (uint8 255=有效区域)
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = total_frames <= 0
    step = max(1, total_frames // sample_num) if not is_live else 1

    road_accum = np.zeros((fh, fw), dtype=np.float32)
    slope_accum = np.zeros((fh, fw), dtype=np.float32)
    valid = 0

    for s in range(sample_num):
        if not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s * step)
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))

        road, slope = generate_road_slope_mask(frame)
        road_accum += (road > 0).astype(np.float32)
        slope_accum += (slope > 0).astype(np.float32)
        valid += 1

    if valid == 0:
        return _default_masks(fw, fh)

    # 多帧投票
    road_mask = (road_accum / valid > 0.5).astype(np.uint8) * 255
    slope_mask = (slope_accum / valid > 0.5).astype(np.uint8) * 255

    # 未分类区域过大时（均匀斜坡场景FastSAM覆盖不全），用CV仅补空洞
    classified = ((road_mask > 0) | (slope_mask > 0)).sum()
    unclassified_pct = (1.0 - classified / (fw * fh)) * 100
    if unclassified_pct > 50 and valid > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, ref_frame = cap.read()
        if ret:
            if ref_frame.shape[1] != fw or ref_frame.shape[0] != fh:
                ref_frame = cv2.resize(ref_frame, (fw, fh))
            from rockfall.road_detector import generate_roi as gen_roi_cv
            cv_slope = gen_roi_cv(ref_frame)
            # 只在未分类区域补充CV结果，不覆盖FastSAM已有分类
            unclassified = np.where(road_mask == 0, 255 - slope_mask, 0)
            cv_fill = cv2.bitwise_and(cv_slope, unclassified)
            slope_mask = cv2.bitwise_or(slope_mask, cv_fill)

    # 后处理
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return road_mask, slope_mask


def _default_masks(fw: int, fh: int) -> tuple[np.ndarray, np.ndarray]:
    """采样失败时的默认ROI"""
    roi = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(roi, [np.array([
        [int(fw * 0.6), 0], [fw, 0], [fw, fh], [int(fw * 0.6), fh],
    ], np.int32)], 255)
    return cv2.bitwise_not(roi), roi


def release_model():
    """释放 FastSAM 显存"""
    global _SAM_MODEL
    if _SAM_MODEL is not None:
        del _SAM_MODEL
        _SAM_MODEL = None
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
