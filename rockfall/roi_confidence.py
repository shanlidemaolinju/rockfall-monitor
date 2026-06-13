"""
ROI 质量评估与自适应降级
==========================
在启动阶段评估道路/边坡分割结果的可靠性, 决定使用自动分割还是回退手动ROI。

评估维度:
  1. 边界清晰度: 道路边界梯度强度
  2. 颜色一致性: 道路区域V通道方差
  3. 形状合理性: 多边形凸度
  4. 底部连通性: 道路是否延伸到底部
  5. 道路占比: 理想 20%-40%
  6. 天空占比: 惩罚含大量天空的分割
  7. 右半部分覆盖: 山区道路通常在画面右侧
  8. 上半部分占比: 边坡应在中上部
"""

import cv2
import numpy as np


def evaluate_roi_quality(
    road_mask: np.ndarray,
    frame: np.ndarray,
    polygon: np.ndarray,
) -> dict:
    """
    评估ROI分割质量, 输出置信度分数与降级建议。

    参数:
        road_mask: 道路掩码 (H, W) uint8, 255=道路
        frame:     原始BGR图像 (H, W, 3)
        polygon:   ROI多边形顶点 (N, 2) int32

    返回:
        {
            'confidence': float       # 综合置信度 (0~1)
            'details': {               # 各维度归一化得分
                'edge_sharpness': float,
                'color_consistency': float,
                'convexity': float,
                'bottom_connectivity': float,
                'road_pct_score': float,
                'sky_score': float,
                'right_coverage': float,
                'upper_half_ratio': float,
            },
            'is_reliable': bool       # 是否可信 (>0.65)
            'needs_fallback': bool     # 是否需要降级到手动ROI (<0.4)
        }
    """
    h, w = road_mask.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. 边界清晰度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    boundary_mask = cv2.Canny(road_mask, 100, 200)
    bgrad = mag[boundary_mask > 0]
    edge_sharp = np.median(bgrad) if len(bgrad) > 0 else 0

    # 2. 颜色一致性
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    road_v = hsv[:, :, 2][road_mask > 0]
    color_cons = 1.0 / (1.0 + np.std(road_v) / 50) if len(road_v) > 0 else 0

    # 3. 形状凸度
    if polygon is not None and len(polygon) >= 3:
        hull = cv2.convexHull(polygon)
        area_p = cv2.contourArea(polygon)
        area_h = cv2.contourArea(hull)
        convexity = area_p / area_h if area_h > 0 else 0
    else:
        convexity = 0

    # 4. 底部连通性
    bottom_cov = np.sum(road_mask[-5:, :] > 0) / (5 * w)

    # 5. 道路占比合理性 (理想 20%-40%)
    road_pct = np.sum(road_mask > 0) / (h * w)
    road_pct_score = max(0.0, 1.0 - abs(road_pct - 0.3) / 0.3)

    # 6. 天空占比惩罚
    if polygon is not None and len(polygon) >= 3:
        poly_area = cv2.contourArea(polygon)
        sky_ratio = 1 - (road_pct * h * w / poly_area) if poly_area > 0 else 1.0
        sky_score = max(0.0, 1.0 - sky_ratio / 0.3)
    else:
        sky_score = 0
        sky_ratio = 1.0

    # 新增: 右侧覆盖率 + 上半部分占比
    right_cov = np.sum(road_mask[:, int(w*0.7):])/(h*w*0.3)
    upper_ratio = np.sum(road_mask[:int(h*0.5),:])/max(np.sum(road_mask),1)

    ns = {
        'edge_sharpness': min(1.0, edge_sharp/100),
        'color_consistency': color_cons,
        'convexity': convexity,
        'bottom_connectivity': bottom_cov,
        'road_pct_score': road_pct_score,
        'sky_score': sky_score,
        'right_coverage': min(1.0, right_cov),
        'upper_half_ratio': min(1.0, upper_ratio),
    }
    total = (ns['right_coverage']*0.30 + ns['upper_half_ratio']*0.25 +
             ns['edge_sharpness']*0.15 + ns['convexity']*0.10 +
             ns['road_pct_score']*0.10 + ns['sky_score']*0.10)

    return {
        'confidence': round(total, 3),
        'details': ns,
        'is_reliable': total>0.65 and right_cov>0.5 and upper_ratio>0.4,
        'needs_fallback': total<0.4 or right_cov<0.3 or upper_ratio<0.3,
    }


def adaptive_params_by_confidence(params, confidence: float):
    """按置信度自适应调整参数"""
    if confidence < 0.4:
        params.saturation_max_road = 120
        params.morph_close_kernel = 15
    elif confidence > 0.7:
        params.saturation_max_road = 70
        params.morph_close_kernel = 7
    return params
