"""
SAHI 切片辅助推理 — 高分辨率帧分块 + 批量推理
==============================================
将大帧切分为重叠的 slice_size × slice_size 瓦片,
批量送入 YOLO 推理 (远快于逐片串行), 最后重映射 + NMS 合并。

参考: SAHI (Slicing Aided Hyper Inference), Akyon et al. 2022
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SAHISlicer:
    """SAHI 切片器 — 计算切片坐标 + 重映射 + NMS 合并。

    支持动态切片数量限制：当切片数超过 max_slices 时自动降低
    overlap_ratio 或增大 slice_size，防止 GPU OOM。
    """

    def __init__(self, slice_size: int = 640, overlap_ratio: float = 0.2,
                 merge_iou: float = 0.5, enabled: bool = False,
                 max_slices: int = 16):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.merge_iou = merge_iou
        self.enabled = enabled
        self.max_slices = max_slices

    def get_slices(self, h: int, w: int,
                   max_slices: int | None = None) -> list:
        """生成覆盖 (h, w) 的切片坐标 [(x1,y1,x2,y2), ...]。

        若切片数超过 max_slices，自动降低 overlap 或增大 slice_size。
        """
        if not self.enabled:
            return [(0, 0, w, h)]

        limit = max_slices if max_slices is not None else self.max_slices

        # 第一轮：使用当前参数计算
        slices = self._compute_slices(h, w, self.slice_size, self.overlap_ratio)
        original_count = len(slices)

        if original_count <= limit:
            return slices

        # 策略 1：降低 overlap（减少重叠 = 减少切片数）
        reduced_overlap = max(0.05, self.overlap_ratio * 0.5)
        slices = self._compute_slices(h, w, self.slice_size, reduced_overlap)

        effective_size = self.slice_size
        if len(slices) > limit:
            # 策略 2：增大 slice_size
            effective_size = min(1280, self.slice_size * 2)
            slices = self._compute_slices(h, w, effective_size, reduced_overlap)

        if len(slices) != original_count:
            from .logger import log_event
            log_event("system", level="WARN",
                      msg=f"SAHI 切片数超限 ({original_count} > {limit}), "
                          f"已自动降为 {len(slices)} "
                          f"(overlap={reduced_overlap:.2f}, size={effective_size})")

        return slices

    @staticmethod
    def _compute_slices(h: int, w: int, slice_size: int,
                        overlap_ratio: float) -> list:
        """内部切片计算逻辑。"""
        stride = int(slice_size * (1 - overlap_ratio))
        if stride < 1:
            stride = 1
        slices = []
        y = 0
        while y < h:
            y2 = min(y + slice_size, h)
            y1 = max(0, y2 - slice_size)
            x = 0
            while x < w:
                x2 = min(x + slice_size, w)
                x1 = max(0, x2 - slice_size)
                slices.append((x1, y1, x2, y2))
                if x2 >= w:
                    break
                x += stride
            if y2 >= h:
                break
            y += stride
        return slices

    @staticmethod
    def remap_detections(tile_dets: list, origin: tuple) -> list:
        """将切片坐标重映射到全图坐标"""
        ox, oy = origin
        remapped = []
        for d in tile_dets:
            new_d = [d[0] + ox, d[1] + oy, d[2] + ox, d[3] + oy, d[4]]
            if len(d) > 5:
                new_d.append(d[5])
            remapped.append(new_d)
        return remapped

    @staticmethod
    def merge_detections(all_dets: list, iou_threshold: float = 0.5) -> list:
        """NMS 合并跨切片重叠检测框 — 优先 torchvision.ops.nms, 回退 CPU"""
        if len(all_dets) <= 1:
            return all_dets

        # 尝试 torchvision NMS (C++/CUDA 加速, 远超 Python 贪心)
        try:
            from torchvision.ops import nms as torch_nms
            boxes_t = torch.tensor([d[:4] for d in all_dets], dtype=torch.float32)
            scores_t = torch.tensor([d[4] for d in all_dets], dtype=torch.float32)
            keep = torch_nms(boxes_t, scores_t, iou_threshold)
            return [all_dets[i] for i in keep.tolist()]
        except ImportError:
            pass

        # 回退: 纯 Python 贪心 NMS
        boxes = np.array([d[:4] for d in all_dets], dtype=np.float32)
        confs = np.array([d[4] for d in all_dets])
        order = confs.argsort()[::-1]

        keep = []
        suppressed = set()
        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            for j in order:
                if j == i or j in suppressed:
                    continue
                iou = SAHISlicer._box_iou(boxes[i], boxes[j])
                if iou >= iou_threshold:
                    suppressed.add(j)
        return [all_dets[i] for i in keep]

    @staticmethod
    def _box_iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)


def sahi_inference(model, frame: np.ndarray, slicer: SAHISlicer, conf: float = 0.3) -> list:
    """
    SAHI 推理: 切片 → 批量 YOLO → 重映射 → NMS 合并。

    批量推理模式下, 所有切片堆叠为 (N, 3, H, W) 张量一次性送入模型,
    相比逐片串行推理快 2-4x (减少 Python↔CUDA 往返次数)。

    批量推理 OOM 时自动回退到逐片推理, 并动态降低 max_slices。
    """
    h, w = frame.shape[:2]
    slices = slicer.get_slices(h, w)

    if not slices:
        return []

    if HAS_TORCH and len(slices) > 1:
        tiles = []
        for x1, y1, x2, y2 in slices:
            tile = frame[y1:y2, x1:x2]
            tiles.append(tile)

        try:
            batch_results = model(tiles, conf=conf, imgsz=slicer.slice_size, verbose=False)
            all_dets = _process_batch_results(batch_results, slices, slicer)
            return slicer.merge_detections(all_dets, slicer.merge_iou)
        except RuntimeError as e:
            # OOM 或 batch 推理失败 → 动态降低 max_slices 并回退逐片
            new_max = max(4, len(slices) // 2)
            slicer.max_slices = new_max
            from .logger import log_event
            log_event("system", level="WARN",
                      msg=f"SAHI 批量推理 OOM, max_slices 降为 {new_max}, 回退逐片: {e}")
        except Exception as e:
            from .logger import log_event
            log_event("system", level="ERROR",
                      msg=f"SAHI 批量推理异常: {e}")
            return []

    # 单切片 / torch 不可用 / 批量回退 → 逐片推理
    all_dets = []
    for x1, y1, x2, y2 in slices:
        tile = frame[y1:y2, x1:x2]
        try:
            results = model(tile, conf=conf, imgsz=slicer.slice_size, verbose=False)
            tile_dets = []
            for r in results:
                if r.boxes is not None:
                    for b in r.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].int().tolist()
                        tile_dets.append([bx1, by1, bx2, by2, b.conf[0].item(), int(b.cls[0].item())])
            remapped = slicer.remap_detections(tile_dets, (x1, y1))
            all_dets.extend(remapped)
        except Exception as e:
            from .logger import log_event
            log_event("system", level="WARN",
                      msg=f"SAHI 切片推理失败: {e}")
            continue

    if not all_dets:
        return []

    return slicer.merge_detections(all_dets, slicer.merge_iou)


def _process_batch_results(batch_results, slices, slicer) -> list:
    """从批量推理结果提取检测框并重映射到全图坐标"""
    all_dets = []
    for (x1, y1, x2, y2), r in zip(slices, batch_results):
        tile_dets = []
        if r.boxes is not None:
            for b in r.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].int().tolist()
                tile_dets.append([bx1, by1, bx2, by2, b.conf[0].item(), int(b.cls[0].item())])
        remapped = slicer.remap_detections(tile_dets, (x1, y1))
        all_dets.extend(remapped)
    return all_dets
