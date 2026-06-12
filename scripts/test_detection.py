"""
测试脚本 — 验证检测流水线
==========================
运行: python scripts/test_detection.py

测试内容:
  1. 模型加载
  2. 图片检测
  3. 图片检测
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rockfall.detector import RockDetector
from rockfall.config import MODEL_PATH


def main():
    print("=" * 60)
    print("落石检测系统 — 功能测试")
    print("=" * 60)

    # 测试1: 模型加载
    print(f"\n[1/3] 模型加载测试")
    print(f"  路径: {MODEL_PATH}")
    print(f"  存在: {Path(MODEL_PATH).exists()}")
    try:
        detector = RockDetector()
        print(f"  结果: ✓ 加载成功, 类别: {detector.model.names}")
    except Exception as e:
        print(f"  结果: ✗ 加载失败: {e}")
        return

    # 测试2: 图片检测
    print(f"\n[2/3] 图片检测测试")
    test_img = Path(__file__).resolve().parent.parent / "data" / "rock.jpg"
    if not test_img.exists():
        print(f"  跳过: 测试图片不存在 ({test_img})")
    else:
        result = detector.detect_image(str(test_img), push_alert=False)
        print(f"  结果: {result['detection']}")
        print(f"  数量: {result.get('count', 0)}")
        print(f"  置信度: {result.get('max_confidence', 'N/A')}")

    # 测试3: 视频检测
    print(f"\n[3/3] 视频检测测试")
    # 按优先级搜索视频目录: 项目内 data/videos/ → 外部试验视频
    search_dirs = [
        Path(__file__).resolve().parent.parent / "data" / "videos",
        Path("d:/rock/钦州落石试验视频/钦州落石试验视频"),
        Path("d:/rock/钦州落石试验视频"),
    ]
    videos = []
    for d in search_dirs:
        if d.exists():
            found = list(d.glob("*.mp4")) + list(d.glob("*.avi")) + list(d.glob("*.mov"))
            if found:
                videos = found
                break
    if videos:
        vid = str(videos[0])
        print(f"  视频: {Path(vid).name}")
        result = detector.detect_video(vid, save_frames=False, push_alerts=False)
        if isinstance(result, dict) and "error" not in result:
            print(f"  总帧数: {result.get('total_frames', 'N/A')}")
            print(f"  检出帧: {result.get('frames_with_detections', 'N/A')}")
            dets = result.get("detections", [])
            for d in dets[:3]:
                print(f"    帧{d['frame']}: {len(d['boxes'])}个目标")
        else:
            print(f"  错误: {result.get('error', result)}")
    else:
        print(f"  跳过: 未找到视频文件 (可将 .mp4 放入 data/videos/ 目录)")

    print(f"\n{'=' * 60}")
    print("测试完成")


if __name__ == "__main__":
    main()
