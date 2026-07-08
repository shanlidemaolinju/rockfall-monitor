"""
批量演示数据生成脚本
====================
对注册在 DEMO_SCENES 中的所有场景批量运行检测并生成摘要数据。

用法:
  # 为单个场景生成
  python scripts/generate_all_demos.py /path/to/video.mp4 --scene qinzhou_demo

  # 批量生成 — 从 JSON 配置文件读取视频→场景映射
  python scripts/generate_all_demos.py --config demo_videos.json

  # 列出所有已注册场景及其数据状态
  python scripts/generate_all_demos.py --list

配置文件格式 (demo_videos.json):
  {
    "qinzhou_demo":  "/data/videos/qinzhou_demo.mp4",
    "qinzhou_cam_1": "/data/videos/1.flv",
    "qinzhou_cam_2": "/data/videos/2.flv",
    "qinzhou_cam_3": "/data/videos/3.flv",
    "qinzhou_cam_4": "/data/videos/4.flv",
    "qinzhou_cam_5": "/data/videos/5.flv",
    "qinzhou_cam_6": "/data/videos/6.flv",
    "qinzhou_test_a": "/data/videos/VID_20230304_114247.mp4",
    "qinzhou_test_b": "/data/videos/VID_20230304_153140.mp4",
    "qinzhou_test_c": "/data/videos/VID_20230304131810.mp4",
    "qinzhou_test_d": "/data/videos/VID_20230304132003.mp4",
    "qinzhou_test_e": "/data/videos/VID_20230304132126.mp4",
    "yibin_s1":      "/data/videos/yibin_landslide.mp4"
  }

依赖: 需要 GPU 环境 (或足够 CPU 算力)
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

# ── 项目根路径 ──
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent

# ── 场景注册 (与 _shared.py 中 DEMO_SCENES 同步) ──
DEMO_SCENES = {
    "qinzhou_demo": {
        "title": "钦州示范视频",
        "subtitle": "钦州公路边坡 — 日间晴好天气落石检测",
        "data_dir": "demo_data/nanning_naan_s1",
        "tags": ["钦州", "日间", "示范"],
    },
    "qinzhou_cam_2": {
        "title": "钦州监测视频 2",
        "subtitle": "钦州公路边坡实时监测 — 2.flv",
        "data_dir": "demo_data/nanning_naan_s2",
        "tags": ["钦州", "监测摄像头", "实时"],
    },
    "qinzhou_cam_3": {
        "title": "钦州监测视频 3",
        "subtitle": "钦州公路边坡实时监测 — 3.flv",
        "data_dir": "demo_data/guilin_g65_s1",
        "tags": ["钦州", "监测摄像头", "实时"],
    },
    "qinzhou_test_a": {
        "title": "钦州落石试验 A",
        "subtitle": "钦州现场落石试验 — VID_20230304_114247.mp4",
        "data_dir": "demo_data/baise_s1",
        "tags": ["钦州", "落石试验", "现场"],
    },
    "qinzhou_test_b": {
        "title": "钦州落石试验 B",
        "subtitle": "钦州现场落石试验 — VID_20230304_153140.mp4",
        "data_dir": "demo_data/qinzhou_s1",
        "tags": ["钦州", "落石试验", "现场"],
    },
    # ── 以下场景已有视频源，待 GPU 生成 ──
    "qinzhou_cam_1": {
        "title": "钦州监测视频 1",
        "subtitle": "钦州公路边坡实时监测 — 1.flv",
        "data_dir": "demo_data/qinzhou_cam_1",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_4": {
        "title": "钦州监测视频 4",
        "subtitle": "钦州公路边坡实时监测 — 4.flv",
        "data_dir": "demo_data/qinzhou_cam_4",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_5": {
        "title": "钦州监测视频 5",
        "subtitle": "钦州公路边坡实时监测 — 5.flv",
        "data_dir": "demo_data/qinzhou_cam_5",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_6": {
        "title": "钦州监测视频 6",
        "subtitle": "钦州公路边坡实时监测 — 6.flv",
        "data_dir": "demo_data/qinzhou_cam_6",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_test_c": {
        "title": "钦州落石试验 C",
        "subtitle": "钦州现场落石试验 — VID_20230304131810.mp4",
        "data_dir": "demo_data/qinzhou_test_c",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "qinzhou_test_d": {
        "title": "钦州落石试验 D",
        "subtitle": "钦州现场落石试验 — VID_20230304132003.mp4",
        "data_dir": "demo_data/qinzhou_test_d",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "qinzhou_test_e": {
        "title": "钦州落石试验 E",
        "subtitle": "钦州现场落石试验 — VID_20230304132126.mp4",
        "data_dir": "demo_data/qinzhou_test_e",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "yibin_s1": {
        "title": "宜宾 G85 渝昆高速滑坡",
        "subtitle": "四川盆地南缘 — 前兆小落石→红色预警→大规模崩塌 (43秒)",
        "data_dir": "demo_data/yibin_s1",
        "tags": ["宜宾", "滑坡", "前兆预警", "红色升级"],
    },
}


def check_data_status() -> dict[str, dict]:
    """检查各场景数据状态。返回 {sid: {exists, summary, frames_count}}"""
    status = {}
    for sid, scene in DEMO_SCENES.items():
        data_dir = _ROOT_DIR / scene["data_dir"]
        summary_path = data_dir / "summary.json"
        frames_dir = data_dir / "frames"

        entry = {
            "title": scene["title"],
            "data_dir": str(data_dir),
            "exists": data_dir.exists(),
        }
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                entry["summary"] = summary.get("alerts", {})
                entry["elapsed_sec"] = summary.get("detection", {}).get("elapsed_sec", 0)
                entry["video"] = summary.get("video", {}).get("file", "?")
            except Exception:
                entry["summary"] = None
        else:
            entry["summary"] = None

        entry["frames_count"] = (
            len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        )
        status[sid] = entry
    return status


def list_scenes():
    """列出所有场景及数据状态。"""
    status = check_data_status()
    print(f"\n{'='*70}")
    print(f"  RockGuard Demo Scenes ({len(DEMO_SCENES)} registered)")
    print(f"{'='*70}")
    print()

    ready = 0
    for sid, info in status.items():
        has_data = info["summary"] is not None
        icon = "[OK]" if has_data else "[  ]"
        if has_data:
            ready += 1

        print(f"  {icon} {sid}")
        print(f"     {info['title']}")
        if has_data:
            alerts = info["summary"]
            print(f"     视频: {info['video']}  |  {info['elapsed_sec']:.0f}s")
            print(f"     预警: R{alerts.get('red',0)} O{alerts.get('orange',0)} "
                  f"Y{alerts.get('yellow',0)} B{alerts.get('blue',0)}  "
                  f"关键帧: {info['frames_count']}")
        else:
            print(f"     [未生成] {info['data_dir']}/")
        print()

    print(f"{'='*70}")
    print(f"  {ready}/{len(DEMO_SCENES)} scenes ready for demo")
    print()

    if ready < len(DEMO_SCENES):
        print("生成命令:")
        print(f"  python scripts/generate_all_demos.py --config demo_videos.json")
        print()


def generate_scene(scene_id: str, video_path: str,
                   max_frames: int = 300, stride: int = 2,
                   img_size: int = 640, conf: float | None = None) -> bool:
    """调用 generate_demo.py 为单个场景生成数据。"""
    scene = DEMO_SCENES.get(scene_id)
    if not scene:
        print(f"[X] 未知场景: {scene_id}")
        return False

    video = Path(video_path)
    if not video.exists():
        print(f"[X] 视频不存在: {video_path}")
        return False

    print(f"\n{'─'*50}")
    print(f"[*] {scene['title']}")
    print(f"    {scene['subtitle']}")
    print(f"    video: {video_path}")
    print(f"    output: {_ROOT_DIR / scene['data_dir']}")
    print(f"{'─'*50}")

    cmd = [
        sys.executable,
        str(_THIS_DIR / "generate_demo.py"),
        str(video_path),
        "--name", scene_id,
        "--max-frames", str(max_frames),
        "--stride", str(stride),
        "--img-size", str(img_size),
        "--out", str(_ROOT_DIR / scene["data_dir"]),
    ]
    if conf is not None:
        cmd.extend(["--conf", str(conf)])

    result = subprocess.run(cmd, cwd=str(_ROOT_DIR))
    success = result.returncode == 0

    if success:
        print(f"   [OK] {scene_id} 生成成功")
    else:
        print(f"   [FAIL] {scene_id} 生成失败 (exit code {result.returncode})")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="RockGuard 批量演示数据生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?", help="输入视频路径 (--scene 模式时必填)")
    parser.add_argument("--scene", default=None,
                        help="单个场景 ID (如 qinzhou_demo)")
    parser.add_argument("--config", default=None,
                        help="JSON 配置文件路径 (批量模式)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="每条视频最多推理帧数 (默认 300)")
    parser.add_argument("--stride", type=int, default=2,
                        help="帧采样步长 (默认 2)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="推理分辨率 (默认 640)")
    parser.add_argument("--conf", type=float, default=0.08,
                        help="检测置信度阈值 (默认 0.08, 配置文件默认 0.30)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有场景及数据状态")
    parser.add_argument("--create-config", default=None,
                        help="生成配置文件模板 (输出路径)")
    args = parser.parse_args()

    # ── 列出场景 ──
    if args.list:
        list_scenes()
        return

    # ── 生成配置模板 ──
    if args.create_config:
        template = {}
        for sid, scene in DEMO_SCENES.items():
            template[sid] = f"/path/to/video/{sid}.mp4  # {scene['title']}"

        config_path = Path(args.create_config)
        config_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] 配置模板已生成: {config_path}")
        print(f"   请编辑文件填入实际视频路径后运行:")
        print(f"   python scripts/generate_all_demos.py --config {config_path}")
        return

    # ── 单场景生成 ──
    if args.scene:
        if not args.video:
            parser.error("--scene 模式需要指定 video 参数")
        success = generate_scene(
            args.scene, args.video,
            max_frames=args.max_frames, stride=args.stride,
            img_size=args.img_size, conf=args.conf,
        )
        sys.exit(0 if success else 1)

    # ── 批量生成 (JSON 配置) ──
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[X] 配置文件不存在: {config_path}")
            print(f"   生成模板: python scripts/generate_all_demos.py --create-config {config_path}")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            video_map = json.load(f)

        results = {}
        for sid, video_path in video_map.items():
            if sid not in DEMO_SCENES:
                print(f"[WARN] 跳过未知场景: {sid}")
                continue
            results[sid] = generate_scene(
                sid, video_path,
                max_frames=args.max_frames, stride=args.stride,
                img_size=args.img_size, conf=args.conf,
            )

        # 汇总
        success_count = sum(1 for v in results.values() if v)
        print(f"\n{'='*50}")
        print(f"  批量生成完成: {success_count}/{len(results)} 成功")
        for sid, ok in results.items():
            print(f"  {'[OK]' if ok else '[FAIL]'} {sid}")
        sys.exit(0 if success_count == len(results) else 1)

    # ── 无参数: 打印帮助 ──
    parser.print_help()
    print()
    list_scenes()


if __name__ == "__main__":
    main()
