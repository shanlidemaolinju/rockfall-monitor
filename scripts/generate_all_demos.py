"""
批量演示数据生成脚本
====================
对注册在 DEMO_SCENES 中的所有场景批量运行检测并生成摘要数据。

用法:
  # 为单个场景生成
  python scripts/generate_all_demos.py /path/to/video.mp4 --scene nanning_naan_s1

  # 批量生成 — 从 JSON 配置文件读取视频→场景映射
  python scripts/generate_all_demos.py --config demo_videos.json

  # 列出所有已注册场景及其数据状态
  python scripts/generate_all_demos.py --list

配置文件格式 (demo_videos.json):
  {
    "nanning_naan_s1": "/data/videos/nanning_s1_day.mp4",
    "nanning_naan_s2": "/data/videos/nanning_s2_rain.mp4",
    "qinzhou_s1":      "/data/videos/qinzhou_coast.mp4",
    "guilin_g65_s1":   "/data/videos/guilin_night.mp4",
    "baise_s1":        "/data/videos/baise_backlight.mp4"
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

# ── 场景注册 (与 app.py 中 DEMO_SCENES 同步) ──
DEMO_SCENES = {
    "nanning_naan_s1": {
        "title": "南宁那安快速路 1 号边坡",
        "subtitle": "广西首府核心路段 — 晴天日间落石检测",
        "data_dir": "demo_data/nanning_naan_s1",
        "tags": ["晴天", "日间", "城市快速路"],
    },
    "nanning_naan_s2": {
        "title": "南宁那安快速路 2 号边坡",
        "subtitle": "雨天湿滑路面 — 可见度降低场景",
        "data_dir": "demo_data/nanning_naan_s2",
        "tags": ["雨天", "湿滑", "低可见度"],
    },
    "qinzhou_s1": {
        "title": "钦州滨海公路 1 号边坡",
        "subtitle": "北部湾沿海路段 — 风化岩体监测",
        "data_dir": "demo_data/qinzhou_s1",
        "tags": ["沿海", "风化", "盐雾"],
    },
    "guilin_g65_s1": {
        "title": "桂林 G65 包茂高速 K2485",
        "subtitle": "喀斯特地貌 — 夜间低光照落石检测",
        "data_dir": "demo_data/guilin_g65_s1",
        "tags": ["喀斯特", "夜间", "低光照"],
    },
    "baise_s1": {
        "title": "百色 G80 广昆高速 K780",
        "subtitle": "桂西山区 — 背光+遮挡复杂场景",
        "data_dir": "demo_data/baise_s1",
        "tags": ["山区", "背光", "遮挡"],
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
                        help="单个场景 ID (如 nanning_naan_s1)")
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
