"""
种子演示数据 — 为比赛看板生成逼真的预警记录
============================================
插入不同等级、不同时段、多站点的预警记录，
让 Cockpit/预警记录/地图视图 都有内容展示。

运行: python scripts/seed_demo_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from datetime import datetime, timedelta
from rockfall.alert_store import get_alert_store

# ── 预设监测站点 (含经纬度) ──
SITES = [
    {"name": "钦州公路边坡监测点", "lat": 21.96, "lon": 108.62, "region": "广西·钦州", "highway": "G75 兰海高速"},
    {"name": "南宁那安快速路 1 号边坡", "lat": 22.817, "lon": 108.366, "region": "广西·南宁", "highway": "G7201 那安快速"},
    {"name": "崇左合那高速 2 号边坡", "lat": 22.379, "lon": 107.365, "region": "广西·崇左", "highway": "G7211 合那高速"},
    {"name": "防城港兰海高速 3 号边坡", "lat": 21.687, "lon": 108.355, "region": "广西·防城港", "highway": "G75 兰海高速"},
    {"name": "凭祥中越跨境公路 4 号边坡", "lat": 22.094, "lon": 106.767, "region": "广西·凭祥", "highway": "G322 跨境公路"},
]

# ── 四级预警分布权重 ──
LEVELS = ["blue", "yellow", "orange", "red"]
LEVEL_WEIGHTS = [40, 35, 18, 7]  # 大部分是低级别
LEVEL_CONF = {"blue": 0.35, "yellow": 0.60, "orange": 0.78, "red": 0.93}
LEVEL_DIAMETER = {"blue": 8, "yellow": 15, "orange": 25, "red": 38}
LEVEL_CLASSES = ["落石", "落石", "落石", "滑坡"]  # 偶尔混入滑坡

# ── 工单流转状态 ──
WORKFLOW_STATES = ["pending", "confirmed", "dispatched", "arrived", "handled", "archived", "false_alarm"]


def seed(record_count: int = 120):
    """生成 record_count 条分布在过去 30 天内的预警记录"""
    store = get_alert_store()
    now = datetime.now()

    # 先清空已有数据
    existing = store.count_alerts()
    if existing > 0:
        print(f"Current alerts: {existing}, appending new data...")

    inserted = 0
    for i in range(record_count):
        # 随机时间 (过去 30 天)
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        ts = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)

        # 随机站点
        site = random.choice(SITES)

        # 随机等级 (加权)
        level = random.choices(LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]

        # 随机检测数据
        conf = LEVEL_CONF[level] * random.uniform(0.85, 1.0)
        count = random.randint(1, 8) if level in ("red", "orange") else random.randint(1, 3)
        diameter = LEVEL_DIAMETER[level] * random.uniform(0.7, 1.3)
        class_name = "滑坡" if random.random() < 0.05 else "落石"

        # 随机工单状态
        wf_state = random.choices(
            WORKFLOW_STATES,
            weights=[15, 10, 8, 5, 30, 30, 2],  # 大部分已处理或归档
            k=1,
        )[0]

        # 插入
        rid = store.save_alert(
            count=count,
            max_confidence=round(conf, 4),
            track_ids=list(range(1, count + 1)),
            alert_level=level,
            class_summary=f"{class_name} ({site['highway']}段)",
            saved_frame=f"results/demo_alert_{i % 20}.jpg",
            clip_path="",
            push_status="sent" if level in ("red", "orange") else "pending",
            rock_diameter_cm=round(diameter, 1),
            monitoring_location=site["name"],
        )

        if rid > 0:
            # 设置工单状态 (通过直接更新)
            store.mark_review(rid, "confirmed" if wf_state != "false_alarm" and wf_state != "pending" else "", "")
            if wf_state == "false_alarm":
                store.mark_review(rid, "false_alarm", "误报: 施工车辆经过")
            inserted += 1

    # 额外插入几条今天的红色/橙色预警 (让大屏更好看)
    for level in ["red", "orange", "orange"]:
        site = random.choice(SITES)
        ts = now - timedelta(minutes=random.randint(5, 120))
        store.save_alert(
            count=random.randint(3, 10),
            max_confidence=round(0.92 if level == "red" else 0.78, 4),
            track_ids=list(range(1, random.randint(3, 8))),
            alert_level=level,
            class_summary=f"落石 ({site['highway']}段·紧急)",
            saved_frame="results/demo_alert_critical.jpg",
            clip_path="",
            push_status="sent",
            rock_diameter_cm=round(35.0 if level == "red" else 24.0, 1),
            monitoring_location=site["name"],
        )

    today = store.count_today_by_level()
    total = store.count_alerts()
    print(f"[OK] Inserted {inserted} demo alerts (total {total} in DB)")
    print(f"     Today: red={today['red']} orange={today['orange']} yellow={today['yellow']} blue={today['blue']}")


if __name__ == "__main__":
    seed()
