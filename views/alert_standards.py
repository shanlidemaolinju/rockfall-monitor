"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def _esc(text: str) -> str:
    """转义 HTML 特殊字符，防止渲染为代码。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _get_alert_standards():
    """从 alert_classifier 获取四级预警标准 (单例缓存, 降级时用内置默认值)。

    关键: _esc() 在两条代码路径中都统一应用，防止未转义的 < > 被
    unsafe_allow_html=True 的 markdown 渲染为 HTML 标签而显示"原代码"。
    """
    if get_response_workflow is not None:
        try:
            workflows = {
                "red": get_response_workflow("red"),
                "orange": get_response_workflow("orange"),
                "yellow": get_response_workflow("yellow"),
                "blue": get_response_workflow("blue"),
            }
            return {
                k: {
                    "level": v["label"].split(" ", 1)[1] if " " in v["label"] else v["label"],
                    "icon": {"red": "🔴", "orange": "🟠", "yellow": "🟡", "blue": "🔵"}[k],
                    "color": {"red": "#D32F2F", "orange": "#E65100", "yellow": "#F9A825", "blue": "#1565C0"}[k],
                    "bg": {"red": "#FFEBEE", "orange": "#FFF3E0", "yellow": "#FFFDE7", "blue": "#E3F2FD"}[k],
                    "trigger": _esc(" 或 ".join(v["trigger_conditions"])),
                    "response": [_esc(step) for step in v["disposal_steps"]],
                    "push_channels": [_PUSH_CHANNEL_DISPLAY.get(ch, ch) for ch in v["push_channels"]],
                    "push_content": _get_push_content_template(k),
                    "cooldown": {"red": "30秒", "orange": "60秒", "yellow": "120秒", "blue": "—"}[k],
                    "requires_sound": v["requires_sound"],
                }
                for k, v in workflows.items()
            }
        except Exception:
            pass
    # 降级: 内置默认值 — 同样应用 _esc 确保 HTML 安全
    return {
        k: {
            **v,
            "trigger": _esc(v["trigger"]),
            "response": [_esc(step) for step in v["response"]],
        }
        for k, v in _BUILTIN_ALERT_STANDARDS.items()
    }

def _get_push_content_template(level: str) -> str:
    """推送内容模板"""
    templates = {
        "red": "【I级预警】时间+地点+落石数量+最大直径+置信度+处置建议",
        "orange": "【II级预警】时间+地点+落石数量+直径+处置建议",
        "yellow": "【III级预警】时间+地点+置信度 — 自动记录",
        "blue": "无推送 — 仅本地数据库记录",
    }
    return templates.get(level, "")

def _safe_html_dedent(text: str) -> str:
    """去除公共缩进后，确保每行无 ≥4 空格前缀，避免触发 Markdown 代码块渲染。"""
    dedented = textwrap.dedent(text)
    lines = dedented.split("\n")
    # 去除每行首部空白——HTML 不需要缩进
    stripped = [line.lstrip() for line in lines]
    return "\n".join(stripped)

# 推送渠道内部标识符 → 用户显示名称映射
_PUSH_CHANNEL_DISPLAY = {
    "pushplus": "微信 (PushPlus)",
    "smtp": "邮件 (SMTP)",
    "wecom": "企业微信 (Webhook)",
    "dingtalk": "钉钉 (Webhook)",
    "feishu": "飞书 (Webhook)",
    "sse": "界面弹窗 (SSE)",
    "sms": "短信",
    "phone": "电话",
}

_BUILTIN_ALERT_STANDARDS = {
    "red": {
        "level": "I 级 · 特别严重",
        "icon": "🔴",
        "color": "#D32F2F",
        "bg": "#FFEBEE",
        "trigger": "置信度 > 0.90 或 落石直径 > 30cm 或 检测到坠落状态",
        "response": [
            "立即通知公路管理部门封闭相关车道",
            "电话通知值班领导 (5分钟内响应)",
            "通知交警部门协助交通管制",
            "调取现场实时画面确认灾情",
            "启动应急预案，派遣巡查人员",
        ],
        "push_channels": ["微信 (PushPlus)", "短信", "电话"],
        "push_content": "【I级预警】时间+地点+落石数量+最大直径+置信度+处置建议",
        "cooldown": "30秒",
    },
    "orange": {
        "level": "II 级 · 严重",
        "icon": "🟠",
        "color": "#E65100",
        "bg": "#FFF3E0",
        "trigger": "置信度 0.70-0.90 或 落石直径 20-30cm 或 检测到滚动状态",
        "response": [
            "通知公路管理部门关注",
            "微信推送预警信息给值班人员",
            "建议限速通行 (≤40km/h)",
            "安排人员30分钟内到场巡查",
            "加密监测频率至 5fps",
        ],
        "push_channels": ["微信 (PushPlus)", "邮件"],
        "push_content": "【II级预警】时间+地点+落石数量+直径+处置建议",
        "cooldown": "60秒",
    },
    "yellow": {
        "level": "III 级 · 较重",
        "icon": "🟡",
        "color": "#F9A825",
        "bg": "#FFFDE7",
        "trigger": "置信度 0.50-0.70 或 落石直径 10-20cm",
        "response": [
            "系统自动记录预警事件",
            "触发界面黄色弹窗提醒",
            "关注后续帧是否有升级趋势",
            "纳入日报汇总",
        ],
        "push_channels": ["界面弹窗 (SSE)"],
        "push_content": "【III级预警】时间+地点+置信度 — 自动记录",
        "cooldown": "120秒",
    },
    "blue": {
        "level": "IV 级 · 一般",
        "icon": "🔵",
        "color": "#1565C0",
        "bg": "#E3F2FD",
        "trigger": "置信度 0.30-0.50 或 落石直径 < 10cm",
        "response": [
            "静默记录至本地数据库",
            "不触发主动通知推送",
            "用于历史趋势分析",
        ],
        "push_channels": ["无 (仅本地记录)"],
        "push_content": "无推送 — 仅本地数据库记录",
        "cooldown": "—",
    },
}

def page_alert_standards():
    """预警标准文档化页面: 四级预警触发条件 + 决策树 + 响应流程"""
    standards = _get_alert_standards()  # 数据来源: alert_classifier.py (单例缓存)

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">预警分级标准</div>
            <div style="font-size:0.8rem;opacity:0.85;">四级体系 &middot; 决策树 &middot; 响应流程 &middot; 推送渠道</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span><span>{TEAM_NAME}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # Section 1: Four-Level Overview
    # ══════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:4px;height:24px;background:#1565C0;border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1.1rem;color:#1B2838;">四级预警体系</div>
        <div style="font-size:0.78rem;color:#5F6B7A;">对齐《公路自然灾害监测预警系统技术指南》</div>
    </div>
    """, unsafe_allow_html=True)

    # 四级卡片概览
    level_cols = st.columns(4)
    for col, (key, std) in zip(level_cols, standards.items()):
        with col:
            st.markdown(f"""
            <div style="padding:0.75rem;background:{std['bg']};border:2px solid {std['color']};
                        border-radius:10px;text-align:center;min-height:180px;">
                <div style="font-size:1.8rem;">{std['icon']}</div>
                <div style="font-weight:700;font-size:0.85rem;color:{std['color']};margin:0.3rem 0;">
                    {std['level']}
                </div>
                <div style="font-size:0.7rem;color:#5F6B7A;line-height:1.5;">
                    {std['trigger'][:60]}...
                </div>
                <div style="margin-top:0.4rem;font-size:0.65rem;color:{std['color']};font-weight:600;">
                    推送: {std['push_channels'][0] if std['push_channels'] else '不推送'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # Section 2: Decision Tree
    # ══════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.75rem;">
        <div style="width:4px;height:24px;background:#E65100;border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1.1rem;color:#1B2838;">预警分级决策树</div>
    </div>
    """, unsafe_allow_html=True)

    # 决策树 HTML 可视化 (CSS 已移至全局样式块)
    st.markdown(_safe_html_dedent("""\
    <div class="tree-container">
      <div class="tree-root">

        <!-- ROOT -->
        <div class="tree-node root">输入: 检测帧 + 跟踪轨迹</div>
        <div class="tree-arrow">▼</div>

        <!-- Decision 1: Confidence -->
        <div class="tree-node decision">最高置信度 max_conf ?</div>
        <div class="tree-branch">
          <div style="text-align:center;">
            <div class="tree-label">&gt; 0.90</div>
            <div class="tree-node leaf-red">🔴 I 级</div>
          </div>
          <div style="text-align:center;">
            <div class="tree-label">0.70 - 0.90</div>
            <div class="tree-arrow">▼</div>
            <div class="tree-node decision">落石直径 ?</div>
            <div class="tree-branch">
              <div style="text-align:center;">
                <div class="tree-label">&gt; 30cm</div>
                <div class="tree-node leaf-red">🔴 I 级 (升级)</div>
              </div>
              <div style="text-align:center;">
                <div class="tree-label">20-30cm</div>
                <div class="tree-node leaf-orange">🟠 II 级</div>
              </div>
              <div style="text-align:center;">
                <div class="tree-label">&lt; 20cm</div>
                <div class="tree-arrow">▼</div>
                <div class="tree-node decision">运动状态 ?</div>
                <div class="tree-branch">
                  <div style="text-align:center;">
                    <div class="tree-label">坠落</div>
                    <div class="tree-node leaf-orange">🟠 II 级</div>
                  </div>
                  <div style="text-align:center;">
                    <div class="tree-label">滚动</div>
                    <div class="tree-node leaf-yellow">🟡 III 级</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div style="text-align:center;">
            <div class="tree-label">0.50 - 0.70</div>
            <div class="tree-arrow">▼</div>
            <div class="tree-node decision">落石直径 ?</div>
            <div class="tree-branch">
              <div style="text-align:center;">
                <div class="tree-label">&gt; 20cm</div>
                <div class="tree-node leaf-orange">🟠 II 级 (升级)</div>
              </div>
              <div style="text-align:center;">
                <div class="tree-label">10-20cm</div>
                <div class="tree-node leaf-yellow">🟡 III 级</div>
              </div>
              <div style="text-align:center;">
                <div class="tree-label">&lt; 10cm</div>
                <div class="tree-arrow">▼</div>
                <div class="tree-node decision">持续帧数 ?</div>
                <div class="tree-branch">
                  <div style="text-align:center;">
                    <div class="tree-label">&gt; 10帧</div>
                    <div class="tree-node leaf-yellow">🟡 III 级</div>
                  </div>
                  <div style="text-align:center;">
                    <div class="tree-label">&lt; 10帧</div>
                    <div class="tree-node leaf-blue">🔵 IV 级</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div style="text-align:center;">
            <div class="tree-label">0.30 - 0.50</div>
            <div class="tree-arrow">▼</div>
            <div class="tree-node leaf-blue">🔵 IV 级</div>
            <div class="tree-label" style="margin-top:0.25rem;">直径 &gt; 10cm → 升级至 III 级</div>
          </div>
        </div>

        <!-- Decision Final -->
        <div style="text-align:center;margin-top:0.5rem;">
          <div class="tree-label">&lt; 0.30</div>
          <div class="tree-node leaf-green">🟢 正常 (不预警)</div>
        </div>

      </div>
    </div>\
    """), unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # Section 3: Detailed Standards Table
    # ══════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:4px;height:24px;background:#2E7D32;border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1.1rem;color:#1B2838;">响应流程与推送配置</div>
    </div>
    """, unsafe_allow_html=True)

    # 选择等级查看详情
    selected_level = st.selectbox(
        "选择预警等级查看详情",
        options=list(standards.keys()),
        format_func=lambda k: f"{standards[k]['icon']} {standards[k]['level']}",
    )

    std = standards[selected_level]

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(_safe_html_dedent(f"""\
        <div class="card" style="border-left:4px solid {std['color']};">
            <div style="font-weight:700;font-size:1rem;color:{std['color']};margin-bottom:0.5rem;">
                {std['icon']} {std['level']}
            </div>

            <div style="margin-bottom:0.75rem;">
                <div style="font-weight:600;font-size:0.8rem;color:#D32F2F;margin-bottom:0.2rem;">触发条件</div>
                <div style="font-size:0.8rem;color:#5F6B7A;padding:0.4rem 0.6rem;background:{std['bg']};
                            border-radius:6px;">{std['trigger']}</div>
            </div>

            <div style="margin-bottom:0.75rem;">
                <div style="font-weight:600;font-size:0.8rem;color:#2E7D32;margin-bottom:0.2rem;">处置流程</div>
                <div style="font-size:0.78rem;color:#5F6B7A;line-height:1.8;">
                    {"".join(f'<div>• {step}</div>' for step in std['response'])}
                </div>
            </div>
        </div>\
        """), unsafe_allow_html=True)

    with col_b:
        st.markdown(_safe_html_dedent(f"""\
        <div class="card">
            <div style="font-weight:600;font-size:0.85rem;color:#1B2838;margin-bottom:0.5rem;">推送配置</div>
            <div style="font-size:0.78rem;color:#5F6B7A;line-height:1.8;">
                <b>推送渠道</b>:<br>
                {"".join(f'<div style="padding-left:0.5rem;">• {ch}</div>' for ch in std['push_channels'])}
                <br><b>推送内容模板</b>:<br>
                <div style="padding:0.4rem 0.6rem;background:#F5F7FA;border-radius:6px;
                            font-size:0.72rem;margin-top:0.2rem;">{std['push_content']}</div>
                <br><b>冷却时间</b>: {std['cooldown']}
            </div>
        </div>

        <div class="card">
            <div style="font-weight:600;font-size:0.85rem;color:#1B2838;margin-bottom:0.5rem;">分级阈值速查</div>
            <table style="width:100%;font-size:0.72rem;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E3E8EF;">
                    <th style="text-align:left;padding:0.3rem;">等级</th>
                    <th style="text-align:right;padding:0.3rem;">置信度</th>
                    <th style="text-align:right;padding:0.3rem;">直径</th>
                </tr>\
        """), unsafe_allow_html=True)
        for k, s in standards.items():
            conf_range = { "red": "&gt; 0.90", "orange": "0.70-0.90", "yellow": "0.50-0.70", "blue": "0.30-0.50" }[k]
            diam_range = { "red": "&gt; 30cm", "orange": "20-30cm", "yellow": "10-20cm", "blue": "&lt; 10cm" }[k]
            st.markdown(_safe_html_dedent(f"""\
            <tr style="border-bottom:1px solid #E3E8EF;">
                <td style="padding:0.3rem;color:{s['color']};font-weight:600;">{s['icon']} {k.upper()}</td>
                <td style="text-align:right;padding:0.3rem;">{conf_range}</td>
                <td style="text-align:right;padding:0.3rem;">{diam_range}</td>
            </tr>\
            """), unsafe_allow_html=True)
        st.markdown("</table></div>", unsafe_allow_html=True)

    # 升级规则
    st.markdown(_safe_html_dedent("""\
    <div class="card" style="border-left:3px solid #D32F2F;margin-top:0.5rem;">
        <div style="font-weight:600;font-size:0.85rem;color:#D32F2F;">预警升级规则</div>
        <div style="font-size:0.78rem;color:#5F6B7A;line-height:1.8;margin-top:0.3rem;">
            • IV级(蓝) → III级(黄): 落石直径 > 10cm 或 同一目标连续检出超过10帧<br>
            • III级(黄) → II级(橙): 落石直径 > 20cm 或 检测到坠落状态 (垂直加速度 > 阈值)<br>
            • II级(橙) → I级(红): 落石直径 > 30cm 或 置信度突破0.90 或 多目标同时坠落 (>3个)
        </div>
    </div>\
    """), unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # Section 4: Push Channel Configuration
    # ══════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:4px;height:24px;background:#6A1B9A;border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1.1rem;color:#1B2838;">推送渠道配置</div>
        <div style="font-size:0.78rem;color:#5F6B7A;">微信 &middot; 邮件 &middot; 企业微信 &middot; SSE弹窗</div>
    </div>
    """, unsafe_allow_html=True)

    channel_cols = st.columns(4)
    channels = [
        ("微信推送", "PushPlus", "已配置", "通过 PushPlus API 推送预警消息到指定微信群/个人", "#2E7D32"),
        ("邮件通知", "SMTP", "可选", "发送预警邮件 (含截图附件) 到值班人员邮箱列表", "#E65100"),
        ("企业微信", "Webhook", "可选", "通过企业微信机器人 Webhook 推送预警卡片消息", "#1565C0"),
        ("SSE弹窗", "Server-Sent Events", "内置", "Web看板实时弹窗 + 分级声音报警", "#6A1B9A"),
    ]
    for col, (name, tech, status, desc, color) in zip(channel_cols, channels):
        with col:
            st.markdown(f"""
            <div style="padding:0.75rem;background:#fff;border:1px solid #E3E8EF;border-radius:10px;
                        border-top:3px solid {color};text-align:center;">
                <div style="font-weight:600;font-size:0.85rem;color:#1B2838;">{name}</div>
                <div style="font-size:0.7rem;color:#5F6B7A;">{tech}</div>
                <div style="margin-top:0.3rem;">
                    <span style="font-size:0.65rem;font-weight:600;color:{color};
                                 background:{color}15;padding:0.1rem 0.5rem;border-radius:3px;">{status}</span>
                </div>
                <div style="font-size:0.68rem;color:#5F6B7A;margin-top:0.4rem;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # Section 5: Alert Content Template
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:4px;height:24px;background:#D32F2F;border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1.1rem;color:#1B2838;">预警内容模板</div>
    </div>
    """, unsafe_allow_html=True)

    # 获取当前监测点位信息
    try:
        active_site = get_active_site() if get_active_site is not None else None
    except Exception:
        active_site = None
    if active_site is not None and hasattr(active_site, 'name'):
        site_loc = f"{active_site.name} ({getattr(active_site, 'highway', '未知路段')})"
    else:
        site_loc = "示例监测点 (G210国道)"

    template_cols = st.columns(2)
    with template_cols[0]:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #D32F2F;">
            <div style="font-weight:700;color:#D32F2F;margin-bottom:0.5rem;">I 级预警推送模板</div>
            <div style="font-size:0.75rem;color:#5F6B7A;line-height:1.8;font-family:monospace;">
                ═══════════════════<br>
                <b>【RockGuard 落石预警 · I 级】</b><br>
                ═══════════════════<br>
                <b>时间</b>: 2026-06-12 14:35:22<br>
                <b>地点</b>: {site_loc}<br>
                <b>等级</b>: I 级 · 特别严重<br>
                <b>落石数量</b>: 3 块<br>
                <b>最大直径</b>: 约 45 cm<br>
                <b>置信度</b>: 0.95<br>
                <b>运动状态</b>: 坠落<br>
                <b>现场截图</b>: [附件]<br>
                ───────────────────<br>
                <b>处置建议</b>:<br>
                1. 立即封闭相关车道<br>
                2. 通知交警部门协助管制<br>
                3. 派员现场确认灾情<br>
                4. 启动应急预案<br>
                ───────────────────<br>
                系统自动发送 · RockGuard v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

    with template_cols[1]:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #F9A825;">
            <div style="font-weight:700;color:#F57F17;margin-bottom:0.5rem;">III 级预警推送模板</div>
            <div style="font-size:0.75rem;color:#5F6B7A;line-height:1.8;font-family:monospace;">
                ═══════════════════<br>
                <b>【RockGuard 落石预警 · III 级】</b><br>
                ═══════════════════<br>
                <b>时间</b>: 2026-06-12 09:15:08<br>
                <b>地点</b>: {site_loc}<br>
                🟡 <b>等级</b>: III 级 · 较重<br>
                <b>落石数量</b>: 1 块<br>
                <b>最大直径</b>: 约 15 cm<br>
                <b>置信度</b>: 0.65<br>
                <b>现场截图</b>: [附件]<br>
                ───────────────────<br>
                <b>关注要点</b>:<br>
                1. 纳入当日监测日报<br>
                2. 关注后续帧趋势<br>
                3. 若升级及时通知<br>
                ───────────────────<br>
                系统自动发送 · RockGuard v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)
