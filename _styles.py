"""RockGuard — 全局 CSS 样式注入"""
import streamlit as st
from _config import *

def inject_css():
    """注入全局 CSS 样式。必须在 st.set_page_config() 之后调用。"""
    st.markdown(f"""
<style>
    /* === 全局 === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; color: {TEXT_PRIMARY}; }}

    /* === 品牌顶栏 === */
    .brand-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #0D47A1 100%);
        border-radius: 10px; color: #fff; margin-bottom: 1rem;
    }}
    .brand-header .logo {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.5px; }}
    .brand-header .meta {{ font-size: 0.75rem; opacity: 0.85; text-align: right; }}
    .brand-header .meta span {{ margin-left: 1rem; }}

    /* === 预警等级标签 === */
    .alert-badge {{
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
        font-size: 0.78rem; font-weight: 600;
    }}
    .alert-badge.red    {{ background: {ALERT_BG['red']};    color: {ALERT_COLORS['red']}; }}
    .alert-badge.orange {{ background: {ALERT_BG['orange']}; color: {ALERT_COLORS['orange']}; }}
    .alert-badge.yellow {{ background: {ALERT_BG['yellow']}; color: #F57F17; }}
    .alert-badge.blue   {{ background: {ALERT_BG['blue']};   color: {ALERT_COLORS['blue']}; }}
    .alert-badge.green  {{ background: {ALERT_BG['green']};  color: {ALERT_COLORS['green']}; }}

    /* === 卡片容器 === */
    .card {{
        background: #fff; border: 1px solid #E3E8EF; border-radius: 10px;
        padding: 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card.active {{ border-color: {PRIMARY_BLUE}; box-shadow: 0 0 0 1px {PRIMARY_BLUE}20; }}

    /* === KPI 指标卡 === */
    .kpi-card {{
        background: #fff; border: 1px solid #E3E8EF; border-radius: 10px;
        padding: 1rem 1.25rem; text-align: center;
    }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {PRIMARY_BLUE}; line-height: 1.2; }}
    .kpi-value.danger {{ color: {ALERT_COLORS['red']}; }}
    .kpi-value.warning {{ color: {ALERT_COLORS['orange']}; }}
    .kpi-label {{ font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 0.25rem; }}

    /* === 场景选择卡 === */
    .scene-card {{
        padding: 1rem; border-radius: 10px; border: 2px solid #E3E8EF;
        background: #fff; margin-bottom: 0.5rem; transition: all 0.15s;
    }}
    .scene-card:hover {{ border-color: {PRIMARY_BLUE}60; }}
    .scene-card.selected {{ border-color: {PRIMARY_BLUE}; background: {PRIMARY_BLUE_LIGHT}; }}

    /* === 状态指示器 === */
    .status-dot {{
        display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
    }}
    .status-dot.live {{ background: #4CAF50; animation: pulse 2s infinite; }}
    .status-dot.idle {{ background: #9E9E9E; }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

    /* === 数据表格 === */
    .dataframe-container {{ border-radius: 8px; overflow: hidden; }}

    /* === 分割线 === */
    hr.divider {{ border: none; border-top: 1px solid #E3E8EF; margin: 1.5rem 0; }}

    /* === 预警分级决策树 === */
    .tree-container {{
        background: linear-gradient(135deg, #F5F7FA 0%, #fff 100%);
        border: 1px solid #E3E8EF; border-radius: 12px;
        padding: 1.5rem 1rem; overflow-x: auto;
    }}
    .tree-root {{
        display: flex; flex-direction: column; align-items: center; gap: 0;
    }}
    .tree-node {{
        padding: 0.5rem 1rem; border-radius: 8px; text-align: center;
        font-weight: 600; font-size: 0.82rem; margin: 0.25rem 0;
    }}
    .tree-node.root {{ background: #1565C0; color: #fff; font-size: 0.9rem; padding: 0.6rem 1.5rem; }}
    .tree-node.decision {{ background: #fff; border: 2px solid #1565C0; color: #1B2838; min-width: 200px; }}
    .tree-node.leaf-red {{ background: #FFEBEE; border: 2px solid #D32F2F; color: #D32F2F; }}
    .tree-node.leaf-orange {{ background: #FFF3E0; border: 2px solid #E65100; color: #E65100; }}
    .tree-node.leaf-yellow {{ background: #FFFDE7; border: 2px solid #F9A825; color: #F57F17; }}
    .tree-node.leaf-blue {{ background: #E3F2FD; border: 2px solid #1565C0; color: #1565C0; }}
    .tree-node.leaf-green {{ background: #E8F5E9; border: 2px solid #2E7D32; color: #2E7D32; }}
    .tree-branch {{ display: flex; gap: 1rem; justify-content: center; margin: 0.3rem 0; flex-wrap: wrap; }}
    .tree-arrow {{ text-align: center; color: #5F6B7A; font-size: 0.8rem; font-weight: 600; }}
    .tree-label {{ font-size: 0.65rem; color: #5F6B7A; text-align: center; margin: 0.1rem 0; }}
</style>
    """, unsafe_allow_html=True)
