"""Aerospace and Mission Control HTML/CSS Components for SatQuery AI.

Provides reusable UI cards, telemetry HUDs, pipeline flow diagrams,
and high-fidelity visual representations of satellite inference results.
"""

from pathlib import Path
from typing import Any, Optional
import html


def render_header(cuda_available: bool = False, active_model_count: int = 2) -> str:
    """Render compact mission-control navbar with live status telemetry."""
    hw_text = "NVIDIA CUDA" if cuda_available else "CPU ENGINE"
    hw_icon = "⚡" if cuda_available else "💻"
    
    return f"""
    <div class="sat-navbar">
        <div class="sat-brand">
            <div class="sat-brand-icon">🛰️</div>
            <div>
                <div class="sat-brand-title">SATQUERY AI</div>
                <div class="sat-brand-subtitle">Satellite Intelligence Platform</div>
            </div>
        </div>
        <div class="sat-nav-telemetry">
            <div class="sat-status-badge">
                <span class="sat-status-ping"></span>
                <span>System Online</span>
            </div>
            <div class="sat-hw-pill">
                <span>{hw_icon} {hw_text}</span>
            </div>
            <div class="sat-hw-pill">
                <span>🛰️ {active_model_count} Models Ready</span>
            </div>
        </div>
    </div>
    """


def render_hero() -> str:
    """Render high-impact mission hero banner with 4-step workflow breadcrumbs."""
    return """
    <div class="sat-hero">
        <div class="sat-hero-header">
            <div class="sat-hero-tag">🛰️ Autonomous Earth Observation & Remote Sensing Intelligence</div>
            <div class="sat-card-badge">MISSION CONTROL v4.2</div>
        </div>
        <div class="sat-hero-headline">ASK. ANALYZE. UNDERSTAND EARTH.</div>
        <div class="sat-hero-sub">
            Natural-language intelligence for satellite imagery — bi-temporal change detection, 
            multispectral land-cover classification, and sub-pixel spatial analytics.
        </div>
        <div class="sat-mission-steps">
            <div class="sat-step-node"><span>01</span> DATA SOURCE</div>
            <div class="sat-step-arrow">➔</div>
            <div class="sat-step-node"><span>02</span> ASK SATQUERY AI</div>
            <div class="sat-step-arrow">➔</div>
            <div class="sat-step-node"><span>03</span> AI ROUTING</div>
            <div class="sat-step-arrow">➔</div>
            <div class="sat-step-node"><span>04</span> MISSION PRODUCTS</div>
        </div>
    </div>
    """


def render_pipeline_flow(
    query: str,
    task: str,
    model_name: str,
    model_id: str,
    tools: list[str],
    outputs_summary: str = "GeoTIFF + GeoJSON + Report",
) -> str:
    """Render interactive AI routing flow diagram showing runtime plan execution."""
    safe_q = html.escape(query)
    if len(safe_q) > 40:
        safe_q = safe_q[:38] + "..."

    tools_summary = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_summary += f" +{len(tools)-3}"

    return f"""
    <div class="pipeline-flow">
        <div class="flow-node completed">
            <div class="flow-step-num">✓ STAGE 01 • INTENT</div>
            <div class="flow-step-name">Query Interpreted</div>
            <div class="flow-step-detail">"{safe_q}"</div>
        </div>
        <div class="flow-connector">➔</div>
        <div class="flow-node completed">
            <div class="flow-step-num">✓ STAGE 02 • ROUTER</div>
            <div class="flow-step-name">Query Router</div>
            <div class="flow-step-detail">{task}</div>
        </div>
        <div class="flow-connector">➔</div>
        <div class="flow-node active">
            <div class="flow-step-num">◉ STAGE 03 • SPECIALIST</div>
            <div class="flow-step-name">{model_name}</div>
            <div class="flow-step-detail">model: {model_id}</div>
        </div>
        <div class="flow-connector">➔</div>
        <div class="flow-node completed">
            <div class="flow-step-num">✓ STAGE 04 • SPATIAL ENGINE</div>
            <div class="flow-step-name">Spatial Intelligence</div>
            <div class="flow-step-detail">{tools_summary}</div>
        </div>
        <div class="flow-connector">➔</div>
        <div class="flow-node completed">
            <div class="flow-step-num">✓ STAGE 05 • PRODUCTS</div>
            <div class="flow-step-name">Mission Products</div>
            <div class="flow-step-detail">{outputs_summary}</div>
        </div>
    </div>
    """


def render_top5_bars(top_5: list[dict[str, Any]]) -> str:
    """Render horizontal animated confidence progress bars for Top 5 predictions."""
    items_html = []
    for idx, item in enumerate(top_5, 1):
        cls_name = html.escape(item.get("class", "Unknown"))
        prob = float(item.get("probability", 0.0))
        pct_str = f"{prob * 100:.2f}%"
        bar_width = f"{max(2.0, min(100.0, prob * 100)):.1f}%"

        items_html.append(f"""
        <div class="pred-item">
            <div class="pred-header">
                <div class="pred-class">
                    <span class="pred-rank">#{idx:02d}</span>
                    <span>{cls_name}</span>
                </div>
                <div class="pred-score">{pct_str}</div>
            </div>
            <div class="pred-track">
                <div class="pred-fill" style="width: {bar_width};"></div>
            </div>
        </div>
        """)

    return f'<div class="pred-list">{"".join(items_html)}</div>'


def render_telemetry_hud(metrics: list[dict[str, str]]) -> str:
    """Render aerospace telemetry metric cards grid."""
    cards_html = []
    for m in metrics:
        lbl = html.escape(m.get("label", ""))
        val = html.escape(str(m.get("value", "")))
        hint = html.escape(m.get("hint", ""))
        hint_html = f'<div class="sat-metric-hint">{hint}</div>' if hint else ""

        cards_html.append(f"""
        <div class="sat-metric-box">
            <div class="sat-metric-label">{lbl}</div>
            <div class="sat-metric-value">{val}</div>
            {hint_html}
        </div>
        """)

    return f'<div class="sat-metrics-grid">{"".join(cards_html)}</div>'


def render_file_badge(filename: str, meta: Any, modality_val: str = "Optical", slot_label: str = "IMAGE READY") -> str:
    """Render clean file metadata pill with status indicator."""
    crs_str = meta.crs if meta and meta.crs else "Unprojected"
    dim_str = f"{meta.width} × {meta.height} px" if meta and meta.width else "Raster"
    bands_str = f"{meta.band_count} Bands" if meta and meta.band_count else ""

    return f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.55rem 0.85rem; background: rgba(18, 26, 46, 0.75); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; margin-bottom: 0.55rem;">
        <div style="display: flex; align-items: center; gap: 0.55rem;">
            <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
            <span style="font-weight: 700; color: #34d399; font-size: 0.75rem; font-family: var(--sat-font-mono); letter-spacing: 0.05em;">● {html.escape(slot_label)}</span>
            <span style="font-weight: 600; color: #f8fafc; font-size: 0.82rem; margin-left: 0.35rem;">{html.escape(filename)}</span>
        </div>
        <div style="font-family: var(--sat-font-mono); font-size: 0.7rem; color: #94a3b8; display: flex; gap: 0.55rem;">
            <span>{dim_str}</span>
            <span>•</span>
            <span>{bands_str}</span>
            <span>•</span>
            <span style="color: #38bdf8;">{html.escape(crs_str)}</span>
        </div>
    </div>
    """
