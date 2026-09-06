"""Master Design System and Aerospace Styling for SatQuery AI.

Implements a NASA Mission Control + Modern AI Earth Observation aesthetic
with dark space glassmorphism, subtle celestial atmospheric glows, crisp typography,
and high-fidelity telemetry components.
"""

SPACE_THEME_CSS = """
<style>
/* ==========================================================================
   DESIGN TOKENS & SYSTEM VARIABLES
   ========================================================================== */
:root {
    --sat-bg-deep: #050811;
    --sat-bg-space: #090e1c;
    --sat-bg-card: rgba(13, 19, 33, 0.72);
    --sat-bg-card-hover: rgba(18, 27, 46, 0.85);
    --sat-bg-card-active: rgba(24, 36, 62, 0.92);
    --sat-border-subtle: rgba(255, 255, 255, 0.07);
    --sat-border-glow: rgba(56, 189, 248, 0.32);
    --sat-border-emerald: rgba(16, 185, 129, 0.35);
    --sat-cyan: #00e5ff;
    --sat-sky: #38bdf8;
    --sat-blue: #2563eb;
    --sat-indigo: #6366f1;
    --sat-violet: #818cf8;
    --sat-emerald: #10b981;
    --sat-amber: #f59e0b;
    --sat-rose: #f43f5e;
    --sat-text-bright: #f8fafc;
    --sat-text-main: #e2e8f0;
    --sat-text-muted: #94a3b8;
    --sat-text-faint: #64748b;
    --sat-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
    --sat-font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, "Courier New", monospace;
}

/* ==========================================================================
   STREAMLIT CORE SHELL OVERRIDES
   ========================================================================== */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--sat-bg-deep) !important;
    background-image: 
        radial-gradient(circle at 12% 15%, rgba(0, 229, 255, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 88% 18%, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 50% 85%, rgba(14, 165, 233, 0.03) 0%, transparent 55%),
        linear-gradient(180deg, #050811 0%, #080d1a 50%, #050811 100%) !important;
    background-attachment: fixed !important;
    color: var(--sat-text-main) !important;
    font-family: var(--sat-font-sans) !important;
}

/* Reduce Streamlit's huge top blank padding */
.main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1400px !important;
}

/* Hide default streamlit decoration header */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.5rem !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--sat-bg-deep);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--sat-sky);
}

/* ==========================================================================
   CELESTIAL PARTICLES & ORBITAL BACKGROUND (CSS/SVG)
   ========================================================================== */
.space-bg-decor {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.orbit-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px dashed rgba(56, 189, 248, 0.08);
    pointer-events: none;
}
.orbit-ring-1 {
    width: 800px;
    height: 800px;
    top: -200px;
    right: -250px;
    animation: orbit-spin 180s linear infinite;
}
.orbit-ring-2 {
    width: 1200px;
    height: 1200px;
    top: -400px;
    right: -450px;
    border-color: rgba(99, 102, 241, 0.05);
    animation: orbit-spin 300s linear infinite reverse;
}

@keyframes orbit-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
    .orbit-ring-1, .orbit-ring-2 {
        animation: none !important;
    }
}

/* ==========================================================================
   NAVIGATION BAR
   ========================================================================== */
.sat-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    background: rgba(10, 15, 28, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 12px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    position: relative;
    z-index: 10;
}

.sat-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.sat-brand-icon {
    font-size: 1.6rem;
    filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.5));
}

.sat-brand-title {
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #ffffff 30%, var(--sat-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.sat-brand-subtitle {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sat-sky);
    font-weight: 600;
}

.sat-nav-telemetry {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.sat-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
}

.sat-status-ping {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: ping-pulse 2s cubic-bezier(0, 0, 0.2, 1) infinite;
}

.sat-hw-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-family: var(--sat-font-mono);
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--sat-text-main);
}

@keyframes ping-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.15); }
}

/* ==========================================================================
   HERO BANNER
   ========================================================================== */
.sat-hero {
    background: linear-gradient(135deg, rgba(13, 22, 40, 0.75) 0%, rgba(10, 16, 30, 0.6) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.sat-hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.6) 50%, transparent 100%);
}

.sat-hero-tag {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--sat-sky);
    margin-bottom: 0.5rem;
}

.sat-hero-headline {
    font-size: 2.1rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sat-hero-sub {
    font-size: 0.98rem;
    color: var(--sat-text-muted);
    max-width: 800px;
    line-height: 1.5;
    margin-bottom: 1.25rem;
}

.sat-mission-steps {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sat-step-node {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    color: var(--sat-text-main);
}

.sat-step-arrow {
    color: var(--sat-sky);
    font-size: 0.85rem;
}

/* ==========================================================================
   GLASSMORPHIC CARDS & PANELS
   ========================================================================== */
.sat-card {
    background: var(--sat-bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
}

.sat-card:hover {
    border-color: var(--sat-border-glow);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px -4px rgba(0, 229, 255, 0.08);
}

.sat-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.sat-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--sat-text-bright);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sat-card-badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: var(--sat-sky);
    font-family: var(--sat-font-mono);
}

/* ==========================================================================
   AEROSPACE METRIC HUDS
   ========================================================================== */
.sat-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.sat-metric-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}

.sat-metric-box:hover {
    border-color: var(--sat-border-glow);
}

.sat-metric-box::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: var(--sat-sky);
}

.sat-metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--sat-text-muted);
    margin-bottom: 0.35rem;
}

.sat-metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    font-family: var(--sat-font-mono);
    color: var(--sat-text-bright);
    line-height: 1.1;
}

.sat-metric-hint {
    font-size: 0.7rem;
    color: var(--sat-text-faint);
    margin-top: 0.25rem;
}

/* ==========================================================================
   HORIZONTAL CONFIDENCE BARS (TOP 5 PREDICTIONS)
   ========================================================================== */
.pred-list {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    margin: 1rem 0;
}

.pred-item {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    transition: all 0.2s ease;
}

.pred-item:hover {
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(20, 30, 55, 0.7);
}

.pred-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.pred-class {
    color: var(--sat-text-bright);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.pred-rank {
    font-family: var(--sat-font-mono);
    color: var(--sat-sky);
    font-size: 0.75rem;
}

.pred-score {
    font-family: var(--sat-font-mono);
    font-weight: 700;
    color: var(--sat-cyan);
}

.pred-track {
    width: 100%;
    height: 7px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 9999px;
    overflow: hidden;
}

.pred-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #0284c7 0%, var(--sat-cyan) 100%);
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ==========================================================================
   AI ROUTING PIPELINE FLOW VISUALIZATION
   ========================================================================== */
.pipeline-flow {
    display: flex;
    align-items: stretch;
    gap: 0.5rem;
    padding: 1rem;
    background: rgba(11, 16, 30, 0.8);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 12px;
    margin: 1rem 0;
    overflow-x: auto;
}

.flow-node {
    flex: 1;
    min-width: 170px;
    background: rgba(20, 28, 48, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.flow-node.active {
    border-color: rgba(0, 229, 255, 0.4);
    background: rgba(14, 165, 233, 0.1);
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
}

.flow-step-num {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sat-sky);
    font-weight: 700;
}

.flow-step-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--sat-text-bright);
    margin: 0.2rem 0;
}

.flow-step-detail {
    font-size: 0.72rem;
    font-family: var(--sat-font-mono);
    color: var(--sat-text-muted);
}

.flow-connector {
    display: flex;
    align-items: center;
    color: var(--sat-sky);
    font-size: 1.1rem;
    opacity: 0.6;
}

/* ==========================================================================
   STREAMLIT FORM WIDGETS OVERRIDES
   ========================================================================== */
/* Primary Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px -2px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(0, 229, 255, 0.45) !important;
    border-color: var(--sat-cyan) !important;
}

button[kind="primary"]:active {
    transform: translateY(1px) !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background: rgba(20, 28, 48, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: var(--sat-text-main) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    background: rgba(30, 42, 70, 0.8) !important;
    border-color: var(--sat-border-glow) !important;
    color: var(--sat-cyan) !important;
}

/* Text Input & Textarea */
.stTextInput input, .stTextArea textarea {
    background: rgba(10, 15, 28, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--sat-sky) !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(13, 19, 33, 0.5) !important;
    border: 1px dashed rgba(56, 189, 248, 0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--sat-sky) !important;
    background: rgba(14, 165, 233, 0.05) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background-color: transparent !important;
    border-bottom: 1px solid var(--sat-border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--sat-text-muted) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(20, 30, 55, 0.9) !important;
    border-color: var(--sat-border-glow) !important;
    color: var(--sat-cyan) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(13, 19, 33, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: var(--sat-text-bright) !important;
    font-weight: 600 !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--sat-border-glow) !important;
    color: var(--sat-sky) !important;
}

/* Radio buttons styled as segmented control pills */
div[role="radiogroup"] {
    gap: 0.75rem !important;
}

/* Metrics in sidebar and elsewhere */
[data-testid="stMetricValue"] {
    font-family: var(--sat-font-mono) !important;
    color: var(--sat-text-bright) !important;
}

/* Status widget */
[data-testid="stStatusWidget"] {
    background: rgba(13, 20, 36, 0.85) !important;
    border: 1px solid var(--sat-border-glow) !important;
    border-radius: 12px !important;
}
</style>

<!-- Background Ambient Elements -->
<div class="space-bg-decor">
    <div class="orbit-ring orbit-ring-1"></div>
    <div class="orbit-ring orbit-ring-2"></div>
</div>
"""
