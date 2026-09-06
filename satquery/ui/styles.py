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
    --sat-bg-space: #080d1a;
    --sat-bg-card: rgba(11, 17, 32, 0.76);
    --sat-bg-card-hover: rgba(16, 25, 46, 0.88);
    --sat-bg-card-active: rgba(22, 34, 60, 0.94);
    --sat-border-subtle: rgba(255, 255, 255, 0.08);
    --sat-border-glow: rgba(0, 229, 255, 0.35);
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
   STREAMLIT CORE SHELL OVERRIDES (COMPACT VIEWPORT OPTIMIZED)
   ========================================================================== */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--sat-bg-deep) !important;
    background-image: 
        radial-gradient(circle at 18% 12%, rgba(0, 229, 255, 0.04) 0%, transparent 40%),
        radial-gradient(circle at 82% 16%, rgba(99, 102, 241, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(14, 165, 233, 0.03) 0%, transparent 50%),
        linear-gradient(180deg, #050811 0%, #080d1a 50%, #050811 100%) !important;
    background-attachment: fixed !important;
    color: var(--sat-text-main) !important;
    font-family: var(--sat-font-sans) !important;
}

/* Optimize Streamlit vertical padding for SIH presentation viewport */
.main .block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1380px !important;
}

/* Hide default streamlit decoration header */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1rem !important;
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
    background: rgba(255, 255, 255, 0.16);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--sat-sky);
}

/* ==========================================================================
   CELESTIAL PARTICLES & SUBTLE ORBITAL SPACE ENVIRONMENT
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

/* Subtle Slow Moving Starfield Layer */
.space-stars {
    position: absolute;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(1px 1px at 25px 35px, rgba(255, 255, 255, 0.4), transparent),
        radial-gradient(1.5px 1.5px at 140px 110px, rgba(255, 255, 255, 0.55), transparent),
        radial-gradient(1px 1px at 280px 70px, rgba(56, 189, 248, 0.4), transparent),
        radial-gradient(1.5px 1.5px at 420px 190px, rgba(255, 255, 255, 0.35), transparent),
        radial-gradient(1px 1px at 600px 120px, rgba(255, 255, 255, 0.45), transparent),
        radial-gradient(2px 2px at 750px 240px, rgba(0, 229, 255, 0.4), transparent),
        radial-gradient(1px 1px at 900px 80px, rgba(255, 255, 255, 0.35), transparent),
        radial-gradient(1.5px 1.5px at 1100px 170px, rgba(255, 255, 255, 0.5), transparent),
        radial-gradient(1px 1px at 1250px 320px, rgba(129, 140, 248, 0.35), transparent),
        radial-gradient(1px 1px at 1350px 90px, rgba(255, 255, 255, 0.4), transparent);
    background-size: 1400px 700px;
    animation: stars-drift 140s linear infinite;
    opacity: 0.65;
}

@keyframes stars-drift {
    from { transform: translateY(0); }
    to { transform: translateY(-700px); }
}

/* Subtle Concentric Orbital Trajectories */
.orbit-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px dashed rgba(56, 189, 248, 0.07);
    pointer-events: none;
}
.orbit-ring-1 {
    width: 900px;
    height: 900px;
    top: -250px;
    right: -250px;
    animation: orbit-spin 220s linear infinite;
}
.orbit-ring-2 {
    width: 1400px;
    height: 1400px;
    top: -450px;
    right: -450px;
    border-color: rgba(99, 102, 241, 0.04);
    animation: orbit-spin 360s linear infinite reverse;
}

/* Faint Satellite Dot traveling along Orbit */
.orbit-ring-1::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 0;
    width: 5px;
    height: 5px;
    background: #00e5ff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e5ff, 0 0 20px rgba(0, 229, 255, 0.6);
}

@keyframes orbit-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
    .space-stars, .orbit-ring-1, .orbit-ring-2 {
        animation: none !important;
    }
}

/* ==========================================================================
   NAVIGATION BAR (COMPACT & SLEEK)
   ========================================================================== */
.sat-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.15rem;
    background: rgba(10, 15, 28, 0.88);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 12px;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    position: relative;
    z-index: 10;
}

.sat-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.sat-brand-icon {
    font-size: 1.5rem;
    filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.5));
}

.sat-brand-title {
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: 0.09em;
    background: linear-gradient(135deg, #ffffff 30%, var(--sat-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.sat-brand-subtitle {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sat-sky);
    font-weight: 600;
}

.sat-nav-telemetry {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.sat-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
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
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
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
   HERO BANNER (COMPACT & REFINED FOR 1080P FIRST VIEWPORT)
   ========================================================================== */
.sat-hero {
    background: linear-gradient(135deg, rgba(13, 22, 42, 0.8) 0%, rgba(8, 14, 28, 0.65) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.9rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 28px 0 rgba(0, 0, 0, 0.4);
}

.sat-hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0, 229, 255, 0.6) 50%, transparent 100%);
}

.sat-hero-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.sat-hero-tag {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--sat-sky);
}

.sat-hero-headline {
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-top: 0.2rem;
    margin-bottom: 0.35rem;
    background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sat-hero-sub {
    font-size: 0.88rem;
    color: var(--sat-text-muted);
    max-width: 820px;
    line-height: 1.45;
    margin-bottom: 0.75rem;
}

.sat-mission-steps {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sat-step-node {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    color: var(--sat-text-main);
}

.sat-step-node span {
    color: var(--sat-cyan);
    font-weight: 800;
    font-family: var(--sat-font-mono);
}

.sat-step-arrow {
    color: var(--sat-sky);
    font-size: 0.75rem;
    opacity: 0.7;
}

/* ==========================================================================
   GLASSMORPHIC CARDS & PANELS
   ========================================================================== */
.sat-card {
    background: var(--sat-bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 0.9rem;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 18px 0 rgba(0, 0, 0, 0.3);
}

.sat-card:hover {
    border-color: var(--sat-border-glow);
    transform: translateY(-1px);
    box-shadow: 0 6px 24px -2px rgba(0, 229, 255, 0.09);
}

.sat-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.sat-card-title {
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--sat-text-bright);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sat-card-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: var(--sat-sky);
    font-family: var(--sat-font-mono);
}

.sat-card-badge.ready {
    background: rgba(16, 185, 129, 0.14);
    border-color: rgba(16, 185, 129, 0.4);
    color: #34d399;
}

.sat-card-badge.waiting {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.35);
    color: #fbbf24;
}

/* ==========================================================================
   AEROSPACE METRIC HUDS
   ========================================================================== */
.sat-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.85rem;
    margin: 0.85rem 0;
}

.sat-metric-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
}

.sat-metric-box:hover {
    border-color: var(--sat-border-glow);
    transform: translateY(-2px);
    background: rgba(18, 28, 52, 0.8);
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
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--sat-text-muted);
    margin-bottom: 0.25rem;
}

.sat-metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    font-family: var(--sat-font-mono);
    color: var(--sat-text-bright);
    line-height: 1.1;
}

.sat-metric-hint {
    font-size: 0.68rem;
    color: var(--sat-text-faint);
    margin-top: 0.2rem;
}

/* ==========================================================================
   HORIZONTAL CONFIDENCE BARS (TOP 5 PREDICTIONS)
   ========================================================================== */
.pred-list {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    margin: 0.85rem 0;
}

.pred-item {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    transition: all 0.2s ease;
}

.pred-item:hover {
    border-color: rgba(0, 229, 255, 0.35);
    background: rgba(18, 28, 52, 0.75);
    transform: translateX(3px);
}

.pred-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
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
    font-size: 0.72rem;
    background: rgba(56, 189, 248, 0.12);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    border: 1px solid rgba(56, 189, 248, 0.2);
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
    animation: bar-grow 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes bar-grow {
    from { opacity: 0.4; }
    to { opacity: 1; }
}

/* ==========================================================================
   AI ROUTING PIPELINE FLOW VISUALIZATION WITH ACTIVE PULSE
   ========================================================================== */
.pipeline-flow {
    display: flex;
    align-items: stretch;
    gap: 0.5rem;
    padding: 0.85rem 1rem;
    background: rgba(11, 16, 30, 0.85);
    border: 1px solid var(--sat-border-subtle);
    border-radius: 12px;
    margin: 0.85rem 0 1.25rem 0;
    overflow-x: auto;
}

.flow-node {
    flex: 1;
    min-width: 160px;
    background: rgba(20, 28, 48, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: all 0.25s ease;
}

.flow-node.completed {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.07);
}

.flow-node.active {
    border-color: rgba(0, 229, 255, 0.55);
    background: rgba(14, 165, 233, 0.12);
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.15);
}

.flow-step-num {
    font-size: 0.64rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sat-sky);
    font-weight: 700;
}

.flow-step-name {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--sat-text-bright);
    margin: 0.15rem 0;
}

.flow-step-detail {
    font-size: 0.7rem;
    font-family: var(--sat-font-mono);
    color: var(--sat-text-muted);
}

.flow-connector {
    display: flex;
    align-items: center;
    color: var(--sat-cyan);
    font-size: 1.1rem;
    opacity: 0.75;
    animation: flow-pulse 2s ease-in-out infinite;
}

@keyframes flow-pulse {
    0%, 100% { opacity: 0.4; transform: scale(0.95); }
    50% { opacity: 1; transform: scale(1.1); filter: drop-shadow(0 0 6px rgba(0, 229, 255, 0.6)); }
}

/* ==========================================================================
   STREAMLIT FORM WIDGETS OVERRIDES & CTA BUTTON
   ========================================================================== */
/* Main Analyze CTA Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.06em !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 11px !important;
    padding: 0.75rem 1.8rem !important;
    box-shadow: 0 4px 22px -2px rgba(37, 99, 235, 0.5), 0 0 15px rgba(0, 229, 255, 0.25) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 229, 255, 0.55), 0 0 25px rgba(0, 229, 255, 0.4) !important;
    border-color: var(--sat-cyan) !important;
}

button[kind="primary"]:active {
    transform: translateY(1px) !important;
}

/* Secondary Buttons & Suggestion Chips */
button[kind="secondary"] {
    background: rgba(20, 28, 48, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: var(--sat-text-main) !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[kind="secondary"]:hover {
    background: rgba(30, 44, 76, 0.85) !important;
    border-color: var(--sat-cyan) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0, 229, 255, 0.2) !important;
}

/* Text Input & Textarea */
.stTextInput input, .stTextArea textarea {
    background: rgba(9, 14, 26, 0.88) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.96rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.5) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--sat-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.25), inset 0 2px 5px rgba(0, 0, 0, 0.5) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(13, 19, 33, 0.55) !important;
    border: 1px dashed rgba(56, 189, 248, 0.3) !important;
    border-radius: 10px !important;
    padding: 0.85rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--sat-cyan) !important;
    background: rgba(14, 165, 233, 0.08) !important;
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
    padding: 0.45rem 0.9rem !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(20, 30, 55, 0.9) !important;
    border-color: var(--sat-cyan) !important;
    color: var(--sat-cyan) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(12, 18, 32, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: var(--sat-text-bright) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
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
    background: rgba(13, 20, 36, 0.9) !important;
    border: 1px solid var(--sat-cyan) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.15) !important;
}
</style>

<!-- Background Ambient Space Environment (CSS Pure & Offline) -->
<div class="space-bg-decor">
    <div class="space-stars"></div>
    <div class="orbit-ring orbit-ring-1"></div>
    <div class="orbit-ring orbit-ring-2"></div>
</div>
"""
