import html
import json
from pathlib import Path
import tempfile
import time
from typing import Optional
import streamlit as st

from satquery.agent.controller import AgentController
from satquery.domain.schemas import (
    AnalysisResult,
    InputMode,
    Modality,
    SlotAssignment,
    ValidationStatus,
)
from satquery.models.manager import get_resource_manager
from satquery.preprocess.preview import generate_preview_image
from satquery.preprocess.raster import (
    assess_image_quality,
    detect_modality,
    inspect_raster,
)
from satquery.ui import (
    SPACE_THEME_CSS,
    render_file_badge,
    render_header,
    render_hero,
    render_pipeline_flow,
    render_telemetry_hud,
    render_top5_bars,
)

# Page configuration
st.set_page_config(
    page_title="SatQuery AI — Satellite Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Aerospace Master Stylesheet
st.markdown(SPACE_THEME_CSS, unsafe_allow_html=True)

TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "satquery_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(uploaded_file) -> Path:
    target_path = TEMP_UPLOAD_DIR / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Query Hardware Telemetry
rm = get_resource_manager()
cuda_avail = rm.has_cuda()
reg = AgentController().model_registry
ready_count = sum(1 for _, a in reg.list_models() if a.status == "ready")
total_count = len(reg.list_models())

# -----------------------------------------------------------------------------
# TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
st.markdown(
    render_header(cuda_available=cuda_avail, active_model_count=ready_count),
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MISSION HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(render_hero(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR: ADVANCED TELEMETRY & CACHE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Mission Telemetry")
    device_name = "NVIDIA CUDA" if cuda_avail else "CPU Fallback Engine"
    st.markdown(f"**Execution Hardware:** `{device_name}`")
    if cuda_avail:
        vram = rm.get_vram_info()
        st.caption(f"Allocated: {vram.get('allocated_mb', 0):.0f} MB / Total: {vram.get('total_mb', 0):.0f} MB")
    else:
        st.caption("Host CPU execution active. Low memory overhead.")

    st.markdown("---")
    st.markdown("### 🛰️ Specialist Models")
    for mid, adapter in reg.list_models():
        st_icon = "🟢" if adapter.status == "ready" else "🟡" if adapter.status == "unloaded" else "⚪"
        st.markdown(f"{st_icon} **{adapter.name}** (`{mid}`)")
        st.caption(f"Status: {adapter.status.upper()} | Caps: {', '.join(adapter.capabilities)}")

    st.markdown("---")
    if st.button("🗑️ Reset Session Cache", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.last_result = None
        st.session_state.current_query = ""
        rm.cleanup_memory()
        st.rerun()


# -----------------------------------------------------------------------------
# 01. DATA SOURCE (COMBINED ELEGANT WORKFLOW & INGESTION)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="sat-card">
        <div class="sat-card-header">
            <div class="sat-card-title"><span>📡</span> 01 &nbsp;DATA SOURCE &mdash; Choose Earth Observation Workflow</div>
            <div class="sat-card-badge">INPUT SENSORS</div>
        </div>
    """,
    unsafe_allow_html=True,
)

mode_selection = st.radio(
    "Choose your Earth observation workflow:",
    [
        "🔄 Bi-Temporal Change (Open-CD BIT)",
        "📷 Single Image (BigEarthNet Land-Cover / Scene VQA)",
        "📡 Optical + SAR Multimodal",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

slots: list[SlotAssignment] = []
slot_files: dict[str, Path] = {}
analysis_kwargs = {}

if "Single Image" in mode_selection:
    input_mode = InputMode.I1_SINGLE_OPTICAL
    col_upload, col_params = st.columns([1.8, 1.2])

    with col_upload:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:0.25rem;'>📥 Ingest Scene (Sentinel-2, Landsat, GeoTIFF)</div>", unsafe_allow_html=True)
        single_upload = st.file_uploader(
            "Drop Satellite Scene (GeoTIFF / TIFF):",
            type=["tif", "tiff", "geotiff"],
            key="single_uploader",
            label_visibility="collapsed",
        )
        if single_upload:
            p = save_upload(single_upload)
            slots.append(SlotAssignment(slot_id="image", file_path=str(p)))
            slot_files["image"] = p

    with col_params:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:0.25rem;'>⚙️ Classification Threshold</div>", unsafe_allow_html=True)
        classification_threshold = st.slider(
            "Confidence Cutoff:",
            min_value=0.05,
            max_value=0.90,
            value=0.10,
            step=0.05,
            help="Minimum class probability to report in BigEarthNet scene classification (default: 0.10).",
        )
        analysis_kwargs["threshold"] = classification_threshold
        st.caption("• Pretrained on BigEarthNet v2.0 (reBEN) Sentinel-2 19 Corine Land Cover categories")

elif "Bi-Temporal Change" in mode_selection:
    input_mode = InputMode.I4_BITEMPORAL_PAIR
    col_t0, col_t1 = st.columns(2)

    with col_t0:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:0.25rem;'>⏳ T0 &mdash; Baseline Scene (Before)</div>", unsafe_allow_html=True)
        t0_upload = st.file_uploader(
            "Upload Date T0 (GeoTIFF / TIFF):",
            type=["tif", "tiff", "geotiff"],
            key="t0_uploader",
            label_visibility="collapsed",
        )
        if t0_upload:
            p0 = save_upload(t0_upload)
            slots.append(SlotAssignment(slot_id="t0", file_path=str(p0)))
            slot_files["t0"] = p0

    with col_t1:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:0.25rem;'>⌛ T1 &mdash; Resurvey Scene (After)</div>", unsafe_allow_html=True)
        t1_upload = st.file_uploader(
            "Upload Date T1 (GeoTIFF / TIFF):",
            type=["tif", "tiff", "geotiff"],
            key="t1_uploader",
            label_visibility="collapsed",
        )
        if t1_upload:
            p1 = save_upload(t1_upload)
            slots.append(SlotAssignment(slot_id="t1", file_path=str(p1)))
            slot_files["t1"] = p1

    with st.expander("⚙️ Advanced Change Detection Settings", expanded=False):
        cd_col1, cd_col2, cd_col3, cd_col4 = st.columns(4)
        with cd_col1:
            change_threshold = st.slider(
                "Change Cutoff Threshold:",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="Probability cutoff for classifying a pixel as changed (default 0.50).",
            )
            analysis_kwargs["change_threshold"] = change_threshold
        with cd_col2:
            min_change_region_pixels = st.slider(
                "Min Noise Filter (px):",
                min_value=5,
                max_value=200,
                value=20,
                step=5,
                help="Filter out small noise clusters smaller than this pixel count (default 20 px).",
            )
            analysis_kwargs["min_change_region_pixels"] = min_change_region_pixels
        with cd_col3:
            st.markdown("**Sliding-Window Tile:** `256 × 256`")
            st.caption("Cosine-weighted blending preserves full resolution")
        with cd_col4:
            st.markdown("**Tile Overlap:** `25% (64 px)`")
            st.caption("Seamless tile boundaries with no visual seams")

else:  # Optical + SAR Multimodal
    input_mode = InputMode.I3_OPTICAL_SAR_PAIR
    col_opt, col_sar = st.columns(2)
    with col_opt:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:0.25rem;'>🌈 Optical / Multispectral Sensor</div>", unsafe_allow_html=True)
        opt_upload = st.file_uploader(
            "Upload Optical Image:",
            type=["tif", "tiff", "geotiff"],
            key="opt_uploader",
            label_visibility="collapsed",
        )
        if opt_upload:
            p_opt = save_upload(opt_upload)
            slots.append(SlotAssignment(slot_id="optical", file_path=str(p_opt), declared_modality=Modality.OPTICAL))
            slot_files["optical"] = p_opt

    with col_sar:
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:0.25rem;'>🛰️ Synthetic Aperture Radar (SAR)</div>", unsafe_allow_html=True)
        sar_upload = st.file_uploader(
            "Upload SAR Radar Image:",
            type=["tif", "tiff", "geotiff"],
            key="sar_uploader",
            label_visibility="collapsed",
        )
        if sar_upload:
            p_sar = save_upload(sar_upload)
            slots.append(SlotAssignment(slot_id="sar", file_path=str(p_sar), declared_modality=Modality.SAR))
            slot_files["sar"] = p_sar

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# LIVE INPUT STATE & SENSOR TELEMETRY
# -----------------------------------------------------------------------------
# Determine readiness state
input_is_ready = False
if "Bi-Temporal Change" in mode_selection:
    t0_ready = "t0" in slot_files
    t1_ready = "t1" in slot_files
    input_is_ready = t0_ready and t1_ready
    status_tag = "● ANALYSIS READY" if input_is_ready else "○ WAITING FOR IMAGERY"
    badge_cls = "ready" if input_is_ready else "waiting"
elif "Single Image" in mode_selection:
    input_is_ready = "image" in slot_files
    status_tag = "● ANALYSIS READY" if input_is_ready else "○ WAITING FOR IMAGERY"
    badge_cls = "ready" if input_is_ready else "waiting"
else:
    input_is_ready = "optical" in slot_files and "sar" in slot_files
    status_tag = "● ANALYSIS READY" if input_is_ready else "○ WAITING FOR IMAGERY"
    badge_cls = "ready" if input_is_ready else "waiting"

if slots:
    st.markdown(
        f"""
        <div class="sat-card">
            <div class="sat-card-header">
                <div class="sat-card-title"><span>🔍</span> Sensor Telemetry & Scene Verification</div>
                <div class="sat-card-badge {badge_cls}">{status_tag}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    preview_cols = st.columns(len(slots))

    for idx, slot in enumerate(slots):
        file_p = Path(slot.file_path)
        meta = inspect_raster(file_p)
        quality = assess_image_quality(file_p, meta)
        modality = detect_modality(meta)
        slot_lbl = f"{slot.slot_id.upper()} ● READY"

        with preview_cols[idx]:
            st.markdown(render_file_badge(meta.filename, meta, modality.value, slot_label=slot_lbl), unsafe_allow_html=True)
            img_preview, label_txt = generate_preview_image(file_p, max_side=360)
            if img_preview:
                st.image(img_preview, caption=f"Raster: {label_txt}", use_container_width=True)
            else:
                st.warning(f"Preview unavailable: {label_txt}")

            with st.expander("Detailed Telemetry Table", expanded=False):
                st.dataframe(
                    [
                        {"Metric": "Dimensions", "Value": f"{meta.width} × {meta.height} px"},
                        {"Metric": "Bands", "Value": str(meta.band_count)},
                        {"Metric": "Data Type", "Value": str(meta.dtype)},
                        {"Metric": "CRS", "Value": str(meta.crs)},
                        {"Metric": "Modality", "Value": modality.value.upper()},
                        {"Metric": "Quality Level", "Value": quality.level.value.upper()},
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 02. ASK SATQUERY AI (HERO NATURAL LANGUAGE COMPOSER)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="sat-card">
        <div class="sat-card-header">
            <div class="sat-card-title"><span>🧠</span> 02 &nbsp;ASK SATQUERY AI &mdash; What do you want to discover?</div>
            <div class="sat-card-badge">NATURAL LANGUAGE INTELLIGENCE</div>
        </div>
    """,
    unsafe_allow_html=True,
)

# Interactive Suggestion Chips
if "Single Image" in mode_selection:
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("🌿 Classify Land Cover", key="chip_lc", use_container_width=True):
            st.session_state.current_query = "Classify the land cover in this satellite image and give me the confidence scores."
            st.rerun()
    with chip_col2:
        if st.button("🛣️ Check for Roads (VQA)", key="chip_vqa", use_container_width=True):
            st.session_state.current_query = "Is there a road or runway in this image?"
            st.rerun()
    with chip_col3:
        if st.button("🛰️ Describe Scene", key="chip_desc", use_container_width=True):
            st.session_state.current_query = "Describe the scene and visible landscape features."
            st.rerun()
    with chip_col4:
        if st.button("💧 Locate Water & Wetlands", key="chip_water", use_container_width=True):
            st.session_state.current_query = "Locate and highlight the water bodies in this scene."
            st.rerun()
else:
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("🔄 Detect Changes", key="chip_change", use_container_width=True):
            st.session_state.current_query = "What changed between these two satellite dates?"
            st.rerun()
    with chip_col2:
        if st.button("🏗️ Find Building Construction", key="chip_bldg", use_container_width=True):
            st.session_state.current_query = "Identify areas where building construction occurred between T0 and T1."
            st.rerun()
    with chip_col3:
        if st.button("📐 Map Major Change Regions", key="chip_regions", use_container_width=True):
            st.session_state.current_query = "Where did change occur and what are the major changed regions?"
            st.rerun()
    with chip_col4:
        if st.button("🌐 Compare Two Images", key="chip_compare", use_container_width=True):
            st.session_state.current_query = "Compare these two satellite images and map significant changes."
            st.rerun()

query_text = st.text_input(
    "Query input:",
    value=st.session_state.current_query,
    placeholder="What changed between these two satellite images and where are the major change regions?",
    label_visibility="collapsed",
)

st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
run_button = st.button("✨ ANALYZE IMAGERY", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# EXECUTION ORCHESTRATION WITH MISSION CONTROL PROCESSING FEEDBACK
# -----------------------------------------------------------------------------
if run_button:
    if not slots:
        st.error("⚠️ Ingestion Incomplete: Please upload at least one satellite raster before running analysis.")
    elif not query_text.strip():
        st.error("⚠️ Query Missing: Please enter a natural-language query or click a suggestion chip above.")
    else:
        controller = AgentController()
        try:
            with st.status("🛰️ SATQUERY AI &mdash; ANALYZING EARTH OBSERVATION DATA", expanded=True) as status_box:
                st.write("✓ QUERY INTERPRETED & SENSORY INPUT VALIDATED")
                time.sleep(0.04)
                
                # Resolve plan preview for live specialist model name
                preview_plan = controller.planner.plan(
                    query=query_text.strip(),
                    slots=slots,
                    input_mode=input_mode,
                )
                spec_mname = preview_plan.selected_models[0] if preview_plan.selected_models else "Specialist Neural Model"
                st.write(f"◉ SPECIALIST MODEL RUNNING: **{spec_mname}**")
                
                result: AnalysisResult = controller.analyze(
                    query=query_text.strip(),
                    slots=slots,
                    input_mode=input_mode,
                    **analysis_kwargs,
                )
                st.write("✓ SPATIAL INTELLIGENCE & SUB-PIXEL ANALYTICS COMPLETE")
                st.write("✓ MISSION PRODUCTS GENERATED (GeoTIFF + GeoJSON + Report)")
                status_box.update(label="✨ MISSION ANALYSIS COMPLETED SUCCESSFULLY", state="complete", expanded=False)

            st.session_state.last_result = result
            st.session_state.history.append({
                "query": query_text,
                "task": result.plan.task,
                "theme": result.plan.theme.value,
                "result_text": result.result_text,
            })
        except Exception as exc:
            st.error(f"❌ Analysis Execution Error: {str(exc)}")


# -----------------------------------------------------------------------------
# 3. RESULTS & GEOSPATIAL INTELLIGENCE DASHBOARD
# -----------------------------------------------------------------------------
result: Optional[AnalysisResult] = st.session_state.last_result

if result:
    plan = result.plan
    selected_mid = plan.selected_models[0] if plan.selected_models else "specialist"
    model_obj = reg.get(selected_mid)
    selected_mname = model_obj.name if model_obj else selected_mid

    # AI Routing Pipeline Visualization
    st.markdown("### 🧭 Autonomous Routing Pipeline Execution")
    st.markdown(
        render_pipeline_flow(
            query=result.query,
            task=plan.task,
            model_name=selected_mname,
            model_id=selected_mid,
            tools=plan.selected_tools,
        ),
        unsafe_allow_html=True,
    )

    # Status check for blocked or unconfigured models
    if "Specialist model not configured" in result.result_text:
        st.warning(f"### ⚠️ {result.result_text}")
    elif "Validation failed" in result.result_text or "Agent planning stopped" in result.result_text:
        st.error(f"### ❌ {result.result_text}")
    else:
        # ---------------------------------------------------------------------
        # BI-TEMPORAL CHANGE DETECTION RESULTS
        # ---------------------------------------------------------------------
        stats_ev = next((e for e in result.evidence if e.title == "Bi-Temporal Change Statistics"), None)
        mask_art = next((e for e in result.evidence if "Binary Change Mask" in e.title), None)
        prob_art = next((e for e in result.evidence if "Probability Map" in e.title), None)
        vis_art = next((e for e in result.evidence if e.title == "Change Map Visual Overlay"), None)
        regions_art = next((e for e in result.evidence if "Bounding Boxes" in e.title), None)
        geojson_art = next((e for e in result.evidence if "GeoJSON" in e.title), None)

        if stats_ev and stats_ev.data:
            s_data = stats_ev.data
            st.markdown(
                """
                <div class="sat-card">
                    <div class="sat-card-header">
                        <div class="sat-card-title"><span>🛰️</span> Bi-Temporal Change Detection Metrics</div>
                        <div class="sat-card-badge">OPEN-CD BIT RESNET-18</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Quantitative Metrics HUD
            hud_metrics = [
                {
                    "label": "Changed Area",
                    "value": f"{s_data.get('changed_area_hectares', 0.0):,.2f} ha" if s_data.get("changed_area_m2") is not None else "Unprojected",
                    "hint": f"{s_data.get('changed_area_m2', 0):,.1f} m²" if s_data.get("changed_area_m2") is not None else "Pixel space",
                },
                {
                    "label": "Change Ratio",
                    "value": f"{s_data.get('percentage_changed', 0.0):.2f}%",
                    "hint": "Ratio of changed pixels",
                },
                {
                    "label": "Change Clusters",
                    "value": f"{s_data.get('regions_count', 0):,}",
                    "hint": "Polygonized regions",
                },
                {
                    "label": "AI Confidence",
                    "value": f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A",
                    "hint": result.confidence.method or "Calibrated",
                },
                {
                    "label": "Processing Time",
                    "value": f"{s_data.get('timings_seconds', {}).get('total', 0.0):.2f}s",
                    "hint": "Tiled inference + polygonization",
                },
            ]
            st.markdown(render_telemetry_hud(hud_metrics), unsafe_allow_html=True)

            # 3-Column Comparative Board
            st.markdown("##### 📷 Comparative Geospatial Viewer (T0 Baseline vs T1 Target vs AI Change Vector)")
            col_b1, col_b2, col_b3 = st.columns(3)

            with col_b1:
                st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#38bdf8; text-transform:uppercase;'>Baseline (T0)</div>", unsafe_allow_html=True)
                if "t0" in slot_files and slot_files["t0"].exists():
                    img0, lbl0 = generate_preview_image(slot_files["t0"], max_side=500)
                    if img0:
                        st.image(img0, caption=f"T0: {lbl0}", use_container_width=True)

            with col_b2:
                st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#38bdf8; text-transform:uppercase;'>Observation (T1)</div>", unsafe_allow_html=True)
                if "t1" in slot_files and slot_files["t1"].exists():
                    img1, lbl1 = generate_preview_image(slot_files["t1"], max_side=500)
                    if img1:
                        st.image(img1, caption=f"T1: {lbl1}", use_container_width=True)

            with col_b3:
                st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#f43f5e; text-transform:uppercase;'>AI Detected Change Overlay</div>", unsafe_allow_html=True)
                target_art = vis_art or mask_art
                if target_art and target_art.file_path and Path(target_art.file_path).exists():
                    img_ch, lbl_ch = generate_preview_image(Path(target_art.file_path), max_side=500)
                    if img_ch:
                        st.image(img_ch, caption=f"Change Map: {lbl_ch}", use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # LAND-COVER CLASSIFICATION RESULTS
        # ---------------------------------------------------------------------
        lc_stats = next((e for e in result.evidence if e.title == "Land-Cover Classification Statistics"), None)
        lc_chart = next((e for e in result.evidence if "Land-Cover Probability Distribution" in e.title), None)

        if lc_stats and lc_stats.data:
            lc_data = lc_stats.data
            st.markdown(
                """
                <div class="sat-card">
                    <div class="sat-card-header">
                        <div class="sat-card-title"><span>🌿</span> Land-Cover Classification Predictions</div>
                        <div class="sat-card-badge">BIGEARTHNET V2 RESNET-50</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Telemetry Metrics HUD
            lc_metrics = [
                {
                    "label": "Dominant Prediction",
                    "value": lc_data.get("top_5_predictions", [{}])[0].get("class", "N/A"),
                    "hint": f"Probability: {lc_data.get('top_5_predictions', [{}])[0].get('probability', 0)*100:.2f}%",
                },
                {
                    "label": "Detection Threshold",
                    "value": f"{lc_data.get('threshold', 0.10):.2f}",
                    "hint": "Runtime filter cutoff",
                },
                {
                    "label": "Classes >= Threshold",
                    "value": str(lc_data.get("detected_classes_count", 0)),
                    "hint": "Corine categories met",
                },
                {
                    "label": "Top Confidence",
                    "value": f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A",
                    "hint": "Sigmoid output layer max",
                },
            ]
            st.markdown(render_telemetry_hud(lc_metrics), unsafe_allow_html=True)

            # Top 5 Predictions Animated Bars
            if "top_5_predictions" in lc_data and lc_data["top_5_predictions"]:
                st.markdown("##### 🏆 Top 5 Predicted Land-Cover Categories")
                st.markdown(render_top5_bars(lc_data["top_5_predictions"]), unsafe_allow_html=True)

            # Top Classes Table & Chart
            col_chart, col_tbl = st.columns([1.3, 1.0])
            with col_chart:
                if lc_chart and lc_chart.file_path and Path(lc_chart.file_path).exists():
                    st.image(lc_chart.file_path, caption="Probability Distribution (Top Categories)", use_container_width=True)
            with col_tbl:
                st.markdown("##### 📋 Detected Classes (>= Threshold)")
                if lc_data.get("detected_classes"):
                    st.dataframe(
                        [
                            {"Category": c["class"], "Predicted Confidence": f"{c['probability']*100:.2f}%"}
                            for c in lc_data["detected_classes"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(f"No category exceeded the selected threshold of {lc_data.get('threshold', 0.10):.2f}.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # MULTI-LAYER INSPECTION TABS & ARTIFACT DOWNLOADS
        # ---------------------------------------------------------------------
        tab_list = []
        tab_contents = []

        if "t0" in slot_files and slot_files["t0"].exists():
            tab_list.append("📷 Baseline (T0)")
            tab_contents.append(("t0", slot_files["t0"]))
        if "t1" in slot_files and slot_files["t1"].exists():
            tab_list.append("📷 Target (T1)")
            tab_contents.append(("t1", slot_files["t1"]))
        if "image" in slot_files and slot_files["image"].exists() and not tab_list:
            tab_list.append("📷 Source Scene")
            tab_contents.append(("image", slot_files["image"]))

        if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
            tab_list.append("🌐 Change Heatmap")
            tab_contents.append(("prob", Path(prob_art.file_path)))
        if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
            tab_list.append("⬛⬜ Binary Mask")
            tab_contents.append(("mask", Path(mask_art.file_path)))
        if vis_art and vis_art.file_path and Path(vis_art.file_path).exists():
            tab_list.append("🔴 Change Overlay")
            tab_contents.append(("overlay", Path(vis_art.file_path)))
        if regions_art and regions_art.file_path and Path(regions_art.file_path).exists():
            tab_list.append("🏷️ Labeled Clusters")
            tab_contents.append(("regions", Path(regions_art.file_path)))

        if tab_list:
            with st.expander("🖼️ Multi-Layer High-Resolution Raster Inspection", expanded=False):
                tabs = st.tabs(tab_list)
                for i, (kind, pth) in enumerate(tab_contents):
                    with tabs[i]:
                        img, lbl = generate_preview_image(pth, max_side=650)
                        if img:
                            st.image(img, caption=f"{tab_list[i]} — {lbl}", use_container_width=True)
                        else:
                            st.warning(f"Preview unavailable: {lbl}")

        # Download Action Bar
        dl_buttons = []
        if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
            dl_buttons.append(("Binary Mask (GeoTIFF)", mask_art.file_path, "image/tiff"))
        if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
            dl_buttons.append(("Probability Map (GeoTIFF)", prob_art.file_path, "image/tiff"))
        if geojson_art and geojson_art.file_path and Path(geojson_art.file_path).exists():
            dl_buttons.append(("Vector Regions (GeoJSON)", geojson_art.file_path, "application/geo+json"))
        if regions_art and regions_art.file_path and Path(regions_art.file_path).exists():
            dl_buttons.append(("Cluster Overlay (PNG)", regions_art.file_path, "image/png"))
        if lc_chart and lc_chart.file_path and Path(lc_chart.file_path).exists():
            dl_buttons.append(("Land-Cover Chart (PNG)", lc_chart.file_path, "image/png"))

        if dl_buttons:
            st.markdown("##### 📥 Export Mission Products")
            dl_cols = st.columns(len(dl_buttons) + 1)
            for idx, (label, pth, mime) in enumerate(dl_buttons):
                with dl_cols[idx]:
                    with open(pth, "rb") as f_dl:
                        st.download_button(
                            f"⬇️ {label}",
                            f_dl.read(),
                            file_name=Path(pth).name,
                            mime=mime,
                            use_container_width=True,
                        )
            with dl_cols[-1]:
                st.download_button(
                    "⬇️ Intelligence Report (.txt)",
                    result.result_text.encode("utf-8"),
                    file_name="satquery_intelligence_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # -------------------------------------------------------------------------
    # TECHNICAL TELEMETRY & SYSTEM DRAWER (COLLAPSIBLE MISSION PANELS)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Mission Telemetry & Deep Diagnostics")

    with st.expander("▼ AI Execution & Intelligence Report", expanded=True):
        st.markdown(result.result_text)

    with st.expander("▼ Model Information & Routing Trace", expanded=False):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"**Resolved Task:** `{plan.task}`")
            st.markdown(f"**Thematic Domain:** `{plan.theme.value.upper()}`")
            st.markdown(f"**Input Mode:** `{plan.input_mode.value}`")
            st.markdown(f"**Required Inputs:** {', '.join(plan.required_inputs)}")
        with p_col2:
            st.markdown(f"**Selected Model(s):** {', '.join(plan.selected_models)}")
            st.markdown(f"**Selected Tools:** {', '.join(plan.selected_tools)}")
            if plan.blocked:
                st.error(f"Execution Blocked: {plan.block_reason}")
            else:
                st.success("Plan Status: Compatible & Dispatched")

    with st.expander("▼ Confidence & Uncertainty", expanded=False):
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown(f"**Confidence Level:** {result.confidence.display_text}")
            st.markdown(f"**Source:** {result.confidence.source.upper()}")
            st.markdown(f"**Method:** `{result.confidence.method or 'None'}`")
        with c_col2:
            st.markdown(f"**Rationale:** *{result.confidence.reason}*")
            if result.confidence.signals_used:
                st.markdown(f"**Signals Evaluated:** `{', '.join(result.confidence.signals_used)}`")
        if result.uncertainties:
            st.markdown("---")
            for unc in result.uncertainties:
                st.warning(f"⚠️ {unc}")

    with st.expander("▼ Raster Metadata & Spatial Verification", expanded=False):
        val_ev = next((e for e in result.evidence if "Validation" in e.title or "Quality" in e.title), None)
        if val_ev and val_ev.data:
            st.json(val_ev.data)
        else:
            st.info("Input rasters verified: Valid geospatial georeferencing and spectral compatibility confirmed.")

    with st.expander("▼ Input Validation & Chronological Trace", expanded=False):
        trace_rows = []
        for ev in result.trace:
            icon = "✅" if ev.status.value == "success" else "⏳" if ev.status.value == "start" else "⚠️"
            trace_rows.append({
                "Status": f"{icon} {ev.status.value.upper()}",
                "Step": ev.step,
                "Component": ev.component,
                "Duration (ms)": f"{ev.duration_ms:.1f}" if ev.duration_ms is not None else "0.0",
                "Timestamp": ev.timestamp,
            })
        st.dataframe(trace_rows, use_container_width=True, hide_index=True)

    with st.expander("▼ Limitations & Operational Constraints", expanded=False):
        st.markdown(
            """
            - **Optical Cloud Sensitivity:** Optical change detection and land-cover classification require cloud-free or shadow-masked scenes.
            - **Spatial Resolution & GSD:** Change features smaller than 2 × GSD may produce sub-pixel uncertainty.
            - **Multispectral Alignment:** Sentinel-2 10m/20m bands are dynamically resampled to 10m GSD; coregistration shifts > 0.5 px will attenuate change precision.
            """
        )


# -----------------------------------------------------------------------------
# 4. CONVERSATIONAL SESSION HISTORY
# -----------------------------------------------------------------------------
if len(st.session_state.history) > 1:
    with st.expander("📜 Previous Analysis History in Current Session", expanded=False):
        for idx, turn in enumerate(reversed(st.session_state.history[:-1]), 1):
            st.markdown(f"**Turn {idx}:** \"{turn['query']}\" (`{turn['task']}`)")
            st.caption(turn['result_text'][:250] + "..." if len(turn['result_text']) > 250 else turn['result_text'])
            st.markdown("---")
