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
from satquery.preprocess.preview import generate_preview_image
from satquery.preprocess.raster import (
    assess_image_quality,
    detect_modality,
    inspect_raster,
)

st.set_page_config(
    page_title="SatQuery AI — Multi-Model Satellite Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.title("🛰️ SATQUERY AI — Satellite Intelligence Platform")
st.markdown("### **Multi-Modal Earth Observation & Specialist AI Inference**")
st.markdown(
    "*Agentic remote-sensing query interpretation, raster validation, and specialist neural network inference. "
    "Features Open-CD BIT bi-temporal change detection, BigEarthNet land-cover classification, and GeoChat VLM integration.*"
)
st.divider()


# -------------------------------------------------------------
# SIDEBAR: CONFIGURATION & SYSTEM STATUS
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    
    from satquery.models.manager import get_resource_manager
    rm = get_resource_manager()
    cuda_avail = rm.has_cuda()
    device_name = "NVIDIA CUDA" if cuda_avail else "CPU Fallback"

    st.markdown(f"**Execution Hardware:** `{device_name}`")
    if cuda_avail:
        vram = rm.get_vram_info()
        st.caption(f"Allocated: {vram.get('allocated_mb', 0):.0f} MB / Total: {vram.get('total_mb', 0):.0f} MB")
    else:
        st.caption("Host CPU execution active. No GPU OOM risk.")

    st.markdown("---")
    st.subheader("Specialist Models Catalog")
    reg = AgentController().model_registry
    for mid, adapter in reg.list_models():
        st_icon = "🟢" if adapter.status == "ready" else "🟡" if adapter.status == "unloaded" else "⚪"
        st.markdown(f"{st_icon} **{adapter.name}** (`{mid}`)")
        st.caption(f"Status: {adapter.status.upper()} | Caps: {', '.join(adapter.capabilities)}")

    st.markdown("---")
    if st.button("🗑️ Clear Session & Cache", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.last_result = None
        st.session_state.current_query = ""
        rm.cleanup_memory()
        st.rerun()


# -------------------------------------------------------------
# 1. DATA INPUT & MODE SELECTION
# -------------------------------------------------------------
st.header("1. Data Input & Workflow Mode")

mode_selection = st.radio(
    "Select Workflow Mode:",
    ["Single Satellite Image", "Bi-Temporal Pair (Change Detection)", "Optical + SAR Pair"],
    horizontal=True,
)

slots: list[SlotAssignment] = []
slot_files: dict[str, Path] = {}

col_in1, col_in2 = st.columns(2)

if mode_selection == "Single Satellite Image":
    input_mode = InputMode.I1_SINGLE_OPTICAL
    with col_in1:
        single_upload = st.file_uploader(
            "Upload Satellite Scene (GeoTIFF / TIFF):",
            type=["tif", "tiff", "geotiff"],
            key="single_uploader",
        )
        if single_upload:
            p = save_upload(single_upload)
            slots.append(SlotAssignment(slot_id="image", file_path=str(p)))
            slot_files["image"] = p

    st.markdown("##### ⚙️ Single-Image Model Parameters")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        classification_threshold = st.slider(
            "Land-Cover Confidence Threshold:",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Minimum class probability to report in BigEarthNet scene classification (default: 0.30).",
        )
    with p_col2:
        st.markdown("**Target Tasks:**")
        st.caption("• Scene Land-Cover Classification (BigEarthNet)\n• Visual Question Answering (GeoChat)\n• Referring Expression Grounding")

elif mode_selection == "Bi-Temporal Pair (Change Detection)":
    input_mode = InputMode.I4_BITEMPORAL_PAIR
    with col_in1:
        t0_upload = st.file_uploader(
            "Upload 'Before' Image (Date T0):",
            type=["tif", "tiff", "geotiff"],
            key="t0_uploader",
        )
        if t0_upload:
            p0 = save_upload(t0_upload)
            slots.append(SlotAssignment(slot_id="t0", file_path=str(p0)))
            slot_files["t0"] = p0
    with col_in2:
        t1_upload = st.file_uploader(
            "Upload 'After' Image (Date T1):",
            type=["tif", "tiff", "geotiff"],
            key="t1_uploader",
        )
        if t1_upload:
            p1 = save_upload(t1_upload)
            slots.append(SlotAssignment(slot_id="t1", file_path=str(p1)))
            slot_files["t1"] = p1

    st.markdown("##### ⚙️ Change Detection Parameters")
    cd_col1, cd_col2, cd_col3, cd_col4 = st.columns(4)
    with cd_col1:
        change_threshold = st.slider(
            "Change Threshold:",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Probability cutoff for classifying a pixel as changed (default 0.50).",
        )
    with cd_col2:
        min_change_region_pixels = st.slider(
            "Min Region Size (px):",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
            help="Filter out small noise clusters smaller than this pixel count (default 20 px).",
        )
    with cd_col3:
        st.markdown("**Tiled Inference:** 256×256 tiles")
        st.caption("Sliding-window with cosine blending (no downsampling)")
    with cd_col4:
        st.markdown("**Tile Overlap:** 25% (64 px)")
        st.caption("Seamless reconstruction without boundary seams")

else:  # Optical + SAR Pair
    input_mode = InputMode.I3_OPTICAL_SAR_PAIR
    with col_in1:
        opt_upload = st.file_uploader(
            "Upload Optical / Multispectral Image:",
            type=["tif", "tiff", "geotiff"],
            key="opt_uploader",
        )
        if opt_upload:
            p_opt = save_upload(opt_upload)
            slots.append(
                SlotAssignment(
                    slot_id="optical",
                    file_path=str(p_opt),
                    declared_modality=Modality.OPTICAL,
                )
            )
            slot_files["optical"] = p_opt
    with col_in2:
        sar_upload = st.file_uploader(
            "Upload Synthetic Aperture Radar (SAR) Image:",
            type=["tif", "tiff", "geotiff"],
            key="sar_uploader",
        )
        if sar_upload:
            p_sar = save_upload(sar_upload)
            slots.append(
                SlotAssignment(
                    slot_id="sar",
                    file_path=str(p_sar),
                    declared_modality=Modality.SAR,
                )
            )
            slot_files["sar"] = p_sar


# -------------------------------------------------------------
# 2. IMAGE METADATA & VISUAL PREVIEWS
# -------------------------------------------------------------
if slots:
    st.header("2. Raster Metadata & Scene Preview")
    preview_cols = st.columns(len(slots))

    for idx, slot in enumerate(slots):
        file_p = Path(slot.file_path)
        meta = inspect_raster(file_p)
        quality = assess_image_quality(file_p, meta)
        modality = detect_modality(meta)

        with preview_cols[idx]:
            st.markdown(f"#### Slot: `{slot.slot_id}` ({meta.filename})")

            img_preview, label_txt = generate_preview_image(file_p, max_side=400)
            if img_preview:
                st.image(img_preview, caption=label_txt, use_container_width=True)
            else:
                st.warning(f"Preview unavailable: {label_txt}")

            meta_data = {
                "Dimensions": f"{meta.width} × {meta.height}" if meta.width else "Not available",
                "Bands": str(meta.band_count) if meta.band_count else "Not available",
                "Data Type": str(meta.dtype) if meta.dtype else "Not available",
                "Format": str(meta.format) if meta.format else "Not available",
                "CRS": str(meta.crs) if meta.crs else "Not available",
                "Resolution": f"{meta.resolution[0]:.6f}, {meta.resolution[1]:.6f}" if meta.resolution else "Not available",
                "Detected Modality": modality.value,
                "Image Quality": f"{quality.level.value} ({', '.join(quality.reasons)})",
            }
            st.table(meta_data)
else:
    st.info("Upload satellite raster file(s) above to inspect real metadata and preview scenes.")


# -------------------------------------------------------------
# 3. NATURAL LANGUAGE QUERY
# -------------------------------------------------------------
st.header("3. Natural Language Query")

st.markdown("**Example queries (click to populate):**")
if mode_selection == "Single Satellite Image":
    ex_cols = st.columns(4)
    with ex_cols[0]:
        if st.button("Classify land-cover", use_container_width=True):
            st.session_state.current_query = "Classify the land-cover categories visible in this satellite scene."
    with ex_cols[1]:
        if st.button("Check for roads (VQA)", use_container_width=True):
            st.session_state.current_query = "Is there a road or runway in this image?"
    with ex_cols[2]:
        if st.button("Describe the scene", use_container_width=True):
            st.session_state.current_query = "Describe the scene and visible landscape features."
    with ex_cols[3]:
        if st.button("Locate water / buildings", use_container_width=True):
            st.session_state.current_query = "Locate and highlight the water bodies in this scene."
else:
    ex_cols = st.columns(3)
    with ex_cols[0]:
        if st.button("What changed between dates?", use_container_width=True):
            st.session_state.current_query = "What changed between these two dates?"
    with ex_cols[1]:
        if st.button("Identify new construction", use_container_width=True):
            st.session_state.current_query = "Identify areas where building construction occurred between T0 and T1."
    with ex_cols[2]:
        if st.button("Compare two images", use_container_width=True):
            st.session_state.current_query = "Compare these two satellite images and map significant changes."

query_text = st.text_input(
    "Enter your question or task instruction for the satellite imagery:",
    value=st.session_state.current_query,
    placeholder="e.g., Classify the land-cover categories visible in this scene.",
)

run_button = st.button("🚀 Run Agent Analysis", type="primary", use_container_width=True)

if run_button:
    if not slots:
        st.error("Please upload at least one image before running analysis.")
    elif not query_text.strip():
        st.error("Please enter a natural-language query or select an example above.")
    else:
        controller = AgentController()
        analysis_kwargs = {}
        if mode_selection == "Bi-Temporal Pair (Change Detection)":
            analysis_kwargs["change_threshold"] = change_threshold
            analysis_kwargs["min_change_region_pixels"] = min_change_region_pixels
        elif mode_selection == "Single Satellite Image":
            analysis_kwargs["threshold"] = classification_threshold

        try:
            with st.status("Executing SatQuery AI Multi-Model Intelligence...", expanded=True) as status_box:
                st.write("🔍 Stage 1: Inspecting rasters and verifying coordinate reference systems...")
                time.sleep(0.05)
                st.write("🧠 Stage 2: Parsing query intent and resolving specialist model pipeline...")
                time.sleep(0.05)
                st.write("🛰️ Stage 3: Executing neural inference and calculating spatial statistics...")
                result: AnalysisResult = controller.analyze(
                    query=query_text.strip(),
                    slots=slots,
                    input_mode=input_mode,
                    **analysis_kwargs,
                )
                st.write("📊 Stage 4: Formatting artifacts, GeoJSON vectors, and structured intelligence report...")
                status_box.update(label="Analysis completed successfully!", state="complete", expanded=False)

            st.session_state.last_result = result
            st.session_state.history.append({
                "query": query_text,
                "task": result.plan.task,
                "theme": result.plan.theme.value,
                "result_text": result.result_text,
            })
        except Exception as exc:
            st.error(f"❌ Analysis execution error: {str(exc)}")


# -------------------------------------------------------------
# 4. RESULTS DISPLAY
# -------------------------------------------------------------
result = st.session_state.last_result

if result:
    st.divider()

    # 4. AGENT PLAN
    st.header("4. Agent Execution Plan")
    plan = result.plan

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Resolved Task", plan.task)
        st.metric("Thematic Domain", plan.theme.value)
    with col_p2:
        st.markdown(f"**Required Inputs:** {', '.join(plan.required_inputs)}")
        st.markdown(f"**Input Mode:** {plan.input_mode.value}")
        if plan.blocked:
            st.error(f"⛔ **Execution Blocked:** {plan.block_reason}")
        else:
            st.success("✅ **Plan Status:** Compatible")
    with col_p3:
        reg = AgentController().model_registry
        model_statuses = []
        for mid in plan.selected_models:
            m = reg.get(mid)
            st_text = m.status.upper() if m else "UNKNOWN"
            model_statuses.append(f"`{mid}` ({st_text})")
        st.markdown(f"**Selected Model(s):** {', '.join(model_statuses)}")
        st.markdown(f"**Selected Tools:** {', '.join(plan.selected_tools)}")

    # 5. ANALYSIS RESULT & SUMMARY
    st.header("5. Technical Intelligence Report")
    if "Specialist model not configured" in result.result_text:
        st.warning(f"### {result.result_text}")
    elif "Validation failed" in result.result_text or "Agent planning stopped" in result.result_text:
        st.error(f"### {result.result_text}")
    else:
        st.markdown(result.result_text)

        # Show Change Detection metric cards if present
        stats_ev = next((e for e in result.evidence if e.title == "Bi-Temporal Change Statistics"), None)
        if stats_ev and stats_ev.data:
            s_data = stats_ev.data
            st.markdown("#### 📊 Change Detection Quantitative Metrics")
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
            with r_col1:
                if s_data.get("changed_area_m2") is not None:
                    st.metric("Changed Area", f"{s_data.get('changed_area_hectares', 0.0):,.2f} ha", help=f"{s_data.get('changed_area_m2', 0):,.1f} m²")
                else:
                    st.metric("Changed Area", "N/A (Unprojected)")
            with r_col2:
                st.metric("Change %", f"{s_data.get('percentage_changed', 0.0)}%")
            with r_col3:
                st.metric("Change Regions", f"{s_data.get('regions_count', 0):,}")
            with r_col4:
                st.metric("Confidence", f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A")
            with r_col5:
                t_total = s_data.get("timings_seconds", {}).get("total")
                st.metric("Processing Time", f"{t_total:.2f}s" if t_total is not None else "N/A")

        # Show Land-Cover metric cards if present
        lc_stats = next((e for e in result.evidence if e.title == "Land-Cover Classification Statistics"), None)
        if lc_stats and lc_stats.data:
            lc_data = lc_stats.data
            st.markdown("#### 🌿 Land-Cover Classification Metrics")
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.metric("Model", lc_data.get("model", "BigEarthNet v2 ResNet-50"))
            with l_col2:
                st.metric("Threshold", f"{lc_data.get('threshold', 0.30):.2f}")
            with l_col3:
                st.metric("Detected Classes", f"{lc_data.get('detected_classes_count', 0)}")
            with l_col4:
                st.metric("Top Confidence", f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A")

            if "top_5_predictions" in lc_data and lc_data["top_5_predictions"]:
                st.markdown("##### 🏆 Top 5 Predicted Land-Cover Categories")
                t5_cols = st.columns(len(lc_data["top_5_predictions"]))
                for col, item in zip(t5_cols, lc_data["top_5_predictions"]):
                    with col:
                        st.metric(label=item["class"], value=f"{item['probability']*100:.1f}%")

    # 6. VISUAL EVIDENCE & ARTIFACTS
    st.header("6. Visual Evidence & Generated Artifacts")

    # Group evidence items
    mask_art = next((e for e in result.evidence if "Binary Change Mask" in e.title), None)
    prob_art = next((e for e in result.evidence if "Probability Map" in e.title), None)
    vis_art = next((e for e in result.evidence if e.title == "Change Map Visual Overlay"), None)
    regions_art = next((e for e in result.evidence if "Bounding Boxes" in e.title), None)
    geojson_art = next((e for e in result.evidence if "GeoJSON" in e.title), None)
    lc_chart = next((e for e in result.evidence if "Land-Cover Probability Distribution" in e.title), None)
    grd_overlay = next((e for e in result.evidence if "Spatial Grounding Visual Overlay" in e.title), None)

    # Dynamic Tabs
    tab_list = []
    tab_contents = []

    if "t0" in slot_files and slot_files["t0"].exists():
        tab_list.append("📷 T0 Image")
        tab_contents.append(("t0", slot_files["t0"]))
    if "t1" in slot_files and slot_files["t1"].exists():
        tab_list.append("📷 T1 Image")
        tab_contents.append(("t1", slot_files["t1"]))
    if "image" in slot_files and slot_files["image"].exists() and not tab_list:
        tab_list.append("📷 Source Scene")
        tab_contents.append(("image", slot_files["image"]))

    if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
        tab_list.append("🌐 Change Probability")
        tab_contents.append(("prob", Path(prob_art.file_path)))
    if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
        tab_list.append("⬛⬜ Binary Change Map")
        tab_contents.append(("mask", Path(mask_art.file_path)))
    if vis_art and vis_art.file_path and Path(vis_art.file_path).exists():
        tab_list.append("🔴 Change Overlay")
        tab_contents.append(("overlay", Path(vis_art.file_path)))
    if regions_art and regions_art.file_path and Path(regions_art.file_path).exists():
        tab_list.append("🏷️ Change Regions")
        tab_contents.append(("regions", Path(regions_art.file_path)))
    if lc_chart and lc_chart.file_path and Path(lc_chart.file_path).exists():
        tab_list.append("📊 Land-Cover Classes")
        tab_contents.append(("lc_chart", Path(lc_chart.file_path)))
    if grd_overlay and grd_overlay.file_path and Path(grd_overlay.file_path).exists():
        tab_list.append("🎯 Grounded Bounding Boxes")
        tab_contents.append(("grounding", Path(grd_overlay.file_path)))

    if tab_list:
        tabs = st.tabs(tab_list)
        for i, (kind, pth) in enumerate(tab_contents):
            with tabs[i]:
                img, lbl = generate_preview_image(pth, max_side=650)
                if img:
                    st.image(img, caption=f"{tab_list[i]} — {lbl}", use_container_width=True)
                else:
                    st.warning(f"Preview unavailable: {lbl}")

    # Artifact Downloads Section
    st.markdown("#### 📥 Download Output Artifacts")
    dl_buttons = []
    if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
        dl_buttons.append(("Binary Mask (GeoTIFF)", mask_art.file_path, "image/tiff"))
    if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
        dl_buttons.append(("Probability Map (GeoTIFF)", prob_art.file_path, "image/tiff"))
    if geojson_art and geojson_art.file_path and Path(geojson_art.file_path).exists():
        dl_buttons.append(("Vector Regions (GeoJSON)", geojson_art.file_path, "application/geo+json"))
    if regions_art and regions_art.file_path and Path(regions_art.file_path).exists():
        dl_buttons.append(("Labeled Regions (PNG)", regions_art.file_path, "image/png"))
    if lc_chart and lc_chart.file_path and Path(lc_chart.file_path).exists():
        dl_buttons.append(("Land-Cover Chart (PNG)", lc_chart.file_path, "image/png"))

    if dl_buttons:
        dl_cols = st.columns(len(dl_buttons))
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
    else:
        st.caption("No downloadable file artifacts generated for this task.")

    # 7 & 8. CONFIDENCE & UNCERTAINTY
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.header("7. Confidence Estimation")
        st.markdown(f"**Confidence Level:** {result.confidence.display_text}")
        st.markdown(f"**Source:** {result.confidence.source}")
        st.markdown(f"**Method:** `{result.confidence.method or 'None'}`")
        st.markdown(f"**Reason:** *{result.confidence.reason}*")
        st.caption("Confidence scores are calibrated from active model activations without synthetic inflation.")

    with c_col2:
        st.header("8. Input Uncertainty Analysis")
        if result.uncertainties:
            for unc in result.uncertainties:
                st.warning(f"⚠️ {unc}")
        else:
            st.success("No input data or spatial uncertainties detected.")

    # 9. EXECUTION TRACE
    st.header("9. Technical Execution Trace")
    st.caption("Inspectable technical execution steps (duration, component, status). No internal chain-of-thought is exposed.")
    trace_rows = []
    for ev in result.trace:
        icon = "✅" if ev.status.value == "success" else "⏳" if ev.status.value == "start" else "⚠️" if ev.status.value == "warn" else "❌"
        trace_rows.append({
            "Status": f"{icon} {ev.status.value.upper()}",
            "Step": ev.step,
            "Component": ev.component,
            "Duration (ms)": f"{ev.duration_ms:.1f}" if ev.duration_ms is not None else "0.0",
            "Timestamp": ev.timestamp,
        })
    st.dataframe(trace_rows, use_container_width=True)


# -------------------------------------------------------------
# 10. SESSION HISTORY
# -------------------------------------------------------------
if len(st.session_state.history) > 1:
    st.divider()
    st.header("10. Conversational Session History")
    for turn in reversed(st.session_state.history[:-1]):
        with st.expander(f"Previous Query: {turn['query']} ({turn['task']})"):
            st.markdown(f"**Theme:** {turn['theme']}")
            st.markdown(f"**Result:** {turn['result_text']}")
