from pathlib import Path
import tempfile
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
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Upload cache directory
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
st.title("🛰️ SATQUERY AI")
st.markdown("### **Interactive Multimodal Remote-Sensing Assistant**")
st.markdown(
    "*Agentic vision-language query orchestration for multimodal Earth observation rasters. "
    "All metadata, validation, and confidence scores are run-derived without generic LLM hallucination.*"
)
st.divider()


# -------------------------------------------------------------
# SIDEBAR: CONFIGURATION & SESSION
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Environment: **Phase 2 Development**\nSpecialist Checkpoints: **Unloaded / Not Configured**")

    if st.button("🗑️ Clear Session & Uploads"):
        st.session_state.history.clear()
        st.session_state.last_result = None
        st.session_state.current_query = ""
        st.rerun()

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("- **Router:** Deterministic Rule Engine")
    st.markdown("- **Engine:** AgentController (10-Stage Pipeline)")
    st.markdown("- **Registries:** ModelRegistry & ToolRegistry")
    st.markdown("- **Confidence:** Honest Null-Safe Estimator")


# -------------------------------------------------------------
# 1. DATA INPUT
# -------------------------------------------------------------
st.header("1. Data Input")

mode_selection = st.radio(
    "Select Input Mode:",
    ["Single Image", "Bi-Temporal Pair", "Optical + SAR Pair"],
    horizontal=True,
)

slots: list[SlotAssignment] = []
slot_files: dict[str, Path] = {}

col_in1, col_in2 = st.columns(2)

if mode_selection == "Single Image":
    input_mode = InputMode.I1_SINGLE_OPTICAL
    with col_in1:
        single_upload = st.file_uploader(
            "Upload Satellite Image (GeoTIFF / TIFF):",
            type=["tif", "tiff", "geotiff"],
            key="single_uploader",
        )
        if single_upload:
            p = save_upload(single_upload)
            slots.append(SlotAssignment(slot_id="image", file_path=str(p)))
            slot_files["image"] = p

elif mode_selection == "Bi-Temporal Pair":
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
    cd_col1, cd_col2, cd_col3 = st.columns(3)
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
        st.markdown("**Tiled Inference:** 256×256 tiles")
        st.caption("Sliding-window with cosine blending (no downsampling)")
    with cd_col3:
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
# 2. IMAGE INFORMATION & PREVIEWS
# -------------------------------------------------------------
if slots:
    st.header("2. Image Information & Visual Preview")
    preview_cols = st.columns(len(slots))

    for idx, slot in enumerate(slots):
        file_p = Path(slot.file_path)
        meta = inspect_raster(file_p)
        quality = assess_image_quality(file_p, meta)
        modality = detect_modality(meta)

        with preview_cols[idx]:
            st.markdown(f"#### Slot: {slot.slot_id} ({meta.filename})")

            # Image Preview
            img_preview, label_txt = generate_preview_image(file_p, max_side=400)
            if img_preview:
                st.image(img_preview, caption=label_txt, use_container_width=True)
            else:
                st.warning(f"Preview unavailable: {label_txt}")

            # Real Metadata Display
            st.markdown("**Raster Metadata:**")
            meta_data = {
                "Dimensions": f"{meta.width} x {meta.height}" if meta.width else "Not available",
                "Bands": str(meta.band_count) if meta.band_count else "Not available",
                "Data Type": str(meta.dtype) if meta.dtype else "Not available",
                "Format": str(meta.format) if meta.format else "Not available",
                "CRS": str(meta.crs) if meta.crs else "Not available",
                "Resolution": f"{meta.resolution[0]:.6f}, {meta.resolution[1]:.6f}" if meta.resolution else "Not available",
                "NoData": str(meta.nodata) if meta.nodata is not None else "Not available",
                "Detected Modality": modality.value,
                "Image Quality": f"{quality.level.value} ({', '.join(quality.reasons)})",
            }
            st.table(meta_data)
else:
    st.info("Upload raster file(s) above to inspect real metadata and preview scenes.")


# -------------------------------------------------------------
# 3. NATURAL LANGUAGE QUERY
# -------------------------------------------------------------
st.header("3. Natural Language Query")

st.markdown("**Example queries (click to populate):**")
ex_cols = st.columns(3)
with ex_cols[0]:
    if st.button("Describe scene land-cover", use_container_width=True):
        st.session_state.current_query = "Describe the land-cover and major objects visible in this image."
with ex_cols[1]:
    if st.button("Check for roads (VQA)", use_container_width=True):
        st.session_state.current_query = "Is there a road in this image?"
with ex_cols[2]:
    if st.button("Highlight water body", use_container_width=True):
        st.session_state.current_query = "Highlight the water body."

ex_cols2 = st.columns(3)
with ex_cols2[0]:
    if st.button("Detect temporal change", use_container_width=True):
        st.session_state.current_query = "What changed between these two dates?"
with ex_cols2[1]:
    if st.button("Built-up growth (Change VQA)", use_container_width=True):
        st.session_state.current_query = "Has the built-up area increased?"
with ex_cols2[2]:
    if st.button("Joint Optical + SAR analysis", use_container_width=True):
        st.session_state.current_query = "Use the optical and SAR images together to identify built-up and water-covered regions."

query_text = st.text_input(
    "Enter your question or task instruction for the Earth observation scene:",
    value=st.session_state.current_query,
    placeholder="e.g., Describe the land-cover visible in this image.",
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
        if mode_selection == "Bi-Temporal Pair":
            analysis_kwargs["change_threshold"] = change_threshold

        with st.spinner("Agent interpreting query, validating compatibility, and planning specialist execution..."):
            result: AnalysisResult = controller.analyze(
                query=query_text.strip(),
                slots=slots,
                input_mode=input_mode,
                **analysis_kwargs,
            )
            st.session_state.last_result = result
            st.session_state.history.append({
                "query": query_text,
                "task": result.plan.task,
                "theme": result.plan.theme.value,
                "result_text": result.result_text,
            })


# -------------------------------------------------------------
# DISPLAY RESULTS IF AVAILABLE
# -------------------------------------------------------------
result = st.session_state.last_result

if result:
    st.divider()

    # 4. AGENT PLAN
    st.header("4. Agent Plan")
    plan = result.plan

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Detected Task", plan.task)
        st.metric("Theme Domain", plan.theme.value)
    with col_p2:
        st.markdown(f"**Required Inputs:** {', '.join(plan.required_inputs)}")
        st.markdown(f"**Input Mode:** {plan.input_mode.value}")
        if plan.blocked:
            st.error(f"⛔ **Plan Blocked:** {plan.block_reason}")
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

    # 5. ANALYSIS RESULT
    st.header("5. Analysis Result")
    if "Specialist model not configured" in result.result_text:
        st.warning(f"### {result.result_text}")
    elif "Validation failed" in result.result_text or "Agent planning stopped" in result.result_text:
        st.error(f"### {result.result_text}")
    else:
        st.success(result.result_text)

        # Display dedicated Change Detection Model & Results if present
        stats_ev = next((e for e in result.evidence if e.evidence_type.value == "statistics"), None)
        if stats_ev and stats_ev.data:
            s_data = stats_ev.data

            st.markdown("#### 🛰️ Specialist Model Specifications")
            m_spec1, m_spec2, m_spec3, m_spec4, m_spec5, m_spec6 = st.columns(6)
            with m_spec1:
                st.metric("Model", "Open-CD BIT")
            with m_spec2:
                st.metric("Version", "r18-levir")
            with m_spec3:
                st.metric("Device", s_data.get("inference_device", "cpu").upper())
            with m_spec4:
                st.metric("Threshold", f"{s_data.get('change_threshold', 0.5):.2f}")
            with m_spec5:
                st.metric("Tile Size", f"{s_data.get('tile_size', 256)}×{s_data.get('tile_size', 256)}")
            with m_spec6:
                st.metric("Overlap", f"{int(s_data.get('tile_overlap', 0.25)*100)}%")

            st.markdown("#### 📊 Change Detection Results")
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
            with r_col1:
                if s_data.get("changed_area_m2") is not None:
                    st.metric("Changed Area", f"{s_data.get('changed_area_hectares', 0.0):,.2f} ha", help=f"{s_data.get('changed_area_m2', 0):,.1f} m²")
                else:
                    st.metric("Changed Area", "N/A (Unprojected)")
            with r_col2:
                st.metric("Change %", f"{s_data.get('percentage_changed', 0.0)}%")
            with r_col3:
                st.metric("Changed Pixels", f"{s_data.get('changed_pixels', 0):,}")
            with r_col4:
                st.metric("Confidence", f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A")
            with r_col5:
                t_total = s_data.get("timings_seconds", {}).get("total")
                st.metric("Processing Time", f"{t_total:.2f}s" if t_total is not None else "N/A")

    # 6. VISUAL EVIDENCE & DOWNLOADS
    st.header("6. Visual Evidence & Artifacts")
    
    # Check if this run generated change detection artifacts
    mask_art = next((e for e in result.evidence if "Binary Change Mask" in e.title), None)
    prob_art = next((e for e in result.evidence if "Probability" in e.title), None)
    vis_art = next((e for e in result.evidence if "Overlay" in e.title or "Visualization" in e.title), None)
    stats_ev = next((e for e in result.evidence if e.evidence_type.value == "statistics"), None)

    if mask_art or prob_art or vis_art:
        tab_t0, tab_t1, tab_prob, tab_mask, tab_overlay = st.tabs([
            "📷 T0 Image",
            "📷 T1 Image",
            "🌐 Change Probability",
            "⬛⬜ Binary Change Map",
            "🔴 Change Overlay",
        ])

        with tab_t0:
            if "t0" in slot_files and slot_files["t0"].exists():
                img_t0, _ = generate_preview_image(slot_files["t0"], max_side=600)
                if img_t0:
                    st.image(img_t0, caption="Date T0 ('Before') Image", use_container_width=True)

        with tab_t1:
            if "t1" in slot_files and slot_files["t1"].exists():
                img_t1, _ = generate_preview_image(slot_files["t1"], max_side=600)
                if img_t1:
                    st.image(img_t1, caption="Date T1 ('After') Image", use_container_width=True)

        with tab_prob:
            if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
                p_img, _ = generate_preview_image(prob_art.file_path, max_side=600)
                if p_img:
                    st.image(p_img, caption="Continuous Change Probability Map (0.0 to 1.0)", use_container_width=True)

        with tab_mask:
            if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
                m_img, _ = generate_preview_image(mask_art.file_path, max_side=600)
                if m_img:
                    st.image(m_img, caption="Georeferenced Binary Change Mask (0 = Unchanged, 1 = Changed)", use_container_width=True)

        with tab_overlay:
            if vis_art and vis_art.file_path and Path(vis_art.file_path).exists():
                o_img, _ = generate_preview_image(vis_art.file_path, max_side=600)
                if o_img:
                    st.image(o_img, caption="Visual Overlay: Changed Regions Highlighted in Red", use_container_width=True)

        st.markdown("#### 📥 Download Artifacts")
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            if mask_art and mask_art.file_path and Path(mask_art.file_path).exists():
                with open(mask_art.file_path, "rb") as f_mask:
                    st.download_button(
                        "⬇️ Binary Mask (GeoTIFF)",
                        f_mask.read(),
                        file_name=Path(mask_art.file_path).name,
                        mime="image/tiff",
                        use_container_width=True,
                    )
        with dl_col2:
            if prob_art and prob_art.file_path and Path(prob_art.file_path).exists():
                with open(prob_art.file_path, "rb") as f_prob:
                    st.download_button(
                        "⬇️ Probability Map (GeoTIFF)",
                        f_prob.read(),
                        file_name=Path(prob_art.file_path).name,
                        mime="image/tiff",
                        use_container_width=True,
                    )
        with dl_col3:
            if stats_ev and stats_ev.data:
                stats_json_str = json.dumps(stats_ev.data, indent=2)
                st.download_button(
                    "⬇️ Statistics (JSON)",
                    stats_json_str,
                    file_name="change_statistics.json",
                    mime="application/json",
                    use_container_width=True,
                )
        with dl_col4:
            if vis_art and vis_art.file_path and Path(vis_art.file_path).exists():
                with open(vis_art.file_path, "rb") as f_vis:
                    st.download_button(
                        "⬇️ Overlay (PNG)",
                        f_vis.read(),
                        file_name=Path(vis_art.file_path).name,
                        mime="image/png",
                        use_container_width=True,
                    )
    elif result.evidence:
        ev_cols = st.columns(min(len(result.evidence), 4))
        for i, ev in enumerate(result.evidence):
            with ev_cols[i % len(ev_cols)]:
                st.markdown(f"**{ev.title}**")
                st.caption(ev.description)
                if ev.file_path and Path(ev.file_path).exists():
                    p_img, _ = generate_preview_image(ev.file_path, max_side=300)
                    if p_img:
                        st.image(p_img, use_container_width=True)
                if ev.metadata:
                    st.json(ev.metadata, expanded=False)
    else:
        st.info("No spatial evidence artifacts produced for this run.")

    # 7 & 8. CONFIDENCE & UNCERTAINTY
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.header("7. Confidence")
        st.markdown(f"**Confidence Level:** {result.confidence.display_text}")
        st.markdown(f"**Source:** {result.confidence.source}")
        st.markdown(f"**Reason:** *{result.confidence.reason}*")
        st.caption("Confidence scores are derived exclusively from running model activations. No default or placeholder scores are displayed.")

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
# 10. CONVERSATIONAL SESSION HISTORY
# -------------------------------------------------------------
if len(st.session_state.history) > 1:
    st.divider()
    st.header("10. Conversational Session History")
    for turn in reversed(st.session_state.history[:-1]):
        with st.expander(f"Previous Query: {turn['query']} ({turn['task']})"):
            st.markdown(f"**Theme:** {turn['theme']}")
            st.markdown(f"**Result:** {turn['result_text']}")
