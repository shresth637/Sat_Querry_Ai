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
        with st.spinner("Agent interpreting query, validating compatibility, and planning specialist execution..."):
            result: AnalysisResult = controller.analyze(
                query=query_text.strip(),
                slots=slots,
                input_mode=input_mode,
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

        # Display dedicated Change Statistics if present
        stats_ev = next((e for e in result.evidence if e.evidence_type.value == "statistics"), None)
        if stats_ev and stats_ev.data:
            st.markdown("#### 📊 Change Detection Statistics")
            s_data = stats_ev.data
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Changed Pixels", f"{s_data.get('changed_pixels', 0):,}")
            with m_col2:
                st.metric("Total Pixels", f"{s_data.get('total_pixels', 0):,}")
            with m_col3:
                st.metric("Percentage Changed", f"{s_data.get('percentage_changed', 0.0)}%")
            with m_col4:
                if s_data.get("changed_area_sq_m") is not None:
                    st.metric("Changed Area", f"{s_data.get('changed_area_sq_m', 0.0):,.1f} m² ({s_data.get('changed_area_ha', 0.0):.2f} ha)")
                else:
                    st.metric("Unchanged Pixels", f"{s_data.get('unchanged_pixels', 0):,}")

    # 6. VISUAL EVIDENCE
    st.header("6. Visual Evidence")
    if result.evidence:
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
