from pathlib import Path
from satquery.agent.controller import AgentController
from satquery.agent.router import QueryPlanner, classify_theme, route_query
from satquery.domain.schemas import (
    InputMode,
    Modality,
    QueryTheme,
    SlotAssignment,
    TaskType,
    ValidationStatus,
)
from satquery.preprocess.raster import inspect_raster


def test_1_single_vqa_routing(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    plan1 = route_query("Is there a road in this image?", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan1.task == TaskType.SINGLE_VQA.value
    assert plan1.blocked is False

    plan2 = route_query("What objects are visible?", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan2.task == TaskType.SINGLE_VQA.value
    assert plan2.blocked is False


def test_2_single_caption_routing(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    query = "Describe the land-cover and major objects visible in this image."
    plan = route_query(query, slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan.task == TaskType.SINGLE_CAPTION.value
    assert "geochat_caption" in plan.selected_models


def test_3_single_grounding_routing(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    query = "Highlight the water body."
    plan = route_query(query, slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan.task == TaskType.SINGLE_GROUNDING.value
    assert "geochat_grounding" in plan.selected_models


def test_4_bitemporal_change_routing(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    metas = {"t0": meta, "t1": meta}

    plan1 = route_query("What changed between these two dates?", slots, metas, InputMode.I4_BITEMPORAL_PAIR)
    assert plan1.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert "opencd_bit_change" in plan1.selected_models

    plan2 = route_query("Where did the change occur?", slots, metas, InputMode.I4_BITEMPORAL_PAIR)
    assert plan2.task == TaskType.BI_TEMPORAL_CHANGE.value


def test_5_change_vqa_routing(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    metas = {"t0": meta, "t1": meta}

    query = "Has the built-up area increased?"
    plan = route_query(query, slots, metas, InputMode.I4_BITEMPORAL_PAIR)
    assert plan.task == TaskType.CHANGE_VQA.value
    assert "cdvqa_baseline" in plan.selected_models


def test_6_optical_sar_analysis_routing(optical_geotiff: Path, sar_geotiff: Path):
    opt_meta = inspect_raster(optical_geotiff)
    sar_meta = inspect_raster(sar_geotiff)
    slots = [
        SlotAssignment(slot_id="optical", file_path=str(optical_geotiff), declared_modality=Modality.OPTICAL),
        SlotAssignment(slot_id="sar", file_path=str(sar_geotiff), declared_modality=Modality.SAR),
    ]
    metas = {"optical": opt_meta, "sar": sar_meta}

    query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    plan = route_query(query, slots, metas, InputMode.I3_OPTICAL_SAR_PAIR)
    assert plan.task == TaskType.OPTICAL_SAR_ANALYSIS.value
    assert plan.blocked is False
    assert "bigearthnet_resnet50_s1s2" in plan.selected_models


def test_7_theme_classification():
    assert classify_theme("What crop types are planted in this farmland?") == QueryTheme.AGRICULTURE
    assert classify_theme("Assess flood damage and disaster impact") == QueryTheme.DISASTER
    assert classify_theme("Identify building expansion and city growth") == QueryTheme.URBAN
    assert classify_theme("Monitor river water levels and deforestation") == QueryTheme.ENVIRONMENT
    assert classify_theme("Is there a bridge or railway here?") == QueryTheme.INFRASTRUCTURE
    assert classify_theme("Analyze this random satellite scene") == QueryTheme.GENERAL


def test_8_invalid_task_input_combinations(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    # Only 1 image provided
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    # Temporal query with only 1 image must NOT route to BI_TEMPORAL_CHANGE
    plan_temp = route_query("What changed between these two dates?", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan_temp.task != TaskType.BI_TEMPORAL_CHANGE.value
    assert plan_temp.blocked is True
    assert "requires two images" in plan_temp.block_reason

    # Optical-SAR query with only 1 image
    plan_osar = route_query("Use the optical and SAR images together", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan_osar.blocked is True
    assert "requires both an optical and a SAR image" in plan_osar.block_reason


def test_9_model_registry_selection(optical_geotiff: Path):
    controller = AgentController()
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    plan = route_query("Describe the land-cover", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    for mid in plan.selected_models:
        adapter = controller.model_registry.get(mid)
        assert adapter is not None
        assert adapter.status == "not_configured"


def test_10_tool_registry_selection(optical_geotiff: Path):
    controller = AgentController()
    meta = inspect_raster(optical_geotiff)
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    metas = {"t0": meta, "t1": meta}

    plan = route_query("What changed between these two dates?", slots, metas, InputMode.I4_BITEMPORAL_PAIR)
    for tid in plan.selected_tools:
        tool = controller.tool_registry.get(tid)
        assert tool is not None
        assert tool.tool_id == tid


def test_11_execution_trace(optical_geotiff: Path):
    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    res = controller.analyze("Is there a road?", slots, InputMode.I1_SINGLE_OPTICAL)

    steps = [e.step for e in res.trace]
    assert "inspect_inputs" in steps
    assert "assess_quality" in steps
    assert "validate_compatibility" in steps
    assert "interpret_query" in steps
    assert "select_specialists" in steps
    assert "execute_specialists" in steps

    # Check that durations are recorded and no chain of thought is present
    for event in res.trace:
        assert not hasattr(event, "chain_of_thought")
        assert not hasattr(event, "thought")


def test_12_not_configured_model_behavior(optical_geotiff: Path):
    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    res = controller.analyze("Describe the scene", slots, InputMode.I1_SINGLE_OPTICAL)

    assert "Specialist model not configured" in res.result_text
    assert res.confidence.is_available is False
    assert res.confidence.display_text == "Confidence not available"


def test_13_agent_error_handling(corrupt_file: Path):
    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(corrupt_file))]
    res = controller.analyze("What is in this image?", slots, InputMode.I1_SINGLE_OPTICAL)

    # Must handle gracefully without crashing
    assert res.validation.status == ValidationStatus.FAIL
    assert "Validation failed" in res.result_text
    assert len(res.uncertainties) > 0


def test_14_agent_controller_planner_and_bitemporal_execution(optical_geotiff: Path):
    """Regression test: AgentController.planner exists and handles bitemporal change detection queries."""
    controller = AgentController()
    assert hasattr(controller, "planner")
    assert hasattr(controller, "plan")
    assert isinstance(controller.planner, QueryPlanner)

    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    query = "Compare these two satellite images and map significant changes."

    # Verify planner.plan execution
    preview_plan = controller.planner.plan(query=query, slots=slots, input_mode=InputMode.I4_BITEMPORAL_PAIR)
    assert preview_plan.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert "opencd_bit_change" in preview_plan.selected_models
    assert not preview_plan.blocked

    # Verify controller.plan matches
    direct_plan = controller.plan(query=query, slots=slots, input_mode=InputMode.I4_BITEMPORAL_PAIR)
    assert direct_plan.task == preview_plan.task
    assert direct_plan.selected_models == preview_plan.selected_models

    # Verify end-to-end analyze execution through Open-CD BIT
    res = controller.analyze(query, slots, InputMode.I4_BITEMPORAL_PAIR)
    assert res.plan.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert "Open-CD BIT" in res.result_text
    assert res.confidence.is_available is True
    assert res.confidence.method == "bit_softmax_mean_confidence"
    
    evidence_titles = [e.title for e in res.evidence]
    assert "Binary Change Mask (GeoTIFF)" in evidence_titles
    assert "Change Map Visual Overlay" in evidence_titles
    assert "Bi-Temporal Change Statistics" in evidence_titles

