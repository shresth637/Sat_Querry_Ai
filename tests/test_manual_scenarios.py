from pathlib import Path
from satquery.agent.controller import AgentController
from satquery.domain.schemas import (
    InputMode,
    Modality,
    SlotAssignment,
    TaskType,
    ValidationStatus,
)
from satquery.preprocess.preview import generate_preview_image


def test_scenario_1_and_2_single_image_vqa(optical_geotiff: Path):
    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]

    # Preview generation check
    preview, label = generate_preview_image(optical_geotiff)
    assert preview is not None
    assert "percentile" in label

    # Analysis check
    res = controller.analyze("Is there a road in this image?", slots, InputMode.I1_SINGLE_OPTICAL)
    assert res.plan.task == TaskType.SINGLE_VQA.value
    assert res.plan.blocked is False
    assert res.validation.status == ValidationStatus.PASS
    assert "Specialist model not configured" in res.result_text
    assert res.confidence.is_available is False
    assert len(res.trace) >= 6


def test_scenario_3_and_4_bitemporal_change(optical_geotiff: Path):
    controller = AgentController()
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]

    res = controller.analyze("What changed between these two dates?", slots, InputMode.I4_BITEMPORAL_PAIR)
    assert res.plan.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert res.plan.blocked is False
    assert res.validation.status == ValidationStatus.PASS
    assert "Specialist model not configured" in res.result_text


def test_scenario_5_and_6_optical_sar_pair(optical_geotiff: Path, sar_geotiff: Path):
    controller = AgentController()
    slots = [
        SlotAssignment(slot_id="optical", file_path=str(optical_geotiff), declared_modality=Modality.OPTICAL),
        SlotAssignment(slot_id="sar", file_path=str(sar_geotiff), declared_modality=Modality.SAR),
    ]

    res = controller.analyze(
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        slots,
        InputMode.I3_OPTICAL_SAR_PAIR,
    )
    assert res.plan.task == TaskType.OPTICAL_SAR_ANALYSIS.value
    assert res.plan.blocked is False
    assert res.validation.status == ValidationStatus.PASS
    assert "Specialist model not configured" in res.result_text


def test_scenario_7_invalid_file(corrupt_file: Path):
    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(corrupt_file))]

    # Preview generation should return None, not crash
    preview, label = generate_preview_image(corrupt_file)
    assert preview is None

    # Analysis should report validation failure gracefully
    res = controller.analyze("Describe this scene", slots, InputMode.I1_SINGLE_OPTICAL)
    assert res.validation.status == ValidationStatus.FAIL
    assert "Validation failed" in res.result_text


def test_scenario_8_wrong_number_of_images(optical_geotiff: Path):
    controller = AgentController()
    # Bi-temporal requires 2 images, but only 1 provided
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]

    res = controller.analyze("What changed between these two dates?", slots, InputMode.I4_BITEMPORAL_PAIR)
    assert res.validation.status == ValidationStatus.FAIL
    assert any(c.code == "SLOT_COUNT_MISMATCH" for c in res.validation.checks)
    assert "requires 2 image(s)" in res.validation.checks[0].message
