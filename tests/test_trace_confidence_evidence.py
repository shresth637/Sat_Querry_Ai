from satquery.confidence.estimator import build_confidence_report
from satquery.domain.schemas import (
    EvidenceType,
    TraceStatus,
)
from satquery.evidence.builder import (
    create_bounding_box_evidence,
    create_change_map_evidence,
    create_mask_evidence,
    create_metadata_evidence,
    create_preview_evidence,
    create_source_image_evidence,
    create_statistics_evidence,
)
from satquery.preprocess.raster import inspect_raster
from satquery.trace.tracer import Tracer


def test_15_trace_events():
    tracer = Tracer()
    assert len(tracer.get_events()) == 0

    # Test span
    with tracer.span(step="inspect_inputs", component="preprocessor", parameters={"files": 1}):
        pass

    events = tracer.get_events()
    assert len(events) == 2
    assert events[0].step == "inspect_inputs"
    assert events[0].status == TraceStatus.START
    assert events[1].status == TraceStatus.SUCCESS
    assert events[1].duration_ms is not None
    assert events[1].duration_ms >= 0.0

    # Ensure no chain of thought or internal reasoning is leaked
    for e in events:
        assert not hasattr(e, "thought")
        assert not hasattr(e, "chain_of_thought")


def test_16_confidence_with_null_score():
    # Null score -> "Confidence not available"
    null_conf = build_confidence_report(score=None)
    assert null_conf.score is None
    assert null_conf.is_available is False
    assert null_conf.display_text == "Confidence not available"
    assert null_conf.source == "not_available"

    # Actual model score -> formatted display
    valid_conf = build_confidence_report(
        score=0.914,
        source="model",
        method="softmax_margin",
        signals_used=["top1_logit", "top2_logit"],
    )
    assert valid_conf.score == 0.914
    assert valid_conf.is_available is True
    assert "91.4" in valid_conf.display_text
    assert valid_conf.source == "model"
    assert valid_conf.method == "softmax_margin"


def test_17_evidence_schema(optical_geotiff):
    meta = inspect_raster(optical_geotiff)

    # 1. Source image
    ev_src = create_source_image_evidence(str(optical_geotiff), title="Optical Scene", crs=meta.crs)
    assert ev_src.evidence_type == EvidenceType.SOURCE_IMAGE
    assert ev_src.file_path == str(optical_geotiff)

    # 2. Preview
    ev_prev = create_preview_evidence("outputs/preview.png", title="RGB Preview")
    assert ev_prev.evidence_type == EvidenceType.PREVIEW

    # 3. Mask
    ev_mask = create_mask_evidence(title="Water Mask", data={"class": "water", "pixels": 1024})
    assert ev_mask.evidence_type == EvidenceType.MASK

    # 4. Bounding box
    boxes = [{"label": "aircraft", "box": [10, 10, 50, 50], "score": 0.88}]
    ev_box = create_bounding_box_evidence(boxes=boxes, title="Grounding Detections")
    assert ev_box.evidence_type == EvidenceType.BOUNDING_BOX
    assert len(ev_box.data["boxes"]) == 1

    # 5. Change map
    ev_change = create_change_map_evidence(title="Change Map", data={"change_ratio": 0.052})
    assert ev_change.evidence_type == EvidenceType.CHANGE_MAP

    # 6. Statistics
    ev_stats = create_statistics_evidence({"mean": 128.5, "std": 34.2})
    assert ev_stats.evidence_type == EvidenceType.STATISTICS

    # 7. Metadata
    ev_meta = create_metadata_evidence(meta, title="Extracted Tags")
    assert ev_meta.evidence_type == EvidenceType.METADATA
    assert ev_meta.metadata["width"] == 64
