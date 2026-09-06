"""Regression tests for SatQuery AI UI rendering and ValidationStatus integrity."""

from pathlib import Path
import html
import markdown_it
import pytest
from satquery.agent.controller import AgentController
from satquery.domain.schemas import (
    InputMode,
    SlotAssignment,
    TaskType,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from satquery.ui.components import (
    render_file_badge,
    render_header,
    render_hero,
    render_pipeline_flow,
    render_telemetry_hud,
    render_top5_bars,
)


def test_validation_status_enum_members():
    """Verify ValidationStatus only uses PASS, WARN, FAIL and does not define or expect VALID."""
    assert hasattr(ValidationStatus, "PASS")
    assert hasattr(ValidationStatus, "WARN")
    assert hasattr(ValidationStatus, "FAIL")
    assert not hasattr(ValidationStatus, "VALID")
    assert ValidationStatus.PASS == "pass"
    assert ValidationStatus.WARN == "warn"
    assert ValidationStatus.FAIL == "fail"


def test_validation_report_rendering_logic():
    """Verify UI validation check logic handles PASS, WARN, and FAIL without AttributeError."""
    # PASS report
    pass_report = ValidationReport(
        status=ValidationStatus.PASS,
        checks=[
            ValidationCheck(code="C1", status=ValidationStatus.PASS, message="OK"),
        ],
    )
    assert pass_report.status == ValidationStatus.PASS
    assert not hasattr(pass_report, "errors")

    # Simulate app.py line 808 logic
    if pass_report.status == ValidationStatus.PASS:
        msg = "All ingested rasters passed validation."
    else:
        msg = "Validation failed."
    assert "passed" in msg

    # WARN report with non-pass checks
    warn_report = ValidationReport(
        status=ValidationStatus.WARN,
        checks=[
            ValidationCheck(code="W1", status=ValidationStatus.WARN, message="Slight spatial discrepancy"),
        ],
    )
    failing_msgs = [c.message for c in warn_report.checks if c.status != ValidationStatus.PASS]
    assert len(failing_msgs) == 1
    assert failing_msgs[0] == "Slight spatial discrepancy"


def test_metric_card_rendering_no_indented_code_blocks():
    """Verify render_telemetry_hud produces clean HTML with zero lines starting with 4+ spaces."""
    metrics = [
        {"label": "Changed Area", "value": "14.20 ha", "hint": "142,000 m²"},
        {"label": "Change Ratio", "value": "4.27%", "hint": "Ratio of changed pixels"},
        {"label": "Change Regions", "value": "18", "hint": "Polygonized clusters"},
        {"label": "Confidence", "value": "92.3%", "hint": "bit_softmax_mean_confidence"},
        {"label": "Processing Time", "value": "0.85s", "hint": "Tiled inference + vectorization"},
    ]

    hud_html = render_telemetry_hud(metrics)

    # 1. No line should start with 4 or more spaces (indented code block in Markdown)
    for line in hud_html.splitlines():
        assert not line.startswith("    "), f"Indented line found: {line}"
        assert not line.startswith("\t"), f"Tab-indented line found: {line}"

    # 2. Key structural tags must be present
    assert '<div class="sat-metrics-grid">' in hud_html
    assert '<div class="sat-metric-box">' in hud_html
    assert '<div class="sat-metric-label">Changed Area</div>' in hud_html
    assert '<div class="sat-metric-value">14.20 ha</div>' in hud_html
    assert '<div class="sat-metric-hint">142,000 m²</div>' in hud_html

    # 3. No markdown escape artifacts
    assert "&#x20;" not in hud_html
    assert r"\<div" not in hud_html
    assert r"\</div>" not in hud_html

    # 4. Markdown parser must parse it as raw HTML, NOT as a code block (<pre><code>)
    md = markdown_it.MarkdownIt()
    tokens = md.parse(hud_html)
    token_types = [t.type for t in tokens]
    assert "code_block" not in token_types
    assert "fence" not in token_types

    rendered = md.render(hud_html)
    assert "<pre><code>" not in rendered
    assert '<div class="sat-metrics-grid">' in rendered


def test_top5_bars_no_indented_code_blocks():
    """Verify render_top5_bars produces clean HTML without indented code block triggers."""
    top5 = [
        {"class": "Continuous urban fabric", "probability": 0.852},
        {"class": "Industrial or commercial units", "probability": 0.124},
    ]
    bars_html = render_top5_bars(top5)
    for line in bars_html.splitlines():
        assert not line.startswith("    "), f"Indented line: {line}"

    md = markdown_it.MarkdownIt()
    tokens = md.parse(bars_html)
    token_types = [t.type for t in tokens]
    assert "code_block" not in token_types
    rendered = md.render(bars_html)
    assert "<pre><code>" not in rendered
    assert '<div class="pred-list">' in rendered


def test_header_hero_and_pipeline_no_indented_code_blocks():
    """Verify header, hero, and pipeline components produce clean unindented HTML."""
    md = markdown_it.MarkdownIt()

    for html_str in [
        render_header(),
        render_hero(),
        render_pipeline_flow(
            query="Compare satellite images",
            task="BI_TEMPORAL_CHANGE",
            model_name="Open-CD BIT ResNet-18",
            model_id="opencd_bit_change",
            tools=["raster_inspection", "change_map"],
        ),
    ]:
        for line in html_str.splitlines():
            assert not line.startswith("    "), f"Indented line: {line}"
        tokens = md.parse(html_str)
        token_types = [t.type for t in tokens]
        assert "code_block" not in token_types
        rendered = md.render(html_str)
        assert "<pre><code>" not in rendered


def test_bitemporal_change_detection_result_integrity(optical_geotiff: Path):
    """Verify change detection result has valid validation report and reaches UI metrics safely."""
    controller = AgentController()
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    query = "Compare these two satellite images and map significant changes."
    result = controller.analyze(query, slots, InputMode.I4_BITEMPORAL_PAIR)

    assert result.plan.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert result.validation is not None
    assert result.validation.status == ValidationStatus.PASS
    assert hasattr(result.validation, "checks")

    stats_ev = next((e for e in result.evidence if e.title == "Bi-Temporal Change Statistics"), None)
    assert stats_ev is not None
    assert stats_ev.data is not None

    s_data = stats_ev.data
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
            "label": "Change Regions",
            "value": f"{s_data.get('regions_count', 0):,}",
            "hint": "Polygonized clusters",
        },
        {
            "label": "Confidence",
            "value": f"{result.confidence.score*100:.1f}%" if result.confidence.score is not None else "N/A",
            "hint": result.confidence.method or "Calibrated",
        },
        {
            "label": "Processing Time",
            "value": f"{s_data.get('timings_seconds', {}).get('total', 0.0):.2f}s",
            "hint": "Tiled inference + vectorization",
        },
    ]
    hud_html = render_telemetry_hud(hud_metrics)
    assert '<div class="sat-metrics-grid">' in hud_html
    assert "Change Ratio" in hud_html
    assert "&#x20;" not in hud_html
