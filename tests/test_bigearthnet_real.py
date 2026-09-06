from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from satquery.agent.controller import AgentController
from satquery.domain.schemas import (
    InputMode,
    SlotAssignment,
    TaskType,
    ValidationStatus,
)
from satquery.models.bigearthnet import BIGEARTHNET_19_CLASSES, BigEarthNetModel


REAL_CHECKPOINT_PATH = Path("models/checkpoints/resnet50_s2_v0.2.0.pth")
SENTINEL2_SAMPLE_PATH = Path("samples/sentinel2_small.tif")


@pytest.fixture
def dummy_rgb_png(tmp_path: Path) -> Path:
    """Create a standard 3-channel RGB PNG file."""
    p = tmp_path / "test_rgb.png"
    img = Image.new("RGB", (128, 128), color=(73, 109, 137))
    img.save(p)
    return p


@pytest.fixture
def single_band_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic 1-band GeoTIFF."""
    p = tmp_path / "single_band.tif"
    data = np.ones((1, 64, 64), dtype=np.uint16) * 500
    transform = from_origin(10.0, 50.0, 0.001, 0.001)
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return p


def test_real_checkpoint_exists():
    """Verify authentic BigEarthNet v2 ResNet-50 checkpoint is on disk."""
    assert REAL_CHECKPOINT_PATH.exists(), f"Missing checkpoint at {REAL_CHECKPOINT_PATH}"
    assert REAL_CHECKPOINT_PATH.stat().st_size > 80 * 1024 * 1024, "Checkpoint file size is suspiciously small."


def test_real_model_loading():
    """Verify BigEarthNetModel successfully loads pretrained weights on device."""
    adapter = BigEarthNetModel(weights_path=str(REAL_CHECKPOINT_PATH), threshold=0.30)
    assert adapter.status == "unloaded"
    adapter.load()
    assert adapter.status == "ready"
    assert adapter.model is not None
    assert adapter.model.in_channels == 10
    assert len(BIGEARTHNET_19_CLASSES) == 19


def test_real_multispectral_prediction():
    """Verify end-to-end inference on authentic Sentinel-2 GeoTIFF."""
    assert SENTINEL2_SAMPLE_PATH.exists(), f"Sample Sentinel-2 tile missing at {SENTINEL2_SAMPLE_PATH}"

    adapter = BigEarthNetModel(weights_path=str(REAL_CHECKPOINT_PATH), threshold=0.20)
    adapter.load()
    assert adapter.status == "ready"

    result = adapter.predict({"image": str(SENTINEL2_SAMPLE_PATH)})
    assert result.status == "success"
    assert result.capability == "classification"
    assert result.text is not None
    assert "EXECUTIVE SUMMARY" in result.text

    # Raw scores validation
    assert result.raw_scores is not None
    assert "confidence_score" in result.raw_scores
    conf = result.raw_scores["confidence_score"]
    assert 0.0 <= conf <= 1.0

    probs = result.raw_scores["probabilities"]
    assert len(probs) == 19
    for cls_name, prob in probs.items():
        assert cls_name in BIGEARTHNET_19_CLASSES
        assert 0.0 <= prob <= 1.0

    # Verify artifacts
    assert len(result.artifacts) == 2
    chart_art = next((a for a in result.artifacts if a["type"] == "land_cover_chart"), None)
    assert chart_art is not None
    assert Path(chart_art["path"]).exists()

    stats_art = next((a for a in result.artifacts if a["type"] == "land_cover_statistics"), None)
    assert stats_art is not None
    assert Path(stats_art["path"]).exists()
    assert stats_art["data"]["detected_classes_count"] >= 0


def test_rejection_of_arbitrary_rgb_png(dummy_rgb_png: Path):
    """Verify model strictly rejects arbitrary RGB PNG inputs with explicit error."""
    adapter = BigEarthNetModel(weights_path=str(REAL_CHECKPOINT_PATH))
    adapter.load()

    result = adapter.predict({"image": str(dummy_rgb_png)})
    assert result.status == "error"
    assert "Arbitrary RGB PNG/JPEG or 3-band images are not valid multispectral inputs" in result.error


def test_rejection_of_single_band_raster(single_band_geotiff: Path):
    """Verify model strictly rejects single-band raster inputs."""
    adapter = BigEarthNetModel(weights_path=str(REAL_CHECKPOINT_PATH))
    adapter.load()

    result = adapter.predict({"image": str(single_band_geotiff)})
    assert result.status == "error"
    assert "Single-band rasters cannot be classified" in result.error


def test_controller_end_to_end_user_query():
    """Verify AgentController executes complete user query with Sentinel-2 GeoTIFF."""
    assert SENTINEL2_SAMPLE_PATH.exists()

    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(SENTINEL2_SAMPLE_PATH))]

    result = controller.analyze(
        query="Classify the land cover in this satellite image and give me the confidence scores.",
        slots=slots,
        input_mode=InputMode.I1_SINGLE_OPTICAL,
    )

    assert result.plan.task == TaskType.CLASSIFICATION.value
    assert result.plan.blocked is False
    assert result.validation.status == ValidationStatus.PASS
    assert "Scene land-cover classification completed via BigEarthNet v2 ResNet-50" in result.result_text
    assert result.confidence.is_available is True
    assert result.confidence.source == "model"
    assert result.confidence.method == "bigearthnet_sigmoid_max_confidence"

    # Evidence items
    assert len(result.evidence) > 0
    chart_ev = next((e for e in result.evidence if "Land-Cover" in e.title), None)
    assert chart_ev is not None
    assert Path(chart_ev.file_path).exists()


def test_runtime_threshold_propagation_and_top5():
    """Verify runtime threshold overrides default, Top 5 is always 5 and sorted, and filtering is decoupled."""
    assert SENTINEL2_SAMPLE_PATH.exists()

    controller = AgentController()
    slots = [SlotAssignment(slot_id="image", file_path=str(SENTINEL2_SAMPLE_PATH))]

    # Test with threshold = 0.10
    result_010 = controller.analyze(
        query="Classify the land cover in this satellite image and give me the confidence scores.",
        slots=slots,
        input_mode=InputMode.I1_SINGLE_OPTICAL,
        threshold=0.10,
    )

    assert result_010.plan.task == TaskType.CLASSIFICATION.value
    # Report displays the runtime threshold and top 5 section
    assert "0.10" in result_010.result_text
    assert "TOP 5 PREDICTED LAND-COVER CATEGORIES" in result_010.result_text

    stats_ev_010 = next((e for e in result_010.evidence if e.title == "Land-Cover Classification Statistics"), None)
    assert stats_ev_010 is not None
    stats_data_010 = stats_ev_010.data
    assert stats_data_010["threshold"] == 0.10

    # Top-5 always contains exactly 5 classes when 19 outputs exist
    top5_010 = stats_data_010["top_5_predictions"]
    assert len(top5_010) == 5

    # Top-5 is sorted descending
    for i in range(len(top5_010) - 1):
        assert top5_010[i]["probability"] >= top5_010[i + 1]["probability"]

    # Multiple classes exceed 0.10
    assert stats_data_010["detected_classes_count"] >= 2
    for c in stats_data_010["detected_classes"]:
        assert c["probability"] >= 0.10

    # Test with high threshold = 0.50 (where 0 classes exceed it)
    result_050 = controller.analyze(
        query="Classify the land cover in this satellite image and give me the confidence scores.",
        slots=slots,
        input_mode=InputMode.I1_SINGLE_OPTICAL,
        threshold=0.50,
    )
    assert "0.50" in result_050.result_text
    assert "TOP 5 PREDICTED LAND-COVER CATEGORIES" in result_050.result_text
    stats_ev_050 = next((e for e in result_050.evidence if e.title == "Land-Cover Classification Statistics"), None)
    assert stats_ev_050 is not None
    assert stats_ev_050.data["threshold"] == 0.50

    # Threshold filtering is independent from top-5
    # Even though 0 classes exceed 0.50, top-5 still contains exactly 5 classes!
    assert stats_ev_050.data["detected_classes_count"] == 0
    assert len(stats_ev_050.data["top_5_predictions"]) == 5
    assert [c["class"] for c in stats_ev_050.data["top_5_predictions"]] == [c["class"] for c in top5_010]
