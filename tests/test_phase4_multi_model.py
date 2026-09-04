import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
import torch

from satquery.agent.controller import AgentController
from satquery.agent.router import QueryIntentParser, route_query
from satquery.domain.schemas import (
    EvidenceType,
    InputMode,
    ModelCapability,
    SlotAssignment,
    TaskType,
)
from satquery.models.base import ModelAdapter
from satquery.models.bigearthnet import BIGEARTHNET_19_CLASSES, BigEarthNetClassifier, BigEarthNetModel
from satquery.models.geochat import GeoChatVLMModel
from satquery.models.grounding import (
    export_grounding_geojson,
    normalize_box_to_pixels,
    pixel_box_to_geo_polygon,
    render_grounding_overlay,
)
from satquery.models.manager import ResourceManager, get_resource_manager
from satquery.preprocess.raster import inspect_raster


def test_1_model_capability_enum():
    """Verify ModelCapability enum and string values."""
    assert ModelCapability.CHANGE_DETECTION.value == "change_detection"
    assert ModelCapability.IMAGE_VQA.value == "image_vqa"
    assert ModelCapability.IMAGE_CAPTION.value == "image_caption"
    assert ModelCapability.GROUNDING.value == "grounding"
    assert ModelCapability.CLASSIFICATION.value == "classification"
    assert ModelCapability.OPTICAL_SAR_FUSION.value == "optical_sar_fusion"


def test_2_resource_manager_device_selection():
    """Verify ResourceManager handles auto/cpu/cuda selection safely without crashing."""
    rm = ResourceManager()
    device_cpu = rm.get_device("cpu")
    assert device_cpu.type == "cpu"

    device_auto = rm.get_device("auto")
    assert device_auto.type in ["cpu", "cuda"]

    vram = rm.get_vram_info()
    assert isinstance(vram, dict)
    assert "cuda_available" in vram


def test_3_resource_manager_memory_cleanup():
    """Verify memory cleanup and garbage collection run without error."""
    rm = ResourceManager()
    rm.cleanup_memory()


def test_4_resource_manager_large_model_exclusivity():
    """Verify ResourceManager only keeps one large model resident at a time."""
    rm = ResourceManager()

    class DummyLargeModel(ModelAdapter):
        def __init__(self, name: str):
            super().__init__(name=name, version="7B", capabilities=["vqa"], status="unloaded")
            self.loaded = False

        def load(self):
            self.loaded = True
            self.status = "ready"

        def unload(self):
            self.loaded = False
            self.status = "unloaded"

        def predict(self, inputs, **kwargs):
            from satquery.domain.schemas import ModelResult
            return ModelResult(registry_id=self.name, capability="vqa", status="success")

    m1 = DummyLargeModel("m1")
    m2 = DummyLargeModel("m2")

    rm.acquire_model("m1", m1, is_large=True)
    assert m1.status == "ready"
    assert rm._resident_large_model_id == "m1"

    # Acquiring m2 should unload m1
    rm.acquire_model("m2", m2, is_large=True)
    assert m2.status == "ready"
    assert m1.status == "unloaded"
    assert rm._resident_large_model_id == "m2"

    rm.release_all()
    assert rm._resident_large_model_id is None


def test_5_bigearthnet_adapter_initialization():
    """Verify 19-class BigEarthNet metadata and class listing."""
    assert len(BIGEARTHNET_19_CLASSES) == 19
    assert "Urban fabric" in BIGEARTHNET_19_CLASSES
    assert "Broad-leaved forest" in BIGEARTHNET_19_CLASSES

    adapter = BigEarthNetModel(weights_path=None)
    assert adapter.status == "not_configured"
    assert "classification" in adapter.capabilities


def test_6_bigearthnet_preprocessing(optical_geotiff: Path):
    """Verify BigEarthNet preprocessing transforms input raster to normalized [1, 3, 224, 224] tensor."""
    adapter = BigEarthNetModel(weights_path=None)
    tensor, meta = adapter.preprocess(str(optical_geotiff))
    assert tensor.shape == (1, 3, 224, 224)
    assert meta["width"] > 0
    assert meta["height"] > 0


def test_7_bigearthnet_prediction_with_mock_checkpoint(optical_geotiff: Path, tmp_path: Path):
    """Create a temporary state dict to verify BigEarthNet inference, chart generation, and stats."""
    model_weights_path = tmp_path / "mock_bigearthnet.pth"
    classifier = BigEarthNetClassifier(num_classes=19)
    torch.save(classifier.state_dict(), str(model_weights_path))

    adapter = BigEarthNetModel(
        weights_path=str(model_weights_path),
        threshold=0.20,
        output_dir=str(tmp_path / "out"),
    )
    assert adapter.status == "unloaded"
    adapter.load()
    assert adapter.status == "ready"

    result = adapter.predict({"image": str(optical_geotiff)})
    assert result.status == "success"
    assert result.capability == "classification"
    assert result.text is not None
    assert "EXECUTIVE SUMMARY" in result.text

    # Verify artifacts
    assert len(result.artifacts) == 2
    chart_art = next((a for a in result.artifacts if a["type"] == "land_cover_chart"), None)
    assert chart_art is not None
    assert Path(chart_art["path"]).exists()

    stats_art = next((a for a in result.artifacts if a["type"] == "land_cover_statistics"), None)
    assert stats_art is not None
    assert Path(stats_art["path"]).exists()
    assert "all_probabilities" in stats_art["data"]


def test_8_bigearthnet_not_configured_fallback(optical_geotiff: Path):
    """Verify unconfigured BigEarthNet reports status='not_configured' cleanly."""
    adapter = BigEarthNetModel(weights_path="non_existent_weights.pth")
    assert adapter.status == "not_configured"
    res = adapter.predict({"image": str(optical_geotiff)})
    assert res.status == "not_configured"
    assert "not configured" in res.text


def test_9_geochat_adapter_not_configured_fallback(optical_geotiff: Path):
    """Verify unconfigured GeoChat reports status='not_configured' and text=None for registry compatibility."""
    adapter = GeoChatVLMModel(weights_path=None, capability="image_vqa")
    assert adapter.status == "not_configured"
    res = adapter.predict({"image": str(optical_geotiff)})
    assert res.status == "not_configured"
    assert res.text is None


def test_10_geochat_grounding_coordinate_parsing():
    """Verify normalized coordinate parsing to pixel and CRS coordinates."""
    # Test [ymin, xmin, ymax, xmax] in [0, 1000] scale
    norm_box = [100.0, 200.0, 500.0, 800.0]
    px_box = normalize_box_to_pixels(norm_box, img_width=1000, img_height=500, source_scale=1000.0)
    assert px_box == (200, 50, 800, 250)

    transform = from_bounds(0, 0, 1000, 500, 1000, 500)
    rings, geo_bbox = pixel_box_to_geo_polygon(px_box, transform)
    assert len(rings[0]) == 5
    assert len(geo_bbox) == 4


def test_11_grounding_geojson_generation(tmp_path: Path):
    """Verify RFC 7946 GeoJSON generation for spatial grounding boxes."""
    boxes = [
        {
            "label": "water_body",
            "confidence": 0.95,
            "pixel_box": [50, 50, 200, 200],
            "geo_bbox": [10.0, 10.0, 20.0, 20.0],
        }
    ]
    out_file = tmp_path / "grounding.geojson"
    geojson = export_grounding_geojson(boxes, out_file, crs="EPSG:4326")

    assert out_file.exists()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feat = geojson["features"][0]
    assert feat["properties"]["label"] == "water_body"
    assert feat["geometry"]["type"] == "Polygon"


def test_12_grounding_overlay_rendering(optical_geotiff: Path, tmp_path: Path):
    """Verify grounding visual overlay PNG generation."""
    boxes = [
        {
            "label": "Building Cluster",
            "confidence": 0.88,
            "pixel_box": [20, 20, 100, 100],
        }
    ]
    out_png = tmp_path / "grounding_overlay.png"
    render_grounding_overlay(optical_geotiff, boxes, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_13_semantic_router_land_cover_queries(optical_geotiff: Path):
    """Verify queries about land cover classification route to CLASSIFICATION and select BigEarthNet."""
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    plan1 = route_query("Classify the land-cover categories in this scene", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan1.task == TaskType.CLASSIFICATION.value
    assert "bigearthnet_resnet50_s1s2" in plan1.selected_models

    plan2 = route_query("What land-use patterns are visible?", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan2.task == TaskType.CLASSIFICATION.value
    assert "bigearthnet_resnet50_s1s2" in plan2.selected_models


def test_14_semantic_router_single_image_grounding(optical_geotiff: Path):
    """Verify grounding queries route to SINGLE_GROUNDING."""
    meta = inspect_raster(optical_geotiff)
    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    metas = {"image": meta}

    plan = route_query("Locate and pinpoint the water body", slots, metas, InputMode.I1_SINGLE_OPTICAL)
    assert plan.task == TaskType.SINGLE_GROUNDING.value
    assert "geochat_grounding" in plan.selected_models


def test_15_controller_multi_model_evidence_integration(optical_geotiff: Path, tmp_path: Path):
    """Verify AgentController executes BigEarthNet and integrates chart/stats evidence."""
    # Create mock BigEarthNet weights
    weights_path = tmp_path / "mock_ben.pth"
    classifier = BigEarthNetClassifier(num_classes=19)
    torch.save(classifier.state_dict(), str(weights_path))

    controller = AgentController()
    ben_adapter = BigEarthNetModel(
        weights_path=str(weights_path),
        threshold=0.20,
        output_dir=str(tmp_path / "ben_runs"),
    )
    controller.model_registry.register("bigearthnet_resnet50_s1s2", ben_adapter, enabled=True)

    slots = [SlotAssignment(slot_id="image", file_path=str(optical_geotiff))]
    result = controller.analyze(
        query="Classify the land-cover in this satellite scene",
        slots=slots,
        input_mode=InputMode.I1_SINGLE_OPTICAL,
    )

    assert result.plan.task == TaskType.CLASSIFICATION.value
    assert "Scene land-cover classification completed" in result.result_text
    assert result.confidence.is_available is True
    assert result.confidence.method == "bigearthnet_sigmoid_max_confidence"

    # Verify chart evidence
    chart_ev = next((e for e in result.evidence if "Land-Cover" in e.title), None)
    assert chart_ev is not None
    assert Path(chart_ev.file_path).exists()
