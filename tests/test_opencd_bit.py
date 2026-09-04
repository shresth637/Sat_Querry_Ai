from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.agent.controller import AgentController
from satquery.domain.schemas import InputMode, SlotAssignment, TaskType
from satquery.models.opencd_bit import OpenCDBITModel


def test_1_opencd_adapter_initialization():
    adapter = OpenCDBITModel()
    assert adapter.name == "Open-CD BIT ResNet-18"
    assert adapter.version == "r18-levir"
    assert "change_detect" in adapter.capabilities
    assert adapter.status == "ready"
    assert adapter.device in ["cpu", "cuda"]


def test_2_missing_checkpoint():
    adapter = OpenCDBITModel(weights_path="models/checkpoints/non_existent.pth")
    assert adapter.status == "not_configured"
    res = adapter.predict({"t0": "a.tif", "t1": "b.tif"})
    assert res.status == "not_configured"
    assert "not ready" in res.error.lower()


def test_3_invalid_input():
    adapter = OpenCDBITModel()
    # Missing t0 and t1
    res = adapter.predict({})
    assert res.status == "error"
    assert "must contain 't0' and 't1'" in res.error


def test_4_incompatible_rasters(corrupt_file: Path):
    adapter = OpenCDBITModel()
    res = adapter.predict({"t0": str(corrupt_file), "t1": str(corrupt_file)})
    assert res.status == "error"
    assert "inference error" in res.error.lower()


def test_5_successful_preprocessing(optical_geotiff: Path):
    adapter = OpenCDBITModel()
    tensor, meta = adapter._preprocess_raster(optical_geotiff, target_size=(256, 256))
    assert tensor.shape == (1, 3, 256, 256)
    assert meta["width"] == 64
    assert meta["height"] == 64
    assert meta["crs"] is not None


def test_6_output_mask_dimensions(optical_geotiff: Path):
    adapter = OpenCDBITModel()
    res = adapter.predict({"t0": str(optical_geotiff), "t1": str(optical_geotiff)})
    assert res.status == "success"
    
    mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")
    mask_path = Path(mask_art["path"])
    assert mask_path.exists()
    
    with rasterio.open(mask_path) as src:
        assert src.width == 64
        assert src.height == 64
        assert src.count == 1


def test_7_crs_preservation(optical_geotiff: Path):
    adapter = OpenCDBITModel()
    res = adapter.predict({"t0": str(optical_geotiff), "t1": str(optical_geotiff)})
    assert res.status == "success"
    
    mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")
    with rasterio.open(mask_art["path"]) as src:
        assert src.crs is not None
        assert "4326" in src.crs.to_string()


def test_8_change_statistics(optical_geotiff: Path):
    adapter = OpenCDBITModel()
    res = adapter.predict({"t0": str(optical_geotiff), "t1": str(optical_geotiff)})
    assert res.status == "success"
    assert res.raw_scores is not None
    assert "changed_pixels" in res.raw_scores
    assert "total_pixels" in res.raw_scores
    assert res.raw_scores["total_pixels"] == 64 * 64
    assert res.raw_scores["percentage_changed"] >= 0.0


def test_9_evidence_generation(optical_geotiff: Path):
    adapter = OpenCDBITModel()
    res = adapter.predict({"t0": str(optical_geotiff), "t1": str(optical_geotiff)})
    assert res.status == "success"
    
    types = [a["type"] for a in res.artifacts]
    assert "change_mask_geotiff" in types
    assert "change_visualization" in types
    assert "change_statistics" in types
    
    vis_art = next(a for a in res.artifacts if a["type"] == "change_visualization")
    assert Path(vis_art["path"]).exists()


def test_10_controller_integration(optical_geotiff: Path):
    controller = AgentController()
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    
    res = controller.analyze("What changed between these two dates?", slots, InputMode.I4_BITEMPORAL_PAIR)
    assert res.plan.task == TaskType.BI_TEMPORAL_CHANGE.value
    assert "Open-CD BIT" in res.result_text
    assert res.confidence.is_available is True
    assert res.confidence.method == "bit_softmax_mean_confidence"
    
    ev_titles = [e.title for e in res.evidence]
    assert "Binary Change Mask (GeoTIFF)" in ev_titles
    assert "Change Map Visual Overlay" in ev_titles
    assert "Bi-Temporal Change Statistics" in ev_titles


def test_11_model_inference_with_simulated_change(tmp_path: Path):
    adapter = OpenCDBITModel()
    
    # Create two synthetic scenes with a distinct change
    t0_path = tmp_path / "scene_t0.tif"
    t1_path = tmp_path / "scene_t1.tif"
    t = from_origin(-122.0, 37.0, 0.001, 0.001)
    
    d0 = np.full((3, 64, 64), 60, dtype=np.uint8)
    d1 = np.full((3, 64, 64), 60, dtype=np.uint8)
    # Add clear changed patch
    d1[:, 15:45, 15:45] = 230
    
    with rasterio.open(t0_path, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8", crs="EPSG:4326", transform=t) as dst:
        dst.write(d0)
    with rasterio.open(t1_path, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8", crs="EPSG:4326", transform=t) as dst:
        dst.write(d1)
        
    res = adapter.predict({"t0": str(t0_path), "t1": str(t1_path)})
    assert res.status == "success"
    assert res.raw_scores["confidence_score"] > 0.5
