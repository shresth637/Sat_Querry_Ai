import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from satquery.agent.controller import AgentController
from satquery.domain.schemas import InputMode, SlotAssignment
from satquery.models.benchmarks import run_bit_benchmark
from satquery.models.geospatial import calculate_raster_area, validate_bitemporal_rasters
from satquery.models.opencd_bit import OpenCDBITModel, get_device_info
from satquery.models.tiling import (
    create_tapered_window,
    generate_tile_windows,
    TiledInferenceEngine,
)


def create_test_geotiff(
    file_path: Path,
    width: int,
    height: int,
    crs: str = "EPSG:32633",
    res: tuple[float, float] = (10.0, 10.0),
    origin: tuple[float, float] = (500000.0, 4649760.0),
    seed: int = 42,
    dtype: str = "uint8",
) -> None:
    """Helper to create synthetic GeoTIFF for testing."""
    np.random.seed(seed)
    data = np.random.randint(20, 220, size=(3, height, width), dtype=np.uint8)
    transform = from_origin(origin[0], origin[1], res[0], res[1])

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture(scope="module")
def bit_model():
    """Load model once for tests."""
    return OpenCDBITModel()


def test_1_cuda_cpu_device_selection():
    """Test device selection logic and reporting."""
    info_auto = get_device_info("auto")
    assert "device" in info_auto
    assert info_auto["device"] in ["cuda", "cpu"]
    assert "cuda_available" in info_auto
    assert "pytorch_version" in info_auto

    info_cpu = get_device_info("cpu")
    assert info_cpu["device"] == "cpu"


def test_2_256x256_inference(bit_model):
    """Test 256x256 inference without downsampling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_256.tif"
        t1 = tmp / "t1_256.tif"
        create_test_geotiff(t0, 256, 256, seed=1)
        create_test_geotiff(t1, 256, 256, seed=2)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        assert res.status == "success"
        stats_art = next(a for a in res.artifacts if a["type"] == "change_statistics")
        assert stats_art["data"]["total_pixels"] == 256 * 256
        assert stats_art["data"]["total_tiles_evaluated"] == 1


def test_3_512x512_tiled_inference(bit_model):
    """Test 512x512 sliding-window inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_512.tif"
        t1 = tmp / "t1_512.tif"
        create_test_geotiff(t0, 512, 512, seed=3)
        create_test_geotiff(t1, 512, 512, seed=4)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        assert res.status == "success"
        stats_art = next(a for a in res.artifacts if a["type"] == "change_statistics")
        assert stats_art["data"]["total_pixels"] == 512 * 512
        # With 256 tile size and 25% overlap (step=192), 512 needs 3x3 = 9 tiles
        assert stats_art["data"]["total_tiles_evaluated"] == 9


def test_4_overlapping_tiles_reconstruction():
    """Verify tapered window and tile generation with overlap."""
    window = create_tapered_window(256, taper_ratio=0.25)
    assert window.shape == (256, 256)
    assert window.min() > 0.0
    assert window.max() <= 1.0
    # Center should have maximum weight
    assert np.isclose(window[128, 128], 1.0)
    # Corners should have low weight
    assert window[0, 0] < 0.1

    windows = generate_tile_windows(512, 512, tile_size=256, overlap_ratio=0.25)
    assert len(windows) == 9
    for y, x, th, tw in windows:
        assert th <= 256 and tw <= 256
        assert y + th <= 512 and x + tw <= 512


def test_5_edge_tiles_odd_dimensions(bit_model):
    """Verify tiled inference handles non-standard dimensions covering edges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_300x450.tif"
        t1 = tmp / "t1_300x450.tif"
        create_test_geotiff(t0, 450, 300, seed=5)
        create_test_geotiff(t1, 450, 300, seed=6)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        assert res.status == "success"
        mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")
        with rasterio.open(mask_art["path"]) as src:
            assert src.width == 450
            assert src.height == 300


def test_6_probability_map_dimensions_and_values(bit_model):
    """Verify float32 probability map GeoTIFF dimensions and range [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_384.tif"
        t1 = tmp / "t1_384.tif"
        create_test_geotiff(t0, 384, 384, seed=7)
        create_test_geotiff(t1, 384, 384, seed=8)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        assert res.status == "success"

        prob_art = next(a for a in res.artifacts if a["type"] == "change_prob_geotiff")
        with rasterio.open(prob_art["path"]) as src:
            assert src.width == 384
            assert src.height == 384
            assert src.dtypes[0] == "float32"
            prob_data = src.read(1)
            assert prob_data.min() >= 0.0
            assert prob_data.max() <= 1.0


def test_7_binary_mask_dimensions_and_values(bit_model):
    """Verify uint8 binary change mask GeoTIFF dimensions and discrete values {0, 1}."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_256.tif"
        t1 = tmp / "t1_256.tif"
        create_test_geotiff(t0, 256, 256, seed=9)
        create_test_geotiff(t1, 256, 256, seed=10)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")
        with rasterio.open(mask_art["path"]) as src:
            assert src.width == 256
            assert src.height == 256
            assert src.dtypes[0] == "uint8"
            mask_data = src.read(1)
            unique_vals = set(np.unique(mask_data))
            assert unique_vals.issubset({0, 1})


def test_8_crs_preservation(bit_model):
    """Verify input CRS is preserved in both output GeoTIFFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_utm.tif"
        t1 = tmp / "t1_utm.tif"
        target_crs = "EPSG:32632"
        create_test_geotiff(t0, 256, 256, crs=target_crs, seed=11)
        create_test_geotiff(t1, 256, 256, crs=target_crs, seed=12)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")
        prob_art = next(a for a in res.artifacts if a["type"] == "change_prob_geotiff")

        with rasterio.open(mask_art["path"]) as src:
            assert src.crs.to_string().upper() == target_crs.upper()
        with rasterio.open(prob_art["path"]) as src:
            assert src.crs.to_string().upper() == target_crs.upper()


def test_9_transform_preservation(bit_model):
    """Verify affine spatial transform is strictly preserved in output GeoTIFFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 256, 256, origin=(600000.0, 5000000.0), res=(5.0, 5.0), seed=13)
        create_test_geotiff(t1, 256, 256, origin=(600000.0, 5000000.0), res=(5.0, 5.0), seed=14)

        with rasterio.open(t0) as src_in:
            orig_transform = src_in.transform

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        mask_art = next(a for a in res.artifacts if a["type"] == "change_mask_geotiff")

        with rasterio.open(mask_art["path"]) as src_out:
            assert src_out.transform == orig_transform


def test_10_threshold_changes_output(bit_model):
    """Verify changing threshold dynamically alters the resulting change mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 256, 256, seed=15)
        create_test_geotiff(t1, 256, 256, seed=16)

        # Low threshold -> more change detected
        res_low = bit_model.predict({"t0": str(t0), "t1": str(t1)}, change_threshold=0.2)
        stats_low = next(a for a in res_low.artifacts if a["type"] == "change_statistics")["data"]

        # High threshold -> less change detected
        res_high = bit_model.predict({"t0": str(t0), "t1": str(t1)}, change_threshold=0.8)
        stats_high = next(a for a in res_high.artifacts if a["type"] == "change_statistics")["data"]

        assert stats_low["changed_pixels"] >= stats_high["changed_pixels"]
        assert stats_low["change_threshold"] == 0.2
        assert stats_high["change_threshold"] == 0.8


def test_11_invalid_crs_validation():
    """Verify mismatched CRS raises validation error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 256, 256, crs="EPSG:4326")
        create_test_geotiff(t1, 256, 256, crs="EPSG:32633")

        with pytest.raises(ValueError, match="CRS mismatch"):
            validate_bitemporal_rasters(t0, t1)


def test_12_mismatched_raster_dimensions_validation():
    """Verify mismatched image dimensions raises validation error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 256, 256)
        create_test_geotiff(t1, 512, 512)

        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_bitemporal_rasters(t0, t1)


def test_13_missing_input(bit_model):
    """Verify missing inputs returns error ModelResult."""
    res = bit_model.predict({"t0": "nonexistent_path_0.tif", "t1": "nonexistent_path_1.tif"})
    assert res.status == "error"
    assert "validation failed" in res.error.lower() or "not found" in res.error.lower()


def test_14_projected_crs_area_calculation():
    """Verify metric area calculation for projected UTM CRS."""
    meta = {
        "crs": "EPSG:32633",
        "res": (10.0, 10.0),
        "bounds": None,
    }
    # 100 pixels * (10m * 10m = 100 m2/pixel) = 10,000 m2 = 1.0 hectare
    area = calculate_raster_area(meta, changed_pixels=100)
    assert area["changed_area_m2"] == 10000.0
    assert area["changed_area_hectares"] == 1.0
    assert area["changed_area_km2"] == 0.01
    assert area["area_calculation_method"] == "projected_planar_metric"


def test_15_geographic_crs_handling():
    """Verify geographic EPSG:4326 area calculation uses WGS84 ellipsoidal scaling instead of deg^2."""
    meta = {
        "crs": "EPSG:4326",
        "res": (0.0001, 0.0001),  # ~11.1 meters at equator
        "bounds": type("Bounds", (), {"bottom": -0.01, "top": 0.01})(),
    }
    area = calculate_raster_area(meta, changed_pixels=100)
    assert area["area_calculation_method"] == "geographic_ellipsoidal_wgs84"
    assert area["changed_area_m2"] is not None
    # 0.0001 deg at equator is ~11.13m, so 1 pixel is ~123 m2. 100 pixels is ~12,300 m2
    assert 10000.0 <= area["changed_area_m2"] <= 15000.0
    assert area["changed_area_hectares"] > 1.0


def test_16_statistics_generation_and_json(bit_model):
    """Verify JSON statistics artifact is created and contains all required metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 256, 256, seed=17)
        create_test_geotiff(t1, 256, 256, seed=18)

        res = bit_model.predict({"t0": str(t0), "t1": str(t1)})
        stats_art = next(a for a in res.artifacts if a["type"] == "change_statistics")
        json_path = Path(stats_art["path"])
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "total_pixels" in data
        assert "changed_pixels" in data
        assert "percentage_changed" in data
        assert "mean_prediction_confidence" in data
        assert "maximum_confidence" in data
        assert "changed_pixel_confidence" in data
        assert "unchanged_pixel_confidence" in data
        assert "low_confidence_pixel_percentage" in data
        assert "area_calculation_method" in data
        assert "timings_seconds" in data
        assert "total_tiles_evaluated" in data


def test_17_controller_integration():
    """Verify AgentController executes production tiled inference and returns enhanced result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_test_geotiff(t0, 512, 512, seed=19)
        create_test_geotiff(t1, 512, 512, seed=20)

        controller = AgentController()
        slots = [
            SlotAssignment(slot_id="t0", file_path=str(t0)),
            SlotAssignment(slot_id="t1", file_path=str(t1)),
        ]

        res = controller.analyze(
            query="Detect changes between these two satellite dates.",
            slots=slots,
            input_mode=InputMode.I4_BITEMPORAL_PAIR,
            change_threshold=0.55,
        )

        assert res.plan.task == "BI_TEMPORAL_CHANGE"
        assert res.plan.selected_models == ["opencd_bit_change"]
        assert "Bi-temporal change detection completed via Open-CD BIT" in res.result_text
        assert res.confidence.method == "bit_softmax_mean_confidence"

        # Check evidence includes both GeoTIFFs, visualization, and stats
        titles = [e.title for e in res.evidence]
        assert "Binary Change Mask (GeoTIFF)" in titles
        assert "Change Probability Map (GeoTIFF)" in titles
        assert "Change Map Visual Overlay" in titles
        assert "Bi-Temporal Change Statistics" in titles


def test_18_benchmark_utility(bit_model):
    """Verify performance benchmark executes and measures actual timings."""
    bench = run_bit_benchmark(model=bit_model, sizes=[(256, 256), (512, 512)])
    assert bench["device"] in ["cpu", "cuda"]
    assert len(bench["benchmarks"]) == 2

    b256 = bench["benchmarks"][0]
    assert b256["dimensions"] == "256x256"
    assert b256["tiles_evaluated"] == 1
    assert b256["total_time_s"] > 0.0

    b512 = bench["benchmarks"][1]
    assert b512["dimensions"] == "512x512"
    assert b512["tiles_evaluated"] == 9
    assert b512["total_time_s"] > 0.0
