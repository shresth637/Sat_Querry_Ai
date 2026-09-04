import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.agent.controller import AgentController
from satquery.agent.router import route_query
from satquery.domain.schemas import InputMode, RasterMeta, SlotAssignment, TaskType
from satquery.models.geospatial import calculate_raster_area
from satquery.models.interpreter import format_change_detection_summary
from satquery.models.opencd_bit import OpenCDBITModel
from satquery.models.regions import extract_change_regions, render_labeled_region_overlay


def create_synthetic_test_raster(
    file_path: Path,
    width: int = 256,
    height: int = 256,
    crs: str = "EPSG:32633",
    seed: int = 42,
) -> None:
    """Create a synthetic 3-band GeoTIFF test raster."""
    np.random.seed(seed)
    data = np.random.randint(30, 210, size=(3, height, width), dtype=np.uint8)
    transform = from_origin(500000.0, 4649760.0, 10.0, 10.0)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# 1. Natural Language Router Tests
# ---------------------------------------------------------------------------

def test_1_nl_router_enhancements():
    """Verify natural-language router correctly handles Phase 3C query variations."""
    t0_meta = RasterMeta(filename="t0.tif", width=256, height=256, band_count=3, crs="EPSG:32633")
    t1_meta = RasterMeta(filename="t1.tif", width=256, height=256, band_count=3, crs="EPSG:32633")
    slots = [
        SlotAssignment(slot_id="t0", file_path="t0.tif"),
        SlotAssignment(slot_id="t1", file_path="t1.tif"),
    ]
    metas = {"t0": t0_meta, "t1": t1_meta}
    input_mode = InputMode.I4_BITEMPORAL_PAIR

    queries_and_expected = [
        ("Where are the changed buildings?", TaskType.BI_TEMPORAL_CHANGE.value),
        ("Show me areas where construction happened", TaskType.BI_TEMPORAL_CHANGE.value),
        ("How much of the area changed?", TaskType.BI_TEMPORAL_CHANGE.value),
        ("Compare these two satellite images", TaskType.BI_TEMPORAL_CHANGE.value),
        ("What changed between these images?", TaskType.BI_TEMPORAL_CHANGE.value),
    ]

    for q, exp_task in queries_and_expected:
        plan = route_query(query=q, slots=slots, metas=metas, input_mode=input_mode)
        assert plan.task == exp_task, f"Failed on query: {q}"
        assert not plan.blocked
        assert "opencd_bit_change" in plan.selected_models

    # Single-image caption query
    single_slots = [SlotAssignment(slot_id="image", file_path="img.tif")]
    single_metas = {"image": t0_meta}
    caption_plan = route_query(
        query="What is visible in this satellite image?",
        slots=single_slots,
        metas=single_metas,
        input_mode=InputMode.I1_SINGLE_OPTICAL,
    )
    assert caption_plan.task == TaskType.SINGLE_CAPTION.value


# ---------------------------------------------------------------------------
# 2. Region Extraction & Filtering Tests
# ---------------------------------------------------------------------------

def test_2_single_region_extraction():
    """Verify extraction of a single clear connected change cluster."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    # 10x10 square = 100 pixels
    mask[20:30, 40:50] = 1
    prob = np.zeros((h, w), dtype=np.float32)
    prob[20:30, 40:50] = 0.85

    meta = {
        "transform": from_origin(500000.0, 4649760.0, 10.0, 10.0),
        "crs": "EPSG:32633",
        "res": (10.0, 10.0),
        "bounds": None,
    }

    regions, geojson, concentration = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert len(regions) == 1
    r = regions[0]
    assert r["pixel_count"] == 100
    assert r["bbox_px"] == [40, 20, 50, 30]
    assert r["rank"] == 1
    assert r["area_m2"] == 10000.0  # 100 * 100m²
    assert np.isclose(r["percentage_of_change"], 100.0)
    assert "Concentrated" in concentration


def test_3_multiple_regions_and_sorting():
    """Verify multiple change regions are sorted descending by area."""
    h, w = 150, 150
    mask = np.zeros((h, w), dtype=np.uint8)
    # Small region: 5x5 = 25 pixels
    mask[10:15, 10:15] = 1
    # Large region: 10x10 = 100 pixels
    mask[60:70, 60:70] = 1
    # Medium region: 7x7 = 49 pixels
    mask[110:117, 110:117] = 1

    prob = np.full((h, w), 0.75, dtype=np.float32)
    meta = {
        "transform": from_origin(500000.0, 4649760.0, 5.0, 5.0),
        "crs": "EPSG:32633",
        "res": (5.0, 5.0),
        "bounds": None,
    }

    regions, geojson, _ = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert len(regions) == 3
    # Largest first
    assert regions[0]["pixel_count"] == 100
    assert regions[0]["rank"] == 1
    assert regions[1]["pixel_count"] == 49
    assert regions[1]["rank"] == 2
    assert regions[2]["pixel_count"] == 25
    assert regions[2]["rank"] == 3


def test_4_small_region_noise_filtering():
    """Verify regions smaller than min_change_region_pixels are filtered."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    # Noise cluster: 3x3 = 9 pixels (below min 20)
    mask[5:8, 5:8] = 1
    # Significant cluster: 6x6 = 36 pixels (above min 20)
    mask[50:56, 50:56] = 1

    prob = np.full((h, w), 0.8, dtype=np.float32)
    meta = {
        "transform": from_origin(500000.0, 4649760.0, 10.0, 10.0),
        "crs": "EPSG:32633",
        "res": (10.0, 10.0),
        "bounds": None,
    }

    regions, geojson, _ = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert len(regions) == 1
    assert regions[0]["pixel_count"] == 36
    assert len(geojson["features"]) == 1


def test_5_empty_change_mask():
    """Verify empty change mask produces 0 regions and valid empty GeoJSON."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.uint8)
    prob = np.zeros((h, w), dtype=np.float32)
    meta = {
        "transform": from_origin(0.0, 0.0, 1.0, 1.0),
        "crs": "EPSG:4326",
        "res": (1.0, 1.0),
        "bounds": None,
    }

    regions, geojson, conc = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert len(regions) == 0
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 0
    assert "No change" in conc


def test_6_full_scene_change_mask():
    """Verify full scene change (100% changed) produces valid region and GeoJSON."""
    h, w = 50, 50
    mask = np.ones((h, w), dtype=np.uint8)
    prob = np.ones((h, w), dtype=np.float32)
    meta = {
        "transform": from_origin(500000.0, 4649760.0, 10.0, 10.0),
        "crs": "EPSG:32633",
        "res": (10.0, 10.0),
        "bounds": None,
    }

    regions, geojson, conc = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert len(regions) == 1
    assert regions[0]["pixel_count"] == 2500
    assert regions[0]["percentage_of_change"] == 100.0
    assert len(geojson["features"]) == 1


# ---------------------------------------------------------------------------
# 3. GeoJSON & Coordinate System Tests
# ---------------------------------------------------------------------------

def test_7_geojson_structure_and_properties():
    """Verify GeoJSON FeatureCollection follows RFC 7946 structure."""
    h, w = 80, 80
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:30, 20:30] = 1  # 100 px
    prob = np.full((h, w), 0.9, dtype=np.float32)
    meta = {
        "transform": from_origin(100.0, 200.0, 2.0, 2.0),
        "crs": "EPSG:32633",
        "res": (2.0, 2.0),
        "bounds": None,
    }

    regions, geojson, _ = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert len(geojson["features"]) == 1

    feature = geojson["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert len(feature["geometry"]["coordinates"]) > 0

    props = feature["properties"]
    assert "region_id" in props
    assert props["pixel_count"] == 100
    assert "area_m2" in props
    assert "centroid_crs" in props

    # Verify GeoJSON serializability with standard json
    serialized = json.dumps(geojson)
    deserialized = json.loads(serialized)
    assert deserialized["type"] == "FeatureCollection"


def test_8_geojson_crs_coordinate_preservation():
    """Verify GeoJSON polygon coordinates are in the native CRS space."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    prob = np.full((h, w), 0.8, dtype=np.float32)

    origin_x, origin_y = 600000.0, 5000000.0
    res_x, res_y = 10.0, 10.0
    meta = {
        "transform": from_origin(origin_x, origin_y, res_x, res_y),
        "crs": "EPSG:32632",
        "res": (res_x, res_y),
        "bounds": None,
    }

    _, geojson, _ = extract_change_regions(mask, prob, meta, min_region_pixels=20)
    coords = geojson["features"][0]["geometry"]["coordinates"][0]

    # Verify all coordinates reflect the CRS origin scale
    for pt in coords:
        assert pt[0] >= origin_x
        assert pt[1] <= origin_y


# ---------------------------------------------------------------------------
# 4. Labeled Overlay & Natural-Language Interpretation Tests
# ---------------------------------------------------------------------------

def test_9_labeled_region_overlay_generation():
    """Verify labeled region visualization renders without modifying underlying dimensions."""
    rgb = np.full((120, 120, 3), 100, dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[30:50, 30:50] = 1  # 400 px

    regions = [
        {
            "region_id": 1,
            "rank": 1,
            "pixel_count": 400,
            "bbox_px": [30, 30, 49, 49],
            "centroid_px": [39.5, 39.5],
        }
    ]

    pil_overlay = render_labeled_region_overlay(rgb, mask, regions, max_labels=5)
    assert pil_overlay.size == (120, 120)
    overlay_arr = np.array(pil_overlay)
    # Verify pixels were modified (colored box / highlight drawn)
    assert not np.array_equal(overlay_arr, rgb)


def test_10_structured_interpretation_formatting():
    """Verify format_change_detection_summary produces all 8 required sections."""
    stats = {
        "total_pixels": 10000,
        "changed_pixels": 500,
        "unchanged_pixels": 9500,
        "percentage_changed": 5.0,
        "percentage_unchanged": 95.0,
        "mean_prediction_confidence": 0.88,
        "changed_pixel_confidence": 0.92,
        "unchanged_pixel_confidence": 0.86,
        "low_confidence_pixel_percentage": 3.2,
        "change_threshold": 0.5,
        "raster_dimensions": {"width": 100, "height": 100},
        "resolution": [10.0, 10.0],
    }
    regions = [
        {
            "region_id": 1,
            "rank": 1,
            "pixel_count": 400,
            "area_m2": 40000.0,
            "percentage_of_change": 80.0,
        },
        {
            "region_id": 2,
            "rank": 2,
            "pixel_count": 100,
            "area_m2": 10000.0,
            "percentage_of_change": 20.0,
        },
    ]
    area_info = {
        "changed_area_m2": 50000.0,
        "changed_area_hectares": 5.0,
        "changed_area_km2": 0.05,
        "area_calculation_method": "projected_planar_metric",
    }

    summary = format_change_detection_summary(
        stats=stats,
        regions=regions,
        area_info=area_info,
        crs_str="EPSG:32633",
        device_str="cpu",
    )

    required_sections = [
        "### SUMMARY",
        "### WHAT CHANGED",
        "### HOW MUCH CHANGED",
        "### SPATIAL DISTRIBUTION",
        "### CONFIDENCE",
        "### GEOSPATIAL INFORMATION",
        "### ARTIFACTS",
        "### LIMITATIONS",
    ]

    for sec in required_sections:
        assert sec in summary, f"Missing section: {sec}"

    # Verify no fabricated semantic claims
    assert "new buildings were constructed" not in summary.lower()
    assert "roads were built" not in summary.lower()
    assert "5.0%" in summary
    assert "LEVIR-CD" in summary


# ---------------------------------------------------------------------------
# 5. Output Isolation & Integration Tests
# ---------------------------------------------------------------------------

def test_11_isolated_output_directory_creation():
    """Verify each prediction generates an isolated run_<id> subdirectory containing all 6 artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0.tif"
        t1 = tmp / "t1.tif"
        create_synthetic_test_raster(t0, seed=1)
        create_synthetic_test_raster(t1, seed=2)

        model = OpenCDBITModel()
        res = model.predict({"t0": str(t0), "t1": str(t1)})
        assert res.status == "success"

        run_id = res.raw_scores["run_id"]
        assert run_id.startswith("run_")

        # Verify all 6 artifacts exist in the isolated directory
        run_dir = Path(model.output_dir) / run_id
        assert run_dir.exists() and run_dir.is_dir()

        expected_files = [
            "change_mask.tif",
            "change_prob.tif",
            "change_overlay.png",
            "change_regions_vis.png",
            "change_regions.geojson",
            "stats.json",
        ]
        for f in expected_files:
            assert (run_dir / f).exists(), f"Missing artifact: {f}"

        # Verify GeoJSON is valid JSON
        with open(run_dir / "change_regions.geojson", "r", encoding="utf-8") as f_geo:
            geo_parsed = json.load(f_geo)
            assert geo_parsed["type"] == "FeatureCollection"


def test_12_controller_integration_phase3c():
    """Verify AgentController coordinates full Phase 3C workflow and returns enhanced evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = tmp / "t0_ctrl.tif"
        t1 = tmp / "t1_ctrl.tif"
        create_synthetic_test_raster(t0, seed=3)
        create_synthetic_test_raster(t1, seed=4)

        controller = AgentController()
        slots = [
            SlotAssignment(slot_id="t0", file_path=str(t0)),
            SlotAssignment(slot_id="t1", file_path=str(t1)),
        ]

        res = controller.analyze(
            query="Where are the changed buildings between these two dates?",
            slots=slots,
            input_mode=InputMode.I4_BITEMPORAL_PAIR,
            change_threshold=0.5,
            min_change_region_pixels=15,
        )

        assert res.plan.task == TaskType.BI_TEMPORAL_CHANGE.value
        assert "SUMMARY" in res.result_text
        assert "WHAT CHANGED" in res.result_text
        assert "SPATIAL DISTRIBUTION" in res.result_text
        assert "LIMITATIONS" in res.result_text

        titles = [e.title for e in res.evidence]
        assert "Binary Change Mask (GeoTIFF)" in titles
        assert "Change Probability Map (GeoTIFF)" in titles
        assert "Change Map Visual Overlay" in titles
        assert "Change Regions with Bounding Boxes" in titles
        assert "Change Regions Vector Map (GeoJSON)" in titles
        assert "Bi-Temporal Change Statistics" in titles
