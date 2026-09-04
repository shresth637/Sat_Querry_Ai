from pathlib import Path
from satquery.preprocess.raster import inspect_raster
from satquery.tools.base import MetadataExtractionTool


def test_1_valid_geotiff_inspection(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    assert meta.is_valid is True
    assert meta.filename == "optical.tif"
    assert meta.format == "GTiff"
    assert meta.width == 64
    assert meta.height == 64
    assert meta.band_count == 3
    assert meta.dtype == "uint8"
    assert meta.crs is not None
    assert "4326" in meta.crs
    assert meta.transform is not None
    assert len(meta.transform) == 6
    assert meta.bounds is not None
    assert meta.resolution is not None
    assert meta.error_message is None


def test_2_invalid_file(corrupt_file: Path):
    meta = inspect_raster(corrupt_file)
    assert meta.is_valid is False
    assert meta.error_message is not None
    assert "inspection failed" in meta.error_message.lower() or "error" in meta.error_message.lower()
    # Attributes should remain None
    assert meta.width is None
    assert meta.height is None
    assert meta.crs is None


def test_3_missing_file():
    missing_path = Path("non_existent_scene_12345.tif")
    meta = inspect_raster(missing_path)
    assert meta.is_valid is False
    assert meta.error_message is not None
    assert "File not found" in meta.error_message
    assert meta.crs is None


def test_4_tiff_format(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    assert meta.format in ["GTiff", "TIFF"]


def test_5_metadata_extraction(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    tool = MetadataExtractionTool()
    extracted = tool.execute(meta)

    assert extracted["filename"] == "optical.tif"
    assert extracted["format"] == "GTiff"
    assert extracted["dimensions"] == "64 x 64"
    assert extracted["bands"] == 3
    assert "4326" in extracted["crs"]
    assert extracted["tags"].get("SENSOR") == "Sentinel-2"


def test_6_unknown_crs(no_crs_geotiff: Path):
    meta = inspect_raster(no_crs_geotiff)
    assert meta.is_valid is True
    # CRS must remain None / missing, never invented!
    assert meta.crs is None
