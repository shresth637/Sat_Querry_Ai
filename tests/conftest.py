from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def optical_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic 3-band optical GeoTIFF with EPSG:4326."""
    file_path = tmp_path / "optical.tif"
    width, height = 64, 64
    transform = from_origin(-122.0, 37.0, 0.001, 0.001)

    # 3 bands: R, G, B gradients
    data = np.zeros((3, height, width), dtype=np.uint8)
    data[0] = np.linspace(10, 240, width * height, dtype=np.uint8).reshape((height, width))
    data[1] = np.linspace(20, 200, width * height, dtype=np.uint8).reshape((height, width))
    data[2] = np.linspace(30, 180, width * height, dtype=np.uint8).reshape((height, width))

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(SENSOR="Sentinel-2", DESCRIPTION="Optical RGB Scene")

    return file_path


@pytest.fixture
def multispectral_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic 4-band multispectral GeoTIFF."""
    file_path = tmp_path / "multispectral.tif"
    width, height = 64, 64
    transform = from_origin(-122.0, 37.0, 0.001, 0.001)
    data = np.ones((4, height, width), dtype=np.uint16) * 1000

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def sar_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic 1-band SAR GeoTIFF with Sentinel-1/polarization tags."""
    file_path = tmp_path / "sar.tif"
    width, height = 64, 64
    transform = from_origin(-122.0, 37.0, 0.001, 0.001)

    # 1 band float32 backscatter
    data = np.random.uniform(0.01, 0.8, size=(1, height, width)).astype(np.float32)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(POLARIZATION="VV", SENSOR="Sentinel-1", MODE="SAR-IW")

    return file_path


@pytest.fixture
def no_crs_geotiff(tmp_path: Path) -> Path:
    """Create a raster without any CRS."""
    file_path = tmp_path / "no_crs.tif"
    width, height = 64, 64
    data = np.random.randint(10, 200, size=(3, height, width), dtype=np.uint8)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def uniform_geotiff(tmp_path: Path) -> Path:
    """Create an extremely uniform (all zero) raster."""
    file_path = tmp_path / "uniform.tif"
    width, height = 64, 64
    data = np.zeros((1, height, width), dtype=np.uint8)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def tiny_geotiff(tmp_path: Path) -> Path:
    """Create a sub-minimum dimension raster (8x8)."""
    file_path = tmp_path / "tiny.tif"
    data = np.ones((1, 8, 8), dtype=np.uint8) * 128

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    """Create a corrupt file with non-raster content."""
    file_path = tmp_path / "corrupt.tif"
    file_path.write_bytes(b"THIS_IS_NOT_A_VALID_TIFF_HEADER")
    return file_path


@pytest.fixture
def different_crs_geotiff(tmp_path: Path) -> Path:
    """Create a raster with a different CRS (EPSG:32632, UTM)."""
    file_path = tmp_path / "utm.tif"
    width, height = 64, 64
    transform = from_origin(500000.0, 4100000.0, 10.0, 10.0)
    data = np.random.randint(10, 200, size=(1, height, width), dtype=np.uint8)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path
