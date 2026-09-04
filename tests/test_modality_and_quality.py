from pathlib import Path
from satquery.domain.schemas import Modality, QualityLevel
from satquery.preprocess.raster import (
    assess_image_quality,
    detect_modality,
    inspect_raster,
)


def test_7_modality_detection(
    optical_geotiff: Path,
    multispectral_geotiff: Path,
    sar_geotiff: Path,
    no_crs_geotiff: Path,
):
    # Optical (3 bands)
    opt_meta = inspect_raster(optical_geotiff)
    assert detect_modality(opt_meta) == Modality.OPTICAL

    # Multispectral (>3 bands)
    multi_meta = inspect_raster(multispectral_geotiff)
    assert detect_modality(multi_meta) == Modality.MULTISPECTRAL

    # SAR (1 band with SAR / polarization metadata)
    sar_meta = inspect_raster(sar_geotiff)
    assert detect_modality(sar_meta) == Modality.SAR

    # 1 band without SAR tags -> UNKNOWN (never guess)
    sar_meta_no_tags = sar_meta.model_copy(update={"metadata": {}})
    assert detect_modality(sar_meta_no_tags) == Modality.UNKNOWN


def test_8_image_quality(
    optical_geotiff: Path,
    uniform_geotiff: Path,
    tiny_geotiff: Path,
    corrupt_file: Path,
):
    # Good quality image
    q_good = assess_image_quality(optical_geotiff)
    assert q_good.level == QualityLevel.GOOD
    assert len(q_good.reasons) > 0

    # Uniform / blank image (variance ~0)
    q_uniform = assess_image_quality(uniform_geotiff)
    assert q_uniform.level == QualityLevel.INSUFFICIENT
    assert any("uniform" in r.lower() for r in q_uniform.reasons)

    # Sub-minimum dimension (8x8 < 16x16)
    q_tiny = assess_image_quality(tiny_geotiff)
    assert q_tiny.level == QualityLevel.INSUFFICIENT
    assert any("too small" in r.lower() for r in q_tiny.reasons)

    # Corrupt / invalid file
    q_corrupt = assess_image_quality(corrupt_file)
    assert q_corrupt.level == QualityLevel.INSUFFICIENT
