from pathlib import Path
from satquery.domain.schemas import (
    InputMode,
    Modality,
    SlotAssignment,
    SpatialCompatibility,
    ValidationStatus,
)
from satquery.preprocess.raster import inspect_raster
from satquery.validation.validator import validate_input


def test_9_single_image_validation(optical_geotiff: Path, sar_geotiff: Path):
    opt_meta = inspect_raster(optical_geotiff)
    sar_meta = inspect_raster(sar_geotiff)

    # I1 with valid optical
    slot_opt = SlotAssignment(slot_id="image", file_path=str(optical_geotiff))
    report_i1 = validate_input([slot_opt], {"image": opt_meta}, InputMode.I1_SINGLE_OPTICAL)
    assert report_i1.status == ValidationStatus.PASS

    # I1 with SAR -> should fail modality mismatch
    slot_sar = SlotAssignment(slot_id="image", file_path=str(sar_geotiff))
    report_i1_sar = validate_input([slot_sar], {"image": sar_meta}, InputMode.I1_SINGLE_OPTICAL)
    assert report_i1_sar.status == ValidationStatus.FAIL
    assert any(c.code == "MODALITY_MISMATCH" for c in report_i1_sar.checks)

    # I2 with SAR -> should pass
    report_i2 = validate_input([slot_sar], {"image": sar_meta}, InputMode.I2_SINGLE_SAR)
    assert report_i2.status == ValidationStatus.PASS


def test_10_optical_sar_validation(optical_geotiff: Path, sar_geotiff: Path):
    opt_meta = inspect_raster(optical_geotiff)
    sar_meta = inspect_raster(sar_geotiff)

    slots = [
        SlotAssignment(slot_id="optical", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="sar", file_path=str(sar_geotiff)),
    ]
    metas = {"optical": opt_meta, "sar": sar_meta}

    report = validate_input(slots, metas, InputMode.I3_OPTICAL_SAR_PAIR)
    assert report.status == ValidationStatus.PASS
    # Both share identical grid, bounds, and EPSG:4326
    assert report.spatial_compatibility == SpatialCompatibility.CONFIRMED


def test_11_bitemporal_validation(optical_geotiff: Path):
    meta = inspect_raster(optical_geotiff)
    slots = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(optical_geotiff)),
    ]
    metas = {"t0": meta, "t1": meta}

    report = validate_input(slots, metas, InputMode.I4_BITEMPORAL_PAIR)
    assert report.status == ValidationStatus.PASS
    assert report.spatial_compatibility == SpatialCompatibility.CONFIRMED


def test_12_incompatible_images(optical_geotiff: Path, different_crs_geotiff: Path, no_crs_geotiff: Path):
    meta_4326 = inspect_raster(optical_geotiff)
    meta_32632 = inspect_raster(different_crs_geotiff)
    meta_no_crs = inspect_raster(no_crs_geotiff)

    # Incompatible CRS (EPSG:4326 vs EPSG:32632)
    slots_diff_crs = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(different_crs_geotiff)),
    ]
    metas_diff_crs = {"t0": meta_4326, "t1": meta_32632}
    report_diff_crs = validate_input(slots_diff_crs, metas_diff_crs, InputMode.I4_BITEMPORAL_PAIR)
    assert report_diff_crs.status == ValidationStatus.FAIL
    assert report_diff_crs.spatial_compatibility == SpatialCompatibility.INCOMPATIBLE
    assert any(c.code == "CRS_MISMATCH" for c in report_diff_crs.checks)

    # Missing CRS -> spatial compatibility is UNKNOWN, never assume co-registered!
    slots_no_crs = [
        SlotAssignment(slot_id="t0", file_path=str(optical_geotiff)),
        SlotAssignment(slot_id="t1", file_path=str(no_crs_geotiff)),
    ]
    metas_no_crs = {"t0": meta_4326, "t1": meta_no_crs}
    report_no_crs = validate_input(slots_no_crs, metas_no_crs, InputMode.I4_BITEMPORAL_PAIR)
    assert report_no_crs.spatial_compatibility == SpatialCompatibility.UNKNOWN
