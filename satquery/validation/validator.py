from pathlib import Path
from typing import Optional

from satquery.domain.schemas import (
    InputMode,
    Modality,
    RasterMeta,
    SlotAssignment,
    SpatialCompatibility,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from satquery.preprocess.raster import detect_modality


def check_bounds_overlap(b1: list[float], b2: list[float]) -> bool:
    """Check if two bounding boxes [left, bottom, right, top] overlap."""
    left = max(b1[0], b2[0])
    bottom = max(b1[1], b2[1])
    right = min(b1[2], b2[2])
    top = min(b1[3], b2[3])
    return (left < right) and (bottom < top)


def validate_input(
    slots: list[SlotAssignment],
    metas: dict[str, RasterMeta],
    input_mode: InputMode,
    allow_png_jpeg_benchmark: bool = False,
) -> ValidationReport:
    """Validate session inputs against the requested InputMode.

    Never assumes two images are co-registered.
    Evaluates format, slot count, modality, and spatial compatibility.
    """
    checks: list[ValidationCheck] = []
    spatial_comp: Optional[SpatialCompatibility] = None
    details: dict = {}

    # 1. Check slot count
    expected_counts = {
        InputMode.I1_SINGLE_OPTICAL: 1,
        InputMode.I2_SINGLE_SAR: 1,
        InputMode.I3_OPTICAL_SAR_PAIR: 2,
        InputMode.I4_BITEMPORAL_PAIR: 2,
    }
    expected_count = expected_counts[input_mode]

    if len(slots) != expected_count:
        checks.append(
            ValidationCheck(
                code="SLOT_COUNT_MISMATCH",
                status=ValidationStatus.FAIL,
                message=f"Input mode {input_mode.value} requires {expected_count} image(s), but received {len(slots)}.",
                is_blocking=True,
            )
        )

    # 2. Check each slot's file readability and format
    slot_detected_modalities: dict[str, Modality] = {}
    for slot in slots:
        meta = metas.get(slot.slot_id)
        if meta is None or not meta.is_valid:
            checks.append(
                ValidationCheck(
                    code="FILE_UNREADABLE",
                    status=ValidationStatus.FAIL,
                    message=f"Slot '{slot.slot_id}' file could not be read: {meta.error_message if meta else 'Metadata missing'}.",
                    is_blocking=True,
                )
            )
            continue

        # Format check
        ext = Path(slot.file_path).suffix.lower()
        fmt = (meta.format or "").lower()
        is_tiff = ext in [".tif", ".tiff"] or "gtiff" in fmt or "tiff" in fmt
        is_png_jpeg = ext in [".png", ".jpg", ".jpeg"] or "png" in fmt or "jpeg" in fmt

        if is_png_jpeg and not allow_png_jpeg_benchmark:
            checks.append(
                ValidationCheck(
                    code="FORMAT_DISALLOWED",
                    status=ValidationStatus.FAIL,
                    message=f"Slot '{slot.slot_id}' is {ext.upper()}. PNG/JPEG disallowed unless explicit benchmark flag is enabled.",
                    is_blocking=True,
                )
            )
        elif not is_tiff and not is_png_jpeg:
            checks.append(
                ValidationCheck(
                    code="FORMAT_UNRECOGNIZED",
                    status=ValidationStatus.FAIL,
                    message=f"Slot '{slot.slot_id}' format '{meta.format}' is not an approved geospatial raster format.",
                    is_blocking=True,
                )
            )

        # Modality check per slot
        modality = slot.declared_modality or detect_modality(meta)
        slot_detected_modalities[slot.slot_id] = modality

    # 3. Modality compatibility checks for specific input modes
    if input_mode == InputMode.I1_SINGLE_OPTICAL and slots:
        mod = slot_detected_modalities.get(slots[0].slot_id)
        if mod == Modality.SAR:
            checks.append(
                ValidationCheck(
                    code="MODALITY_MISMATCH",
                    status=ValidationStatus.FAIL,
                    message="I1 requires an optical/multispectral image, but SAR imagery was detected.",
                    is_blocking=True,
                )
            )
        elif mod == Modality.UNKNOWN:
            checks.append(
                ValidationCheck(
                    code="MODALITY_UNCERTAIN",
                    status=ValidationStatus.WARN,
                    message="Modality could not be verified as optical; continuing under declared slot.",
                    is_blocking=False,
                )
            )

    elif input_mode == InputMode.I2_SINGLE_SAR and slots:
        mod = slot_detected_modalities.get(slots[0].slot_id)
        if mod in [Modality.OPTICAL, Modality.MULTISPECTRAL]:
            checks.append(
                ValidationCheck(
                    code="MODALITY_MISMATCH",
                    status=ValidationStatus.FAIL,
                    message="I2 requires a SAR image, but optical imagery was detected.",
                    is_blocking=True,
                )
            )
        elif mod == Modality.UNKNOWN:
            checks.append(
                ValidationCheck(
                    code="MODALITY_UNCERTAIN",
                    status=ValidationStatus.WARN,
                    message="Modality could not be verified as SAR; continuing under declared slot.",
                    is_blocking=False,
                )
            )

    elif input_mode == InputMode.I3_OPTICAL_SAR_PAIR and len(slots) == 2:
        mods = [slot_detected_modalities.get(s.slot_id) for s in slots]
        has_optical = any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in mods)
        has_sar = any(m == Modality.SAR for m in mods)

        if not has_optical or not has_sar:
            checks.append(
                ValidationCheck(
                    code="PAIR_MODALITY_INCOMPLETE",
                    status=ValidationStatus.WARN,
                    message=f"I3 requires one optical and one SAR image. Detected: {[m.value if m else 'UNKNOWN' for m in mods]}.",
                    is_blocking=False,
                )
            )

    # 4. Spatial compatibility and co-registration check for pairs
    if input_mode in [InputMode.I3_OPTICAL_SAR_PAIR, InputMode.I4_BITEMPORAL_PAIR] and len(slots) == 2:
        meta1 = metas.get(slots[0].slot_id)
        meta2 = metas.get(slots[1].slot_id)

        if meta1 and meta2 and meta1.is_valid and meta2.is_valid:
            crs1, crs2 = meta1.crs, meta2.crs
            b1, b2 = meta1.bounds, meta2.bounds
            dim1 = (meta1.width, meta1.height)
            dim2 = (meta2.width, meta2.height)

            details["slot1_dims"] = dim1
            details["slot2_dims"] = dim2
            details["slot1_crs"] = crs1
            details["slot2_crs"] = crs2

            if not crs1 or not crs2:
                spatial_comp = SpatialCompatibility.UNKNOWN
                checks.append(
                    ValidationCheck(
                        code="CRS_MISSING",
                        status=ValidationStatus.WARN,
                        message="One or both images lack CRS georeferencing. Co-registration is UNKNOWN.",
                        is_blocking=False,
                    )
                )
            elif crs1 != crs2:
                spatial_comp = SpatialCompatibility.INCOMPATIBLE
                checks.append(
                    ValidationCheck(
                        code="CRS_MISMATCH",
                        status=ValidationStatus.FAIL,
                        message=f"CRS mismatch between pairs ({crs1} vs {crs2}). Re-projection required.",
                        is_blocking=True,
                    )
                )
            else:
                # Same CRS: check bounds overlap and dimensions
                if b1 and b2:
                    if not check_bounds_overlap(b1, b2):
                        spatial_comp = SpatialCompatibility.INCOMPATIBLE
                        checks.append(
                            ValidationCheck(
                                code="BOUNDS_DISJOINT",
                                status=ValidationStatus.FAIL,
                                message="Geographic bounds do not intersect. Pair is INCOMPATIBLE.",
                                is_blocking=True,
                            )
                        )
                    else:
                        # Check exact grid alignment
                        bounds_match = all(abs(v1 - v2) < 1e-4 for v1, v2 in zip(b1, b2))
                        dims_match = (dim1 == dim2)

                        if bounds_match and dims_match:
                            spatial_comp = SpatialCompatibility.CONFIRMED
                            checks.append(
                                ValidationCheck(
                                    code="COREGISTRATION_CONFIRMED",
                                    status=ValidationStatus.PASS,
                                    message="CRS, bounds, and pixel dimensions match. Co-registration CONFIRMED.",
                                    is_blocking=False,
                                )
                            )
                        else:
                            spatial_comp = SpatialCompatibility.INCOMPATIBLE
                            checks.append(
                                ValidationCheck(
                                    code="GRID_MISMATCH",
                                    status=ValidationStatus.WARN,
                                    message="Images overlap in space but have different resolutions or extents. Pixel resampling required.",
                                    is_blocking=False,
                                )
                            )
                else:
                    spatial_comp = SpatialCompatibility.UNKNOWN

    # Determine overall status
    if any(c.is_blocking or c.status == ValidationStatus.FAIL for c in checks):
        overall_status = ValidationStatus.FAIL
    elif any(c.status == ValidationStatus.WARN for c in checks):
        overall_status = ValidationStatus.WARN
    else:
        overall_status = ValidationStatus.PASS
        checks.append(
            ValidationCheck(
                code="ALL_CHECKS_PASSED",
                status=ValidationStatus.PASS,
                message="All input and compatibility validation checks passed.",
                is_blocking=False,
            )
        )

    return ValidationReport(
        status=overall_status,
        checks=checks,
        spatial_compatibility=spatial_comp,
        details=details,
    )
