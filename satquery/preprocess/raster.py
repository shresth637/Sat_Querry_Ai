from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from satquery.domain.schemas import (
    ImageQuality,
    Modality,
    QualityLevel,
    RasterMeta,
)


def inspect_raster(path: Union[str, Path]) -> RasterMeta:
    """Inspect a raster file using rasterio.

    Extracts real metadata without fabrication.
    Missing attributes remain None.
    Handles missing or invalid files gracefully without raising exceptions.
    """
    file_path = Path(path)
    filename = file_path.name

    if not file_path.exists():
        return RasterMeta(
            filename=filename,
            is_valid=False,
            error_message=f"File not found: {file_path}",
        )

    try:
        with rasterio.open(file_path) as src:
            crs_str = None
            if src.crs:
                crs_str = src.crs.to_string()

            transform_list = None
            if src.transform:
                transform_list = [float(v) for v in src.transform][:6]

            bounds_list = None
            if src.bounds:
                bounds_list = [
                    float(src.bounds.left),
                    float(src.bounds.bottom),
                    float(src.bounds.right),
                    float(src.bounds.top),
                ]

            resolution_list = None
            if src.res:
                resolution_list = [float(src.res[0]), float(src.res[1])]

            nodata_val = None
            if src.nodata is not None:
                try:
                    nodata_val = float(src.nodata)
                except (ValueError, TypeError):
                    nodata_val = None

            # Collect driver-level tags / metadata safely
            meta_dict: dict[str, Any] = {}
            try:
                tags = src.tags()
                if tags:
                    meta_dict.update(tags)
            except Exception:
                pass

            dtype_str = src.dtypes[0] if src.dtypes else None

            return RasterMeta(
                filename=filename,
                format=src.driver,
                width=int(src.width),
                height=int(src.height),
                band_count=int(src.count),
                dtype=str(dtype_str) if dtype_str else None,
                crs=crs_str,
                transform=transform_list,
                bounds=bounds_list,
                resolution=resolution_list,
                nodata=nodata_val,
                metadata=meta_dict,
                is_valid=True,
                error_message=None,
            )

    except (RasterioIOError, Exception) as exc:
        return RasterMeta(
            filename=filename,
            is_valid=False,
            error_message=f"Raster inspection failed: {str(exc)}",
        )


def detect_modality(
    meta: RasterMeta,
    sample_array: Optional[np.ndarray] = None,
) -> Modality:
    """Conservatively determine the sensor modality.

    Returns OPTICAL, MULTISPECTRAL, SAR, or UNKNOWN.
    Does NOT guess based on arbitrary filenames.
    Only returns a specific modality if band structure or metadata strongly indicates it.
    If ambiguous, returns Modality.UNKNOWN.
    """
    if not meta.is_valid or meta.band_count is None or meta.band_count == 0:
        return Modality.UNKNOWN

    tags_lower = {str(k).lower(): str(v).lower() for k, v in meta.metadata.items()}
    all_tag_text = " ".join(f"{k} {v}" for k, v in tags_lower.items())

    # Check for explicit SAR metadata signals (polarization, Sentinel-1, radar)
    sar_keywords = ["sar", "sentinel-1", "radar", "backscatter", "sigma0", "gamma0", "polarization", "polsar"]
    has_sar_tags = any(kw in all_tag_text for kw in sar_keywords)
    has_pol_tags = any(pol in all_tag_text for pol in ["vv", "vh", "hh", "hv"])

    if has_sar_tags or has_pol_tags:
        return Modality.SAR

    # Optical / Multispectral signals
    optical_keywords = ["optical", "rgb", "sentinel-2", "landsat", "planet", "true_color", "rgb_color"]
    has_optical_tags = any(kw in all_tag_text for kw in optical_keywords)

    if meta.band_count == 3:
        # 3-band raster is classically optical RGB
        return Modality.OPTICAL

    if meta.band_count > 3:
        # >3 bands indicates multispectral imagery (e.g. RGB + NIR, or 10-13 bands)
        return Modality.MULTISPECTRAL

    if meta.band_count == 1:
        # 1-band raster: could be panchromatic, elevation, thermal, or SAR without tags.
        # Conservative: unless optical tags exist, return UNKNOWN.
        if has_optical_tags:
            return Modality.OPTICAL
        return Modality.UNKNOWN

    if meta.band_count == 2:
        # 2-band raster: could be dual-pol SAR or dual-spectral.
        if has_optical_tags:
            return Modality.OPTICAL
        return Modality.UNKNOWN

    return Modality.UNKNOWN


def assess_image_quality(
    path: Union[str, Path],
    meta: Optional[RasterMeta] = None,
) -> ImageQuality:
    """Assess deterministic image quality factors.

    Checks:
    - Dimensions (minimum spatial size)
    - Valid pixel ratio (nodata/nan/inf presence)
    - Value uniformity (blank or near-zero variance images)
    - Metadata completeness (presence of CRS and georeferencing)

    Returns QualityLevel (GOOD, MEDIUM, LOW, INSUFFICIENT) with explicit reasons.
    Not a scientifically calibrated metric.
    """
    file_path = Path(path)
    if meta is None:
        meta = inspect_raster(file_path)

    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if not meta.is_valid:
        return ImageQuality(
            level=QualityLevel.INSUFFICIENT,
            reasons=[f"Invalid or unreadable raster: {meta.error_message or 'Unknown error'}"],
            metrics={"valid_file": False},
        )

    width = meta.width or 0
    height = meta.height or 0
    band_count = meta.band_count or 0

    metrics["width"] = width
    metrics["height"] = height
    metrics["band_count"] = band_count

    # Check 1: Dimensions
    if width < 16 or height < 16:
        reasons.append(f"Image dimensions too small ({width}x{height} < 16x16)")
        return ImageQuality(
            level=QualityLevel.INSUFFICIENT,
            reasons=reasons,
            metrics=metrics,
        )

    if width < 64 or height < 64:
        reasons.append(f"Image dimensions are low ({width}x{height} < 64x64)")

    # Check 2: Metadata completeness
    has_crs = meta.crs is not None
    has_transform = meta.transform is not None
    metrics["has_crs"] = has_crs
    metrics["has_transform"] = has_transform

    if not has_crs:
        reasons.append("Missing spatial CRS georeferencing")

    # Check 3: Pixel content analysis (sample first band)
    try:
        with rasterio.open(file_path) as src:
            # Read first band, capping sample size to 1024x1024 for speed if large
            out_shape = (min(src.height, 512), min(src.width, 512))
            data = src.read(1, out_shape=out_shape).astype(np.float32)

            # Mask nodata / nan / inf
            valid_mask = np.isfinite(data)
            if meta.nodata is not None:
                valid_mask = valid_mask & (data != meta.nodata)

            total_pixels = data.size
            valid_pixels = int(np.sum(valid_mask))
            valid_ratio = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

            metrics["valid_pixel_ratio"] = round(valid_ratio, 4)

            if valid_ratio < 0.10:
                reasons.append(f"Severe missing data: only {valid_ratio:.1%} valid pixels")
                return ImageQuality(
                    level=QualityLevel.INSUFFICIENT,
                    reasons=reasons,
                    metrics=metrics,
                )

            if valid_ratio < 0.70:
                reasons.append(f"Substantial missing data: {valid_ratio:.1%} valid pixels")

            # Check for extreme uniformity (e.g. completely black, white, or constant)
            if valid_pixels > 1:
                valid_values = data[valid_mask]
                std_dev = float(np.std(valid_values))
                val_range = float(np.ptp(valid_values))
                metrics["std_dev"] = round(std_dev, 6)
                metrics["value_range"] = round(val_range, 6)

                if std_dev < 1e-4 or val_range < 1e-4:
                    reasons.append("Extremely uniform image (near-zero pixel variance)")
                    return ImageQuality(
                        level=QualityLevel.INSUFFICIENT,
                        reasons=reasons,
                        metrics=metrics,
                    )
                elif std_dev < 1e-2:
                    reasons.append("Very low contrast / low dynamic range")

    except Exception as exc:
        reasons.append(f"Could not read pixel data for quality check: {str(exc)}")
        return ImageQuality(
            level=QualityLevel.LOW,
            reasons=reasons,
            metrics=metrics,
        )

    # Determine overall quality level based on accumulated flags
    critical_flags = [r for r in reasons if "too small" in r or "zero" in r or "Severe" in r]
    if critical_flags:
        level = QualityLevel.INSUFFICIENT
    elif len(reasons) >= 2 or (metrics.get("valid_pixel_ratio", 1.0) < 0.85):
        level = QualityLevel.LOW
    elif len(reasons) == 1:
        level = QualityLevel.MEDIUM
    else:
        level = QualityLevel.GOOD
        reasons.append("Dimensions, pixel validity, and contrast meet quality checks")

    return ImageQuality(
        level=level,
        reasons=reasons,
        metrics=metrics,
    )
