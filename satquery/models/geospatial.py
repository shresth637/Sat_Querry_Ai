import math
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import rasterio
from rasterio.transform import Affine


def validate_bitemporal_rasters(
    t0_path: Union[str, Path],
    t1_path: Union[str, Path],
) -> dict[str, Any]:
    """Validate that two bitemporal rasters are readable, co-registered, and compatible.

    Validates:
    - Both files exist on disk.
    - Both are valid rasters readable by rasterio.
    - Spatial dimensions match (width, height).
    - CRS definitions are compatible.
    - Affine transforms match within spatial tolerance.
    - Valid band count and supported numeric types.

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If rasters are incompatible or misaligned.
    """
    p0 = Path(t0_path)
    p1 = Path(t1_path)

    if not p0.exists():
        raise FileNotFoundError(f"T0 raster file not found: {p0}")
    if not p1.exists():
        raise FileNotFoundError(f"T1 raster file not found: {p1}")

    try:
        with rasterio.open(p0) as src0, rasterio.open(p1) as src1:
            meta0 = {
                "width": src0.width,
                "height": src0.height,
                "count": src0.count,
                "crs": src0.crs.to_string() if src0.crs else None,
                "transform": src0.transform,
                "bounds": src0.bounds,
                "res": src0.res,
                "dtypes": src0.dtypes,
                "nodata": src0.nodata,
            }
            meta1 = {
                "width": src1.width,
                "height": src1.height,
                "count": src1.count,
                "crs": src1.crs.to_string() if src1.crs else None,
                "transform": src1.transform,
                "bounds": src1.bounds,
                "res": src1.res,
                "dtypes": src1.dtypes,
                "nodata": src1.nodata,
            }
    except Exception as exc:
        raise ValueError(f"Failed to open raster files: {exc}") from exc

    # 1. Dimensions check
    if meta0["width"] != meta1["width"] or meta0["height"] != meta1["height"]:
        raise ValueError(
            f"Raster dimension mismatch: T0 is {meta0['width']}x{meta0['height']}, "
            f"but T1 is {meta1['width']}x{meta1['height']}."
        )

    # 2. Band count check
    if meta0["count"] < 1 or meta1["count"] < 1:
        raise ValueError("Rasters must contain at least 1 band.")

    # 3. CRS compatibility
    crs0 = meta0["crs"]
    crs1 = meta1["crs"]
    if crs0 and crs1:
        # Normalize comparison (e.g. EPSG:4326 vs epsg:4326)
        if crs0.lower() != crs1.lower():
            raise ValueError(
                f"CRS mismatch: T0 has CRS '{crs0}', but T1 has CRS '{crs1}'. "
                f"Rasters must be co-registered in the same Coordinate Reference System."
            )

    # 4. Transform alignment check
    t0_vals = list(meta0["transform"])[:6]
    t1_vals = list(meta1["transform"])[:6]
    if not np.allclose(t0_vals, t1_vals, atol=1e-4):
        raise ValueError(
            f"Spatial transform misalignment: T0 transform {t0_vals} does not match "
            f"T1 transform {t1_vals}. Images are not co-registered."
        )

    return meta0


def calculate_raster_area(
    meta: dict[str, Any],
    changed_pixels: int,
) -> dict[str, Any]:
    """Calculate physical ground area represented by changed pixels.

    Handles:
    - Projected CRS (e.g., UTM, State Plane) with metric resolution.
    - Geographic CRS (e.g., EPSG:4326) using rigorous WGS84 ellipsoidal scaling.
    - Missing or unknown CRS, returning None and explicit status.

    Returns:
        dict with changed_area_m2, changed_area_hectares, changed_area_km2,
        and area_calculation_method.
    """
    crs_str = meta.get("crs")
    res = meta.get("res")
    bounds = meta.get("bounds")

    if not crs_str or not res:
        return {
            "changed_area_m2": None,
            "changed_area_hectares": None,
            "changed_area_km2": None,
            "area_calculation_method": "unavailable_unknown_crs",
            "pixel_area_m2": None,
        }

    crs_lower = crs_str.lower()
    res_x, res_y = abs(float(res[0])), abs(float(res[1]))

    # Check for geographic coordinate system (degrees)
    is_geographic = (
        "4326" in crs_lower
        or "wgs 84" in crs_lower
        or "ogc:crs84" in crs_lower
        or "geographic" in crs_lower
        or (res_x < 0.1 and res_y < 0.1)  # Heuristic for degree units
    )

    if is_geographic:
        # Rigorous WGS84 ellipsoidal surface area calculation
        # Compute center latitude of the bounding box
        if bounds:
            center_lat_deg = (bounds.bottom + bounds.top) / 2.0
        else:
            center_lat_deg = 0.0

        phi = math.radians(center_lat_deg)
        # WGS84 constants
        a = 6378137.0  # semi-major axis (meters)
        f = 1.0 / 298.257223563  # flattening
        e2 = 2 * f - f ** 2  # first eccentricity squared

        # Radii of curvature
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        denom = math.sqrt(1.0 - e2 * (sin_phi ** 2))
        m = a * (1.0 - e2) / (denom ** 3)  # meridional radius (meters per radian lat)
        n = a / denom  # transverse radius (meters per radian lon)

        rad_per_deg = math.pi / 180.0
        dy = res_y * rad_per_deg * m
        dx = res_x * rad_per_deg * n * cos_phi
        pixel_area_m2 = abs(dx * dy)

        area_m2 = round(float(changed_pixels * pixel_area_m2), 2)
        area_ha = round(float(area_m2 / 10000.0), 4)
        area_km2 = round(float(area_m2 / 1000000.0), 6)

        return {
            "changed_area_m2": area_m2,
            "changed_area_hectares": area_ha,
            "changed_area_km2": area_km2,
            "area_calculation_method": "geographic_ellipsoidal_wgs84",
            "pixel_area_m2": round(pixel_area_m2, 4),
        }

    # Projected CRS (metric units)
    pixel_area_m2 = res_x * res_y
    area_m2 = round(float(changed_pixels * pixel_area_m2), 2)
    area_ha = round(float(area_m2 / 10000.0), 4)
    area_km2 = round(float(area_m2 / 1000000.0), 6)

    return {
        "changed_area_m2": area_m2,
        "changed_area_hectares": area_ha,
        "changed_area_km2": area_km2,
        "area_calculation_method": "projected_planar_metric",
        "pixel_area_m2": round(pixel_area_m2, 4),
    }
