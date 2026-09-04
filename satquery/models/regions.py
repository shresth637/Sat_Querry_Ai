import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine

from satquery.models.geospatial import calculate_raster_area


def extract_change_regions(
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    meta: dict[str, Any],
    min_region_pixels: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Extract connected change regions from a binary change mask using Rasterio.

    Args:
        binary_mask: 2D uint8 numpy array (0 = unchanged, 1 = changed).
        prob_map: 2D float32 numpy array in [0, 1].
        meta: Raster metadata dictionary containing 'transform', 'crs', 'res', 'bounds'.
        min_region_pixels: Minimum pixel count threshold to filter noise.

    Returns:
        regions: List of structured region dictionaries sorted descending by pixel count.
        geojson_data: RFC 7946 GeoJSON FeatureCollection dictionary.
        spatial_concentration: Qualitative summary of spatial distribution (e.g., 'northwestern').
    """
    height, width = binary_mask.shape
    total_changed_pixels = int(np.sum(binary_mask == 1))

    if total_changed_pixels == 0:
        empty_geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": meta.get("crs") or "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": [],
        }
        return [], empty_geojson, "No change regions detected in the analyzed scene."

    transform: Affine = meta.get("transform") or Affine.identity()

    # Extract polygon shapes in pixel coordinate space (for bbox and centroid in px)
    pixel_shapes = list(
        shapes(
            binary_mask.astype(np.int32),
            mask=(binary_mask == 1),
            connectivity=8,
        )
    )

    # Extract polygon shapes in native geospatial CRS space
    geo_shapes = list(
        shapes(
            binary_mask.astype(np.int32),
            mask=(binary_mask == 1),
            transform=transform,
            connectivity=8,
        )
    )

    extracted_regions = []
    features = []

    # Map each shape to pixel and geospatial properties
    for idx, (px_geom, val) in enumerate(pixel_shapes):
        if val != 1:
            continue

        # Extract coordinates in pixel space
        px_coords = px_geom["coordinates"][0]  # exterior ring
        xs = [p[0] for p in px_coords]
        ys = [p[1] for p in px_coords]

        min_x = max(0, int(np.floor(min(xs))))
        max_x = min(width - 1, int(np.ceil(max(xs))))
        min_y = max(0, int(np.floor(min(ys))))
        max_y = min(height - 1, int(np.ceil(max(ys))))

        # Count actual changed pixels within bounding slice
        sub_mask = binary_mask[min_y : max_y + 1, min_x : max_x + 1]
        pixel_count = int(np.sum(sub_mask == 1))

        # Filter noise regions smaller than threshold
        if pixel_count < min_region_pixels:
            continue

        # Centroid in pixel coordinates
        cx_px = round(float(np.mean(xs)), 1)
        cy_px = round(float(np.mean(ys)), 1)

        # Centroid in CRS coordinates
        cx_crs, cy_crs = transform * (cx_px, cy_px)

        # Physical area calculation for this region
        region_area_info = calculate_raster_area(meta, pixel_count)
        area_m2 = region_area_info["changed_area_m2"]
        area_ha = region_area_info["changed_area_hectares"]

        # Mean change probability within this region
        sub_prob = prob_map[min_y : max_y + 1, min_x : max_x + 1]
        mean_prob = round(float(np.mean(sub_prob[sub_mask == 1])), 4) if pixel_count > 0 else 0.0

        pct_of_total_change = (
            round(float(pixel_count / total_changed_pixels) * 100.0, 2)
            if total_changed_pixels > 0
            else 0.0
        )

        region_entry = {
            "region_id": idx + 1,
            "pixel_count": pixel_count,
            "area_m2": area_m2,
            "area_ha": area_ha,
            "percentage_of_change": pct_of_total_change,
            "mean_probability": mean_prob,
            "bbox_px": [min_x, min_y, max_x, max_y],
            "centroid_px": [cx_px, cy_px],
            "centroid_crs": [round(cx_crs, 4), round(cy_crs, 4)],
        }
        extracted_regions.append(region_entry)

        # Corresponding geospatial GeoJSON geometry
        geo_geom = geo_shapes[idx][0] if idx < len(geo_shapes) else px_geom
        feature = {
            "type": "Feature",
            "id": idx + 1,
            "geometry": geo_geom,
            "properties": {
                "region_id": idx + 1,
                "pixel_count": pixel_count,
                "area_m2": area_m2,
                "area_ha": area_ha,
                "percentage_of_change": pct_of_total_change,
                "mean_probability": mean_prob,
                "centroid_crs": [round(cx_crs, 4), round(cy_crs, 4)],
            },
        }
        features.append(feature)

    # Sort descending by pixel count (largest area first)
    extracted_regions.sort(key=lambda r: r["pixel_count"], reverse=True)
    # Re-assign sequential rank IDs
    for rank, reg in enumerate(extracted_regions, start=1):
        reg["rank"] = rank

    # Sort GeoJSON features matching rank
    features.sort(key=lambda f: f["properties"]["pixel_count"], reverse=True)

    crs_name = meta.get("crs") or "urn:ogc:def:crs:OGC:1.3:CRS84"
    geojson_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name},
        },
        "features": features,
    }

    # Determine spatial concentration description
    if not extracted_regions:
        concentration = (
            f"All detected changes ({total_changed_pixels} px) were in small clusters "
            f"below the noise threshold ({min_region_pixels} px)."
        )
    else:
        # Calculate center of mass of significant regions
        total_sig_px = sum(r["pixel_count"] for r in extracted_regions)
        weighted_y = sum(r["centroid_px"][1] * r["pixel_count"] for r in extracted_regions) / total_sig_px
        weighted_x = sum(r["centroid_px"][0] * r["pixel_count"] for r in extracted_regions) / total_sig_px

        v_pos = "northern" if weighted_y < height / 3 else "southern" if weighted_y > 2 * height / 3 else "central"
        h_pos = "western" if weighted_x < width / 3 else "eastern" if weighted_x > 2 * width / 3 else ""

        sector = f"{v_pos}-{h_pos}".strip("-") if h_pos and h_pos != v_pos else v_pos
        top_region = extracted_regions[0]
        top_area_str = (
            f"{top_region['area_m2']:,.1f} m²"
            if top_region["area_m2"] is not None
            else f"{top_region['pixel_count']:,} pixels"
        )
        concentration = (
            f"Concentrated primarily in the {sector} sector across {len(extracted_regions)} "
            f"distinct region(s). The largest contiguous change region spans {top_area_str} "
            f"({top_region['percentage_of_change']}% of all detected change)."
        )

    return extracted_regions, geojson_collection, concentration


def render_labeled_region_overlay(
    rgb_image: np.ndarray,
    binary_mask: np.ndarray,
    regions: list[dict[str, Any]],
    max_labels: int = 20,
) -> Image.Image:
    """Generate an overlay showing T1 imagery, change highlighting, and labeled bounding boxes.

    Args:
        rgb_image: Raw visual RGB uint8 image of shape (H, W, 3).
        binary_mask: 2D uint8 mask (0 = unchanged, 1 = changed).
        regions: Sorted list of region dictionaries from extract_change_regions.
        max_labels: Maximum number of top regions to label with boxes and badges.

    Returns:
        PIL.Image containing the enhanced visual overlay.
    """
    h, w, _ = rgb_image.shape
    overlay = np.copy(rgb_image)

    # 1. Semi-transparent red highlight on changed pixels
    changed_bool = binary_mask == 1
    if np.any(changed_bool):
        alpha = 0.45
        overlay[changed_bool, 0] = np.clip(overlay[changed_bool, 0] * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)
        overlay[changed_bool, 1] = np.clip(overlay[changed_bool, 1] * (1 - alpha) + 30 * alpha, 0, 255).astype(np.uint8)
        overlay[changed_bool, 2] = np.clip(overlay[changed_bool, 2] * (1 - alpha) + 30 * alpha, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # 2. Draw bounding boxes and badges for top regions
    top_regions = regions[:max_labels]
    for r in top_regions:
        min_x, min_y, max_x, max_y = r["bbox_px"]
        rank = r.get("rank", r["region_id"])
        px_count = r["pixel_count"]

        # Crisp yellow/orange box with 2px width
        box_color = (255, 215, 0, 255)
        draw.rectangle([min_x, min_y, max_x, max_y], outline=box_color, width=2)

        # Label text badge
        label_text = f"#{rank} ({px_count}px)"
        # Use simple character width estimation for fallback default font
        badge_w = len(label_text) * 7 + 6
        badge_h = 14

        badge_x0 = min_x
        badge_y0 = max(0, min_y - badge_h - 2)
        badge_x1 = min(w - 1, badge_x0 + badge_w)
        badge_y1 = badge_y0 + badge_h

        # Dark translucent background for text readability
        draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=(15, 15, 20, 220))
        draw.text((badge_x0 + 3, badge_y0 + 1), label_text, fill=(255, 235, 120, 255))

    return pil_img
