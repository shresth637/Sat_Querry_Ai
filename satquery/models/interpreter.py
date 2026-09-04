from typing import Any, Optional


def format_change_detection_summary(
    stats: dict[str, Any],
    regions: list[dict[str, Any]],
    area_info: dict[str, Any],
    crs_str: Optional[str] = None,
    device_str: str = "cpu",
    artifacts_names: Optional[list[str]] = None,
) -> str:
    """Format a rigorous, factual natural-language interpretation of bi-temporal change detection results.

    Follows strict engineering standards:
    - Reports exact metrics (percentages, pixel counts, physical area, confidence).
    - Describes spatial distribution and concentration without fabricating semantic objects.
    - Explicitly details LEVIR-CD pretraining limitations.
    """
    total_px = stats.get("total_pixels", 0)
    changed_px = stats.get("changed_pixels", 0)
    unchanged_px = stats.get("unchanged_pixels", total_px - changed_px)
    pct_changed = stats.get("percentage_changed", 0.0)
    pct_unchanged = stats.get("percentage_unchanged", 100.0 - pct_changed)

    mean_conf = stats.get("mean_prediction_confidence", 0.0)
    changed_conf = stats.get("changed_pixel_confidence")
    unchanged_conf = stats.get("unchanged_pixel_confidence")
    low_conf_pct = stats.get("low_confidence_pixel_percentage", 0.0)
    threshold = stats.get("change_threshold", 0.5)

    num_regions = len(regions)
    top_region = regions[0] if regions else None

    # Physical area text
    area_m2 = area_info.get("changed_area_m2")
    area_ha = area_info.get("changed_area_hectares")
    area_km2 = area_info.get("changed_area_km2")
    area_method = area_info.get("area_calculation_method", "unavailable")

    if area_m2 is not None:
        area_summary_str = f"{area_m2:,.1f} m² ({area_ha:,.2f} hectares / {area_km2:.4f} km²)"
    else:
        area_summary_str = "Physical area calculation unavailable (unprojected or missing spatial reference)."

    # Spatial distribution text
    if num_regions == 0:
        spatial_dist_str = (
            "No significant contiguous change clusters were identified above the noise threshold."
        )
    else:
        largest_area_str = (
            f"{top_region['area_m2']:,.1f} m²"
            if top_region.get("area_m2") is not None
            else f"{top_region['pixel_count']:,} pixels"
        )
        spatial_dist_str = (
            f"Detected changes are clustered into {num_regions} significant contiguous region(s). "
            f"The largest contiguous change cluster spans {largest_area_str} "
            f"({top_region['percentage_of_change']}% of all detected change)."
        )

    # Artifacts listing
    art_list = artifacts_names or [
        "change_mask.tif (Binary GeoTIFF)",
        "change_prob.tif (Probability GeoTIFF)",
        "change_regions.geojson (Vector Regions)",
        "change_regions_vis.png (Labeled Region Overlay)",
        "change_overlay.png (Visual Overlay)",
        "stats.json (Detailed Statistics)",
    ]
    artifacts_str = "\n".join(f"- `{name}`" for name in art_list)

    # Detailed report sections
    report = (
        f"### SUMMARY\n"
        f"Bi-temporal change detection completed via Open-CD BIT on {device_str.upper()}.\n"
        f"Approximately **{pct_changed}%** of the analyzed satellite scene was classified as changed "
        f"at a probability threshold of {threshold:.2f}.\n\n"
        f"### WHAT CHANGED\n"
        f"The neural model identified spatial and spectral differences between Date T0 and Date T1. "
        f"In total, **{changed_px:,} pixels** exhibited significant state transitions exceeding the detection threshold, "
        f"while **{unchanged_px:,} pixels** ({pct_unchanged}%) remained stable.\n\n"
        f"### HOW MUCH CHANGED\n"
        f"- **Changed Area:** {area_summary_str}\n"
        f"- **Total Analyzed Pixels:** {total_px:,}\n"
        f"- **Changed Pixel Count:** {changed_px:,} ({pct_changed}%)\n"
        f"- **Unchanged Pixel Count:** {unchanged_px:,} ({pct_unchanged}%)\n"
        f"- **Area Calculation Method:** `{area_method}`\n\n"
        f"### SPATIAL DISTRIBUTION\n"
        f"{spatial_dist_str}\n\n"
        f"### CONFIDENCE\n"
        f"- **Mean Prediction Confidence:** {mean_conf * 100:.1f}%\n"
        f"- **Changed-Pixel Confidence:** {f'{changed_conf * 100:.1f}%' if changed_conf is not None else 'N/A'}\n"
        f"- **Unchanged-Pixel Confidence:** {f'{unchanged_conf * 100:.1f}%' if unchanged_conf is not None else 'N/A'}\n"
        f"- **Low-Confidence Pixels (<60%):** {low_conf_pct}%\n\n"
        f"### GEOSPATIAL INFORMATION\n"
        f"- **Coordinate Reference System:** `{crs_str or 'Not available'}`\n"
        f"- **Spatial Dimensions:** {stats.get('raster_dimensions', {}).get('width')} × {stats.get('raster_dimensions', {}).get('height')}\n"
        f"- **Pixel Resolution:** {stats.get('resolution')}\n\n"
        f"### ARTIFACTS\n"
        f"{artifacts_str}\n\n"
        f"### LIMITATIONS\n"
        f"The BIT model was pretrained on the LEVIR-CD benchmark, which is optimized for optical building change detection. "
        f"While the model reliably detects optical differencing, it does not assign high-level semantic categories (e.g., 'road', 'forest'). "
        f"Performance on radar (SAR) or multi-modal imagery should be interpreted with caution."
    )

    return report
