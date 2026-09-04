import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio

from satquery.domain.schemas import GroundingBox

logger = logging.getLogger(__name__)


def normalize_box_to_pixels(
    box: list[float],
    img_width: int,
    img_height: int,
    source_scale: float = 1000.0,
) -> tuple[int, int, int, int]:
    """Convert normalized [ymin, xmin, ymax, xmax] (e.g., in [0, 1000]) to pixel [xmin, ymin, xmax, ymax]."""
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinate values, received {len(box)}")

    ymin, xmin, ymax, xmax = box
    px_xmin = int(np.clip((xmin / source_scale) * img_width, 0, img_width))
    px_ymin = int(np.clip((ymin / source_scale) * img_height, 0, img_height))
    px_xmax = int(np.clip((xmax / source_scale) * img_width, 0, img_width))
    px_ymax = int(np.clip((ymax / source_scale) * img_height, 0, img_height))

    # Ensure min <= max
    if px_xmin > px_xmax:
        px_xmin, px_xmax = px_xmax, px_xmin
    if px_ymin > px_ymax:
        px_ymin, px_ymax = px_ymax, px_ymin

    return px_xmin, px_ymin, px_xmax, px_ymax


def pixel_box_to_geo_polygon(
    pixel_box: tuple[int, int, int, int],
    transform: Any,
) -> tuple[list[list[list[float]]], list[float]]:
    """Convert pixel [xmin, ymin, xmax, ymax] into CRS coordinates and a GeoJSON Polygon ring.

    Returns:
        coordinates: list of polygon rings [[[x, y], ...]]
        geo_bbox: [minx, miny, maxx, maxy] in CRS
    """
    px_minx, px_miny, px_maxx, px_maxy = pixel_box

    # Map the 4 corners: (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
    p1 = rasterio.transform.xy(transform, px_miny, px_minx, offset="ul")
    p2 = rasterio.transform.xy(transform, px_miny, px_maxx, offset="ur")
    p3 = rasterio.transform.xy(transform, px_maxy, px_maxx, offset="lr")
    p4 = rasterio.transform.xy(transform, px_maxy, px_minx, offset="ll")

    ring = [
        [float(p1[0]), float(p1[1])],
        [float(p2[0]), float(p2[1])],
        [float(p3[0]), float(p3[1])],
        [float(p4[0]), float(p4[1])],
        [float(p1[0]), float(p1[1])],  # Close ring
    ]

    xs = [p[0] for p in ring[:-1]]
    ys = [p[1] for p in ring[:-1]]
    geo_bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    return [ring], geo_bbox


def export_grounding_geojson(
    boxes: list[dict[str, Any]],
    output_path: Path,
    crs: Optional[str] = None,
) -> dict[str, Any]:
    """Export standardized grounding boxes into an RFC 7946 compliant GeoJSON FeatureCollection."""
    features = []
    for idx, b in enumerate(boxes, start=1):
        props = {
            "id": idx,
            "label": b.get("label", "detected_object"),
            "confidence": float(b.get("confidence", 1.0)),
            "pixel_box": b.get("pixel_box"),
        }
        if "geo_bbox" in b:
            props["geo_bbox"] = b["geo_bbox"]

        geometry = b.get("geometry")
        if not geometry and "geo_bbox" in b:
            minx, miny, maxx, maxy = b["geo_bbox"]
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny],
                ]],
            }

        features.append({
            "type": "Feature",
            "id": idx,
            "properties": props,
            "geometry": geometry or {"type": "Polygon", "coordinates": []},
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "features": features,
    }

    if crs:
        geojson_dict["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.replace('EPSG:', '')}" if crs.startswith("EPSG:") else crs},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)

    return geojson_dict


def render_grounding_overlay(
    image_path: Union[str, Path],
    boxes: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render high-contrast labeled bounding boxes on the raster image."""
    with rasterio.open(str(image_path)) as src:
        if src.count >= 3:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        else:
            b = src.read(1).astype(np.float32)
            rgb = np.stack([b, b, b], axis=0)

    # Normalize to uint8 RGB
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    base_img = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()
    box_color = (255, 69, 0, 255)      # Orange-red border
    fill_color = (255, 69, 0, 45)       # Semi-transparent fill
    badge_bg = (220, 20, 60, 230)      # Crimson badge
    text_color = (255, 255, 255, 255)

    for idx, b in enumerate(boxes, start=1):
        px_box = b.get("pixel_box")
        if not px_box or len(px_box) != 4:
            continue
        xmin, ymin, xmax, ymax = px_box

        # Semi-transparent box fill
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color, outline=box_color, width=3)

        label = b.get("label", f"Object #{idx}")
        conf = b.get("confidence")
        badge_text = f"#{idx} {label} ({conf:.2f})" if conf is not None else f"#{idx} {label}"

        # Draw badge label
        bbox = draw.textbbox((xmin, max(0, ymin - 16)), badge_text, font=font)
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=badge_bg)
        draw.text((bbox[0], bbox[1]), badge_text, fill=text_color, font=font)

    result = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    result.convert("RGB").save(str(output_path), "PNG")
    return output_path
