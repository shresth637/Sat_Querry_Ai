from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from satquery.domain.schemas import (
    ImageQuality,
    InputMode,
    RasterMeta,
    SlotAssignment,
    ValidationReport,
)
from satquery.preprocess.raster import (
    assess_image_quality,
    inspect_raster,
)
from satquery.validation.validator import validate_input


class Tool(ABC):
    """Abstract base class for all deterministic and analytical tools."""

    def __init__(self, tool_id: str, name: str, description: str):
        self.tool_id = tool_id
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool deterministically."""
        pass


class RasterInspectionTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="raster_inspection",
            name="Raster Inspection Tool",
            description="Inspects GeoTIFF/TIFF rasters and extracts metadata via rasterio.",
        )

    def execute(self, path: Union[str, Path]) -> RasterMeta:
        return inspect_raster(path)


class MetadataExtractionTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="metadata_extraction",
            name="Metadata Extraction Tool",
            description="Extracts structured geospatial and raster tags from RasterMeta.",
        )

    def execute(self, meta: RasterMeta) -> dict[str, Any]:
        return {
            "filename": meta.filename,
            "format": meta.format,
            "dimensions": f"{meta.width} x {meta.height}",
            "bands": meta.band_count,
            "dtype": meta.dtype,
            "crs": meta.crs or "None (unprojected)",
            "bounds": meta.bounds,
            "resolution": meta.resolution,
            "nodata": meta.nodata,
            "tags": meta.metadata,
        }


class ImageQualityTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="image_quality",
            name="Image Quality Tool",
            description="Performs deterministic raster quality, valid-pixel, and uniformity checks.",
        )

    def execute(self, path: Union[str, Path], meta: Optional[RasterMeta] = None) -> ImageQuality:
        return assess_image_quality(path, meta)


class ValidationTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="validation",
            name="Input Validation Tool",
            description="Validates session slots, file counts, and spatial co-registration for I1-I4.",
        )

    def execute(
        self,
        slots: list[SlotAssignment],
        metas: dict[str, RasterMeta],
        input_mode: InputMode,
        allow_png_jpeg_benchmark: bool = False,
    ) -> ValidationReport:
        return validate_input(slots, metas, input_mode, allow_png_jpeg_benchmark)


class VisualizationTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="visualization",
            name="Visualization Tool",
            description="Prepares normalized raster arrays for display without inventing data.",
        )

    def execute(self, array: np.ndarray, percentile_min: float = 2.0, percentile_max: float = 98.0) -> np.ndarray:
        if array.size == 0:
            return array
        valid = array[np.isfinite(array)]
        if valid.size == 0:
            return np.zeros_like(array, dtype=np.uint8)
        p_low = np.percentile(valid, percentile_min)
        p_high = np.percentile(valid, percentile_max)
        if p_high <= p_low:
            return np.zeros_like(array, dtype=np.uint8)
        stretched = np.clip((array - p_low) / (p_high - p_low), 0.0, 1.0)
        return (stretched * 255.0).astype(np.uint8)


class ChangeMapTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="change_map",
            name="Change Map Tool",
            description="Deterministic change-difference and threshold computation between aligned rasters.",
        )

    def execute(self, t0: np.ndarray, t1: np.ndarray, threshold: float = 0.25) -> dict[str, Any]:
        if t0.shape != t1.shape:
            raise ValueError(f"Shape mismatch for change map: {t0.shape} vs {t1.shape}")
        diff = np.abs(t1.astype(np.float32) - t0.astype(np.float32))
        change_mask = diff > threshold
        return {
            "diff": diff,
            "mask": change_mask,
            "change_ratio": float(np.mean(change_mask)),
        }


class SpatialStatisticsTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="spatial_statistics",
            name="Spatial Statistics Tool",
            description="Calculates summary statistics across valid raster pixels.",
        )

    def execute(self, array: np.ndarray, nodata: Optional[float] = None) -> dict[str, Any]:
        valid = array[np.isfinite(array)]
        if nodata is not None:
            valid = valid[valid != nodata]
        if valid.size == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": int(valid.size),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }


class ReportGenerationTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            tool_id="report_generation",
            name="Report Generation Tool",
            description="Packages session query, metadata, validation, answers, and trace into exportable structures.",
        )

    def execute(self, session_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_id": session_data.get("session_id", "report"),
            "query": session_data.get("query", ""),
            "status": "packaged",
            "sections": list(session_data.keys()),
        }
