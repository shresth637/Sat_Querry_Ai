from typing import Any, Optional

from satquery.domain.schemas import Evidence, EvidenceType, RasterMeta


def create_source_image_evidence(file_path: str, title: str = "Source Image", crs: Optional[str] = None) -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.SOURCE_IMAGE,
        title=title,
        description="Original uploaded raster scene",
        file_path=file_path,
        crs=crs,
    )


def create_preview_evidence(file_path: str, title: str = "Preview", description: str = "") -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.PREVIEW,
        title=title,
        description=description or "RGB visual preview derived from raw raster bands",
        file_path=file_path,
    )


def create_mask_evidence(
    title: str,
    file_path: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    crs: Optional[str] = None,
    description: str = "",
) -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.MASK,
        title=title,
        description=description or "Spatial binary/categorical segmentation mask",
        file_path=file_path,
        data=data,
        crs=crs,
    )


def create_bounding_box_evidence(
    boxes: list[dict[str, Any]],
    title: str = "Grounding Bounding Boxes",
    crs: Optional[str] = None,
) -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.BOUNDING_BOX,
        title=title,
        description=f"{len(boxes)} grounded bounding box regions",
        data={"boxes": boxes},
        crs=crs,
    )


def create_change_map_evidence(
    title: str,
    file_path: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    crs: Optional[str] = None,
    description: str = "",
) -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.CHANGE_MAP,
        title=title,
        description=description or "Bi-temporal change detection raster map",
        file_path=file_path,
        data=data,
        crs=crs,
    )


def create_statistics_evidence(stats: dict[str, Any], title: str = "Spatial Statistics") -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.STATISTICS,
        title=title,
        description="Summary statistics computed across valid raster cells",
        data=stats,
    )


def create_metadata_evidence(meta: RasterMeta, title: str = "Raster Metadata") -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.METADATA,
        title=title,
        description="Real extracted raster dimensions, format, and CRS tags",
        metadata={
            "filename": meta.filename,
            "format": meta.format,
            "width": meta.width,
            "height": meta.height,
            "band_count": meta.band_count,
            "crs": meta.crs,
            "dtype": meta.dtype,
            "transform": meta.transform,
            "resolution": meta.resolution,
        },
        crs=meta.crs,
    )
