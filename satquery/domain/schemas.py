from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    MULTISPECTRAL = "MULTISPECTRAL"
    SAR = "SAR"
    UNKNOWN = "UNKNOWN"


class QualityLevel(str, Enum):
    GOOD = "GOOD"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class InputMode(str, Enum):
    I1_SINGLE_OPTICAL = "I1_SINGLE_OPTICAL"
    I2_SINGLE_SAR = "I2_SINGLE_SAR"
    I3_OPTICAL_SAR_PAIR = "I3_OPTICAL_SAR_PAIR"
    I4_BITEMPORAL_PAIR = "I4_BITEMPORAL_PAIR"


class SpatialCompatibility(str, Enum):
    CONFIRMED = "CONFIRMED"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class EvidenceType(str, Enum):
    SOURCE_IMAGE = "source_image"
    PREVIEW = "preview"
    MASK = "mask"
    BOUNDING_BOX = "bounding_box"
    CHANGE_MAP = "change_map"
    STATISTICS = "statistics"
    METADATA = "metadata"
    CLASSIFICATION = "classification"


class TraceStatus(str, Enum):
    START = "start"
    SUCCESS = "success"
    WARN = "warn"
    ERROR = "error"


class TaskType(str, Enum):
    SINGLE_VQA = "SINGLE_VQA"
    SINGLE_CAPTION = "SINGLE_CAPTION"
    SINGLE_GROUNDING = "SINGLE_GROUNDING"
    CLASSIFICATION = "CLASSIFICATION"
    BI_TEMPORAL_CHANGE = "BI_TEMPORAL_CHANGE"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"


class ModelCapability(str, Enum):
    CHANGE_DETECTION = "change_detection"
    IMAGE_VQA = "image_vqa"
    IMAGE_CAPTION = "image_caption"
    GROUNDING = "grounding"
    CLASSIFICATION = "classification"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class GroundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = 1.0
    box_2d: list[float]  # [ymin, xmin, ymax, xmax] in pixel coordinates
    geo_bbox: Optional[list[float]] = None  # [minx, miny, maxx, maxy] in CRS
    geometry: Optional[dict[str, Any]] = None  # GeoJSON polygon


class LandCoverPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str
    probability: float
    threshold: float = 0.3
    is_detected: bool = True


class QueryTheme(str, Enum):
    AGRICULTURE = "AGRICULTURE"
    DISASTER = "DISASTER"
    URBAN = "URBAN"
    ENVIRONMENT = "ENVIRONMENT"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    GENERAL = "GENERAL"


class RasterMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    dtype: Optional[str] = None
    crs: Optional[str] = None
    transform: Optional[list[float]] = None
    bounds: Optional[list[float]] = None
    resolution: Optional[list[float]] = None
    nodata: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_valid: bool = True
    error_message: Optional[str] = None


class ImageQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: QualityLevel
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SlotAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    file_path: str
    declared_modality: Optional[Modality] = None
    declared_timestamp: Optional[str] = None


class ValidationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    status: ValidationStatus
    message: str
    is_blocking: bool = False


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus
    checks: list[ValidationCheck] = Field(default_factory=list)
    spatial_compatibility: Optional[SpatialCompatibility] = None
    details: dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Optional[str] = None
    capability_tags: list[str] = Field(default_factory=list)
    entity_phrases: list[str] = Field(default_factory=list)
    raw_query: str = ""
    target_modality: Optional[Modality] = None


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    required_inputs: list[str] = Field(default_factory=list)
    input_mode: InputMode
    selected_models: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    permitted_parameters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    theme: QueryTheme = QueryTheme.GENERAL


class ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_id: str
    capability: str
    status: str
    text: Optional[str] = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    raw_scores: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ConfidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: Optional[float] = None
    source: str = "not_available"
    method: Optional[str] = None
    is_available: bool = False
    display_text: str = "Confidence not available"
    signals_used: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_type: EvidenceType
    title: str
    description: str = ""
    file_path: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    crs: Optional[str] = None


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    component: str
    status: TraceStatus
    timestamp: str
    duration_ms: Optional[float] = None
    parameters: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    plan: AgentPlan
    result_text: str
    confidence: ConfidenceReport
    evidence: list[Evidence] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    validation: ValidationReport
    metas: dict[str, RasterMeta] = Field(default_factory=dict)
    qualities: dict[str, ImageQuality] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
