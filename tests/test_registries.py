from pathlib import Path
import numpy as np

from satquery.models.base import ModelAdapter
from satquery.models.unconfigured import UnconfiguredVQAModel
from satquery.registry.models import ModelRegistry
from satquery.registry.tools import ToolRegistry
from satquery.tools.base import (
    ChangeMapTool,
    ImageQualityTool,
    MetadataExtractionTool,
    RasterInspectionTool,
    ReportGenerationTool,
    SpatialStatisticsTool,
    ValidationTool,
    VisualizationTool,
)


def test_13_model_registry():
    registry = ModelRegistry.from_yaml("config/models.yaml")
    all_models = registry.list_models()
    assert len(all_models) == 6

    # Verify GeoChat VQA adapter from YAML
    geochat = registry.get("geochat_vqa")
    assert geochat is not None
    assert "vqa" in geochat.capabilities
    # Enabled should be False in default config
    assert registry.is_enabled("geochat_vqa") is False
    assert geochat.status == "not_configured"

    # Prediction on unconfigured model should cleanly report status='not_configured'
    res = geochat.predict({"image": "dummy"})
    assert res.status == "not_configured"
    assert res.text is None

    # Custom registration
    class CustomMockAdapter(ModelAdapter):
        def __init__(self):
            super().__init__(name="mock_vqa", version="v1", capabilities=["vqa"], status="ready")

        def load(self):
            pass

        def predict(self, inputs, **kwargs):
            from satquery.domain.schemas import ModelResult
            return ModelResult(registry_id=self.name, capability="vqa", status="success", text="Land cover: Forest")

    custom = CustomMockAdapter()
    registry.register("custom_mock", custom, enabled=True)
    assert registry.get("custom_mock") is not None
    assert registry.is_enabled("custom_mock") is True
    res_custom = registry.get("custom_mock").predict({})
    assert res_custom.status == "success"
    assert res_custom.text == "Land cover: Forest"


def test_14_tool_registry(optical_geotiff: Path):
    reg = ToolRegistry.default()
    tools = reg.list_tools()
    assert len(tools) == 8

    # Verify presence of all required 8 interfaces
    assert reg.get("raster_inspection") is not None
    assert reg.get("metadata_extraction") is not None
    assert reg.get("image_quality") is not None
    assert reg.get("validation") is not None
    assert reg.get("visualization") is not None
    assert reg.get("change_map") is not None
    assert reg.get("spatial_statistics") is not None
    assert reg.get("report_generation") is not None

    # Test tool execution
    inspect_tool = reg.get("raster_inspection")
    meta = inspect_tool.execute(optical_geotiff)
    assert meta.is_valid is True

    # Spatial statistics tool execution
    stats_tool = reg.get("spatial_statistics")
    test_array = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = stats_tool.execute(test_array)
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0

    # Visualization tool execution
    viz_tool = reg.get("visualization")
    stretched = viz_tool.execute(test_array)
    assert stretched.dtype == np.uint8
    assert stretched.shape == test_array.shape
