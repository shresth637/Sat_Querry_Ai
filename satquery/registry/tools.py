from pathlib import Path
from typing import Optional, Union
import yaml

from satquery.tools.base import (
    ChangeMapTool,
    ImageQualityTool,
    MetadataExtractionTool,
    RasterInspectionTool,
    ReportGenerationTool,
    SpatialStatisticsTool,
    Tool,
    ValidationTool,
    VisualizationTool,
)


class ToolRegistry:
    """Registry for discovering, listing, and resolving deterministic tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, tool: Tool, enabled: bool = True) -> None:
        self._tools[tool.tool_id] = tool
        self._enabled[tool.tool_id] = enabled

    def get(self, tool_id: str) -> Optional[Tool]:
        return self._tools.get(tool_id)

    def is_enabled(self, tool_id: str) -> bool:
        return self._enabled.get(tool_id, False)

    def list_tools(self, enabled_only: bool = False) -> list[tuple[str, Tool]]:
        items = []
        for tid, tool in self._tools.items():
            if enabled_only and not self._enabled.get(tid, False):
                continue
            items.append((tid, tool))
        return items

    @classmethod
    def default(cls) -> "ToolRegistry":
        """Construct registry with the standard 8 tool interfaces."""
        reg = cls()
        reg.register(RasterInspectionTool(), enabled=True)
        reg.register(MetadataExtractionTool(), enabled=True)
        reg.register(ImageQualityTool(), enabled=True)
        reg.register(ValidationTool(), enabled=True)
        reg.register(VisualizationTool(), enabled=True)
        reg.register(ChangeMapTool(), enabled=True)
        reg.register(SpatialStatisticsTool(), enabled=True)
        reg.register(ReportGenerationTool(), enabled=True)
        return reg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ToolRegistry":
        reg = cls.default()
        config_path = Path(path)
        if not config_path.exists():
            return reg

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for entry in data.get("tools", []):
            tid = entry.get("id")
            enabled = bool(entry.get("enabled", True))
            if tid in reg._tools:
                reg._enabled[tid] = enabled

        return reg
