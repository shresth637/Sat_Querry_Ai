import importlib
from pathlib import Path
from typing import Any, Optional, Union
import yaml

from satquery.models.base import ModelAdapter


class ModelRegistry:
    """Registry for discovering, listing, and resolving specialist models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelAdapter] = {}
        self._enabled: dict[str, bool] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        model_id: str,
        adapter: ModelAdapter,
        enabled: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._models[model_id] = adapter
        self._enabled[model_id] = enabled
        self._metadata[model_id] = metadata or {}

    def get(self, model_id: str) -> Optional[ModelAdapter]:
        return self._models.get(model_id)

    def is_enabled(self, model_id: str) -> bool:
        return self._enabled.get(model_id, False)

    def list_models(self, enabled_only: bool = False) -> list[tuple[str, ModelAdapter]]:
        items = []
        for mid, adapter in self._models.items():
            if enabled_only and not self._enabled.get(mid, False):
                continue
            items.append((mid, adapter))
        return items

    def list_by_capability(
        self, capability: str, enabled_only: bool = False
    ) -> list[ModelAdapter]:
        matching = []
        for mid, adapter in self._models.items():
            if enabled_only and not self._enabled.get(mid, False):
                continue
            if capability in adapter.capabilities:
                matching.append(adapter)
        return matching

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelRegistry":
        """Build a registry from a YAML specification without loading large weights."""
        registry = cls()
        config_path = Path(path)
        if not config_path.exists():
            return registry

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for entry in data.get("models", []):
            mid = entry.get("id")
            adapter_path = entry.get("adapter")
            enabled = bool(entry.get("enabled", False))

            if not mid or not adapter_path:
                continue

            try:
                module_name, class_name = adapter_path.rsplit(".", 1)
                mod = importlib.import_module(module_name)
                adapter_cls = getattr(mod, class_name)
                init_kwargs = {}
                weights_path = entry.get("weights_path")
                if weights_path:
                    init_kwargs["weights_path"] = weights_path
                for param in ["tile_size", "tile_overlap", "change_threshold", "batch_size", "device"]:
                    if param in entry:
                        init_kwargs[param] = entry[param]

                try:
                    adapter_inst = adapter_cls(**init_kwargs)
                except TypeError:
                    try:
                        adapter_inst = adapter_cls(weights_path=weights_path) if weights_path else adapter_cls()
                    except TypeError:
                        adapter_inst = adapter_cls()

                registry.register(model_id=mid, adapter=adapter_inst, enabled=enabled, metadata=entry)
            except Exception:
                # Log or keep record of failure without crashing
                pass

        return registry
