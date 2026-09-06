import gc
import logging
from typing import Any, Optional
import torch

from satquery.models.base import ModelAdapter

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages model lifecycles, memory governance, and device placement.

    Ensures:
    1. CUDA is used when available, with deterministic fallback to CPU.
    2. VRAM constraints (e.g., 6 GB RTX 4050) are respected.
    3. Lazy loading of model weights on-demand.
    4. Only one large model (e.g. 7B VLM or heavy detector) is resident at a time.
    5. Immediate garbage collection and CUDA cache flushing on model release.
    """

    def __init__(self, max_vram_gb: float = 5.5) -> None:
        self.max_vram_gb = max_vram_gb
        self._resident_models: dict[str, ModelAdapter] = {}
        self._resident_large_model_id: Optional[str] = None

    @staticmethod
    def has_cuda() -> bool:
        """Return True if a CUDA-capable device is available to PyTorch."""
        return torch.cuda.is_available()

    def get_device(self, preferred: str = "auto") -> torch.device:
        """Resolve the target execution device safely."""
        pref_lower = preferred.lower() if preferred else "auto"
        if pref_lower == "cpu":
            return torch.device("cpu")
        if pref_lower in ["cuda", "auto"]:
            if self.has_cuda():
                return torch.device("cuda:0")
            return torch.device("cpu")
        try:
            return torch.device(preferred)
        except Exception:
            return torch.device("cpu")

    def get_vram_info(self) -> dict[str, Any]:
        """Return VRAM statistics in MiB if CUDA is available."""
        if not self.has_cuda():
            return {
                "cuda_available": False,
                "device_name": "CPU",
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
            }
        try:
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            total = props.total_memory / (1024 * 1024)
            return {
                "cuda_available": True,
                "device_name": props.name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(total, 2),
            }
        except Exception as e:
            logger.warning(f"Failed to query VRAM: {e}")
            return {"cuda_available": True, "error": str(e)}

    def acquire_model(
        self,
        model_id: str,
        adapter: ModelAdapter,
        is_large: bool = False,
    ) -> ModelAdapter:
        """Acquire and prepare a model for inference.

        If the model is marked as 'large', any previously resident large model
        is unloaded first to protect the VRAM budget.
        """
        if is_large and self._resident_large_model_id and self._resident_large_model_id != model_id:
            logger.info(
                f"Unloading prior resident large model '{self._resident_large_model_id}' "
                f"before loading '{model_id}'."
            )
            self.release_model(self._resident_large_model_id)

        if model_id in self._resident_models and self._resident_models[model_id] is not adapter:
            self.release_model(model_id)

        if model_id not in self._resident_models:
            if adapter.status != "ready":
                adapter.load()
            self._resident_models[model_id] = adapter
            if is_large:
                self._resident_large_model_id = model_id

        return self._resident_models[model_id]

    def release_model(self, model_id: str) -> None:
        """Unload a model from memory and trigger cache cleanup."""
        if model_id in self._resident_models:
            adapter = self._resident_models.pop(model_id)
            try:
                adapter.unload()
            except Exception as e:
                logger.warning(f"Error unloading model {model_id}: {e}")

            if self._resident_large_model_id == model_id:
                self._resident_large_model_id = None

            self.cleanup_memory()

    def release_all(self) -> None:
        """Unload all resident models and reset resource state."""
        for mid in list(self._resident_models.keys()):
            self.release_model(mid)
        self.cleanup_memory()

    def cleanup_memory(self) -> None:
        """Force garbage collection and flush CUDA cache."""
        gc.collect()
        if self.has_cuda():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug(f"CUDA empty cache notice: {e}")


# Default global instance
_default_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Retrieve or create the global singleton ResourceManager."""
    global _default_resource_manager
    if _default_resource_manager is None:
        _default_resource_manager = ResourceManager()
    return _default_resource_manager
