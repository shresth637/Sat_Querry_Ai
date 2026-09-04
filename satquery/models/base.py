from abc import ABC, abstractmethod
from typing import Any, Optional
from satquery.domain.schemas import ModelResult


class ModelAdapter(ABC):
    """Abstract base class for all specialist remote-sensing models."""

    def __init__(
        self,
        name: str,
        version: str,
        capabilities: list[str],
        status: str = "unloaded",
    ):
        self.name = name
        self.version = version
        self.capabilities = capabilities
        self.status = status

    @abstractmethod
    def load(self) -> None:
        """Load weights or prepare model for inference."""
        pass

    @abstractmethod
    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        """Execute inference and return typed ModelResult."""
        pass

    def is_available(self) -> bool:
        return self.status == "ready"
