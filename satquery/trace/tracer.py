from contextlib import contextmanager
from datetime import datetime, timezone
import time
from typing import Any, Generator, Optional

from satquery.domain.schemas import TraceEvent, TraceStatus


class Tracer:
    """Records observable technical execution events.

    Does not implement or expose internal chain-of-thought.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record_event(
        self,
        step: str,
        component: str,
        status: TraceStatus,
        duration_ms: Optional[float] = None,
        parameters: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            step=step,
            component=component,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            parameters=parameters,
            error=error,
            details=details,
        )
        self.events.append(event)
        return event

    @contextmanager
    def span(
        self,
        step: str,
        component: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Generator[TraceEvent, None, None]:
        """Measure execution duration and record start and completion events."""
        start_time = time.perf_counter()
        start_event = self.record_event(
            step=step,
            component=component,
            status=TraceStatus.START,
            parameters=parameters,
        )
        try:
            yield start_event
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.record_event(
                step=step,
                component=component,
                status=TraceStatus.SUCCESS,
                duration_ms=round(elapsed_ms, 2),
                parameters=parameters,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.record_event(
                step=step,
                component=component,
                status=TraceStatus.ERROR,
                duration_ms=round(elapsed_ms, 2),
                parameters=parameters,
                error=str(exc),
            )
            raise

    def get_events(self) -> list[TraceEvent]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()
