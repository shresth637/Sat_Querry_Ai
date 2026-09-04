from typing import Optional

from satquery.domain.schemas import ConfidenceReport


def build_confidence_report(
    score: Optional[float] = None,
    source: str = "not_available",
    method: Optional[str] = None,
    signals_used: Optional[list[str]] = None,
    reason: Optional[str] = None,
) -> ConfidenceReport:
    """Construct an honest ConfidenceReport.

    Never invents or fabricates confidence scores.
    If score is None, sets display_text to 'Confidence not available'.
    """
    signals = signals_used or []

    if score is None:
        return ConfidenceReport(
            score=None,
            source=source if source != "model" else "not_available",
            method=method,
            is_available=False,
            display_text="Confidence not available",
            signals_used=signals,
            reason=reason or "No run-derived model or tool confidence score available",
        )

    # Clean valid numeric score between 0.0 and 1.0
    clamped_score = max(0.0, min(1.0, float(score)))
    return ConfidenceReport(
        score=round(clamped_score, 4),
        source=source,
        method=method,
        is_available=True,
        display_text=f"{clamped_score:.2%}",
        signals_used=signals,
        reason=reason,
    )
