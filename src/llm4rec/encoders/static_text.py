"""Static text encoder interface placeholder."""

from __future__ import annotations

from typing import Any


class StaticTextEvidenceEncoder:
    """Deterministic metadata vectorizer for smoke diagnostics, not a trained model."""

    reportable = False

    def encode_item_text(self, item: dict[str, Any]) -> list[float]:
        text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "description", "category", "raw_text")
        )
        tokens = {token.lower() for token in text.split() if token}
        return [float(len(tokens)), float(sum(len(token) for token in tokens))]
