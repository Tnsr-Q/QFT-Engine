from __future__ import annotations

import json
import os
import tempfile
from typing import Any


class OrbaxAtomicStateIO:
    """Atomic state persistence with Orbax when available.

    Falls back to atomic temp-file rename when Orbax is unavailable.
    """

    def __init__(self, path: str):
        self.path = path

    def save(self, state: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            import orbax.checkpoint as ocp  # type: ignore

            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(self.path, state, force=True)
            return
        except Exception:
            pass

        fd, tmp_path = tempfile.mkstemp(prefix="tmp_state_", suffix=".json", dir=os.path.dirname(self.path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def restore(self) -> dict[str, Any]:
        try:
            import orbax.checkpoint as ocp  # type: ignore

            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(self.path)
            if isinstance(restored, dict):
                return restored
        except Exception:
            pass

        with open(self.path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("State payload must be a dict")
        return payload
