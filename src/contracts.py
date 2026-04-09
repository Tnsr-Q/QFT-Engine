from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar


@dataclass(frozen=True)
class FakeonCertificate:
    """Typed contract for fakeon virtualization certificate payloads."""

    Re_alpha_at_M2: float
    fakeon_virtualized: bool
    trajectory: Any
    t_grid: Any
    status: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


SchemaT = TypeVar("SchemaT")


def enforce_schema(output_schema: type[SchemaT]) -> Callable[[Callable[..., Any]], Callable[..., SchemaT]]:
    """Validate function output by constructing the target dataclass/schema type."""

    def decorator(func: Callable[..., Any]) -> Callable[..., SchemaT]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> SchemaT:
            raw_result = func(*args, **kwargs)
            if isinstance(raw_result, output_schema):
                return raw_result
            if isinstance(raw_result, dict):
                return output_schema(**raw_result)
            raise TypeError(f"{func.__name__} must return {output_schema.__name__} or dict")

        return wrapper

    return decorator
