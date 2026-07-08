"""Job Scout package."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["agent", "root_agent"]


def __getattr__(name: str) -> Any:
    if name == "agent":
        return import_module(".agent", __name__)
    if name == "root_agent":
        return import_module(".agent", __name__).root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

