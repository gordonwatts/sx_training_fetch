"""Public package API with lazy imports.

Keeping imports lazy here avoids importing the heavy ServiceX/query stack during CLI
startup (for example ``calratio_training_data --help``), while preserving the
existing public symbols.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "RunConfig",
    "build_preselection",
    "fetch_training_data",
    "fetch_training_data_to_file",
    "run_query",
]

if TYPE_CHECKING:
    from .training_query import (
        RunConfig,
        build_preselection,
        fetch_training_data,
        fetch_training_data_to_file,
        run_query,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose selected symbols from ``training_query`` on first access."""
    if name in __all__:
        training_query_module = import_module(".training_query", __name__)
        return getattr(training_query_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
