"""Expose model helpers lazily so HF inference does not import training datasets."""

__all__ = ["available_model_names", "available_models", "get_model_description", "load"]


def __getattr__(name):
    if name in __all__:
        from . import models

        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
