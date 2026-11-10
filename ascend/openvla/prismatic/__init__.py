"""Prismatic public API.

The robot inference path imports configuration/model classes directly from the
package.  Keep the package import lightweight so optional RLDS dependencies
(``dlimp``, TensorFlow datasets, ...) are only needed by training/data APIs.
"""

from importlib import import_module


def __getattr__(name):
    if name in {"available_model_names", "available_models", "get_model_description", "load"}:
        return getattr(import_module(".models", __name__), name)
    raise AttributeError(name)


__all__ = ["available_model_names", "available_models", "get_model_description", "load"]
