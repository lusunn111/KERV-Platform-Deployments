"""Lightweight exports for the HuggingFace-compatible Prismatic models."""

from importlib import import_module

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

__all__ = [
    "PrismaticConfig",
    "OpenVLAConfig",
    "PrismaticPreTrainedModel",
    "PrismaticVisionBackbone",
    "PrismaticProjector",
    "PrismaticForConditionalGeneration",
    "OpenVLAForActionPrediction",
]


def __getattr__(name):
    if name in __all__[2:]:
        return getattr(import_module(".modeling_prismatic", __name__), name)
    raise AttributeError(name)
