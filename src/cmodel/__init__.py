"""Strict stacked-SRAM architecture C-model."""

from .config import CModelConfig, load_cmodel_config
from .operators import AttentionMode, AttentionOp, Conv2dOp, GemmOp, LayerNormOp, SoftmaxOp
from .simulator import CModelRun, StrictCModel

__all__ = [
    "AttentionMode",
    "AttentionOp",
    "CModelConfig",
    "CModelRun",
    "Conv2dOp",
    "GemmOp",
    "LayerNormOp",
    "SoftmaxOp",
    "StrictCModel",
    "load_cmodel_config",
]
