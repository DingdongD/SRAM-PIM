"""Strict cycle-level C-model for stacked-SRAM NPU architectures."""

from .architecture import ArchitectureSpec, ComputePlacement
from .config import CModelConfig, load_cmodel_config
from .functional import integer_gemm_reference
from .ir import GemmOp, MappingSpec
from .simulator import CModelRun, StrictCModel

__all__ = [
    "ArchitectureSpec",
    "CModelConfig",
    "CModelRun",
    "ComputePlacement",
    "GemmOp",
    "MappingSpec",
    "StrictCModel",
    "integer_gemm_reference",
    "load_cmodel_config",
]
