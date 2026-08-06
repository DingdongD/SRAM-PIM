"""Bit-exact functional reference helpers."""

from __future__ import annotations

import numpy as np

from .errors import ConfigurationError
from .ir import GemmOp


def integer_gemm_reference(op: GemmOp, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (op.m, op.k):
        raise ConfigurationError(f"A shape must be {(op.m, op.k)}, got {a.shape}")
    if b.shape != (op.k, op.n):
        raise ConfigurationError(f"B shape must be {(op.k, op.n)}, got {b.shape}")
    if op.accumulator_bits > 64:
        raise ConfigurationError("NumPy reference supports accumulator_bits <= 64")

    a64 = np.asarray(a, dtype=np.int64)
    b64 = np.asarray(b, dtype=np.int64)
    result = a64 @ b64
    signed_min = -(1 << (op.accumulator_bits - 1))
    signed_max = (1 << (op.accumulator_bits - 1)) - 1
    if np.any(result < signed_min) or np.any(result > signed_max):
        raise OverflowError(
            "GEMM accumulation exceeds the declared accumulator width; "
            "strict mode does not silently saturate or wrap"
        )
    dtype = np.int32 if op.accumulator_bits <= 32 else np.int64
    return result.astype(dtype)
