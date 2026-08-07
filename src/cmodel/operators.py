"""Functional operators accepted by the strict C-model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ConfigurationError


class AttentionMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class GemmOp:
    op_id: str
    m: int
    n: int
    k: int
    input_bits: int
    weight_bits: int
    output_bits: int
    accumulator_bits: int

    def __post_init__(self) -> None:
        _positive_name(self.op_id, "GEMM op_id")
        _positive_ints(
            ("m", self.m),
            ("n", self.n),
            ("k", self.k),
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("output_bits", self.output_bits),
            ("accumulator_bits", self.accumulator_bits),
        )
        _byte_aligned(
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("output_bits", self.output_bits),
            ("accumulator_bits", self.accumulator_bits),
        )

    @property
    def mac_count(self) -> int:
        return self.m * self.n * self.k


@dataclass(frozen=True, slots=True)
class Conv2dOp:
    op_id: str
    batch: int
    in_channels: int
    out_channels: int
    input_h: int
    input_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    groups: int
    input_bits: int
    weight_bits: int
    output_bits: int
    accumulator_bits: int

    def __post_init__(self) -> None:
        _positive_name(self.op_id, "CONV op_id")
        _positive_ints(
            ("batch", self.batch),
            ("in_channels", self.in_channels),
            ("out_channels", self.out_channels),
            ("input_h", self.input_h),
            ("input_w", self.input_w),
            ("kernel_h", self.kernel_h),
            ("kernel_w", self.kernel_w),
            ("stride_h", self.stride_h),
            ("stride_w", self.stride_w),
            ("dilation_h", self.dilation_h),
            ("dilation_w", self.dilation_w),
            ("groups", self.groups),
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("output_bits", self.output_bits),
            ("accumulator_bits", self.accumulator_bits),
        )
        if self.pad_h < 0 or self.pad_w < 0:
            raise ConfigurationError("CONV padding cannot be negative")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ConfigurationError("CONV groups must divide input and output channels")
        if self.output_h <= 0 or self.output_w <= 0:
            raise ConfigurationError("CONV output spatial shape is non-positive")
        _byte_aligned(
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("output_bits", self.output_bits),
            ("accumulator_bits", self.accumulator_bits),
        )

    @property
    def output_h(self) -> int:
        extent = self.dilation_h * (self.kernel_h - 1) + 1
        return (self.input_h + 2 * self.pad_h - extent) // self.stride_h + 1

    @property
    def output_w(self) -> int:
        extent = self.dilation_w * (self.kernel_w - 1) + 1
        return (self.input_w + 2 * self.pad_w - extent) // self.stride_w + 1


@dataclass(frozen=True, slots=True)
class SoftmaxOp:
    op_id: str
    rows: int
    cols: int
    element_bits: int

    def __post_init__(self) -> None:
        _positive_name(self.op_id, "Softmax op_id")
        _positive_ints(("rows", self.rows), ("cols", self.cols), ("element_bits", self.element_bits))
        _byte_aligned(("element_bits", self.element_bits),)


@dataclass(frozen=True, slots=True)
class LayerNormOp:
    op_id: str
    rows: int
    cols: int
    element_bits: int
    parameter_bits: int
    epsilon: float

    def __post_init__(self) -> None:
        _positive_name(self.op_id, "LayerNorm op_id")
        _positive_ints(
            ("rows", self.rows),
            ("cols", self.cols),
            ("element_bits", self.element_bits),
            ("parameter_bits", self.parameter_bits),
        )
        _byte_aligned(("element_bits", self.element_bits), ("parameter_bits", self.parameter_bits))
        if self.epsilon <= 0.0:
            raise ConfigurationError("LayerNorm epsilon must be positive")


@dataclass(frozen=True, slots=True)
class AttentionOp:
    op_id: str
    batch: int
    query_tokens: int
    kv_tokens: int
    model_dim: int
    heads: int
    head_dim: int
    mode: AttentionMode
    causal: bool
    input_bits: int
    weight_bits: int
    activation_bits: int
    accumulator_bits: int

    def __post_init__(self) -> None:
        _positive_name(self.op_id, "Attention op_id")
        _positive_ints(
            ("batch", self.batch),
            ("query_tokens", self.query_tokens),
            ("kv_tokens", self.kv_tokens),
            ("model_dim", self.model_dim),
            ("heads", self.heads),
            ("head_dim", self.head_dim),
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("activation_bits", self.activation_bits),
            ("accumulator_bits", self.accumulator_bits),
        )
        if self.heads * self.head_dim != self.model_dim:
            raise ConfigurationError("Attention heads * head_dim must equal model_dim")
        if self.mode is AttentionMode.DECODE and self.query_tokens != 1:
            raise ConfigurationError("decode attention requires query_tokens == 1")
        if self.mode is AttentionMode.PREFILL and self.query_tokens != self.kv_tokens:
            raise ConfigurationError("prefill attention requires query_tokens == kv_tokens")
        _byte_aligned(
            ("input_bits", self.input_bits),
            ("weight_bits", self.weight_bits),
            ("activation_bits", self.activation_bits),
            ("accumulator_bits", self.accumulator_bits),
        )


Operator = GemmOp | Conv2dOp | SoftmaxOp | LayerNormOp | AttentionOp


def _positive_name(value: str, name: str) -> None:
    if not value:
        raise ConfigurationError(f"{name} must be non-empty")


def _positive_ints(*pairs: tuple[str, int]) -> None:
    for name, value in pairs:
        if isinstance(value, bool) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer")


def _byte_aligned(*pairs: tuple[str, int]) -> None:
    for name, value in pairs:
        if value % 8:
            raise ConfigurationError(f"{name} must be byte aligned")
