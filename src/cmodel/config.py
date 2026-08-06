"""Strict YAML loader for the architecture C-model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .architecture import (
    ArchitectureSpec,
    ArraySpec,
    BufferSpec,
    ComputePlacement,
    LinkSpec,
    ReductionSpec,
    StackedSRAMSpec,
)
from .errors import ConfigurationError
from .ir import MappingSpec


@dataclass(frozen=True, slots=True)
class BackendConfig:
    kind: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in {"scalesim", "recorded"}:
            raise ConfigurationError(
                f"unsupported strict backend {self.kind!r}; expected scalesim or recorded"
            )


@dataclass(frozen=True, slots=True)
class CModelConfig:
    architecture: ArchitectureSpec
    mapping: MappingSpec
    backend: BackendConfig


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _strict_keys(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigurationError(f"unknown keys at {path}: {sorted(unknown)}")


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ConfigurationError(f"missing required key {path}.{key}")
    return data[key]


def _int(data: Mapping[str, Any], key: str, path: str) -> int:
    value = _required(data, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}.{key} must be an integer")
    return value


def load_cmodel_config(path: str | Path) -> CModelConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "root")
    _strict_keys(root, {"architecture", "mapping", "backend"}, "root")

    arch_raw = _mapping(_required(root, "architecture", "root"), "architecture")
    _strict_keys(
        arch_raw,
        {
            "name",
            "frequency_hz",
            "compute_placement",
            "arrays",
            "stacked_sram",
            "vertical_link",
            "global_buffer",
            "noc",
            "reduction",
            "bank_groups",
        },
        "architecture",
    )

    arrays_raw = _mapping(
        _required(arch_raw, "arrays", "architecture"), "architecture.arrays"
    )
    _strict_keys(
        arrays_raw, {"count", "rows", "cols", "dataflow"}, "architecture.arrays"
    )
    arrays = ArraySpec(
        count=_int(arrays_raw, "count", "architecture.arrays"),
        rows=_int(arrays_raw, "rows", "architecture.arrays"),
        cols=_int(arrays_raw, "cols", "architecture.arrays"),
        dataflow=str(_required(arrays_raw, "dataflow", "architecture.arrays")),
    )

    sram_raw = _mapping(
        _required(arch_raw, "stacked_sram", "architecture"),
        "architecture.stacked_sram",
    )
    _strict_keys(
        sram_raw,
        {
            "tiers",
            "banks_per_tier",
            "bank_capacity_bytes",
            "line_bytes",
            "read_ports_per_bank",
            "write_ports_per_bank",
            "read_latency_cycles",
            "write_latency_cycles",
            "read_bytes_per_cycle",
            "write_bytes_per_cycle",
        },
        "architecture.stacked_sram",
    )
    sram = StackedSRAMSpec(
        tiers=_int(sram_raw, "tiers", "architecture.stacked_sram"),
        banks_per_tier=_int(
            sram_raw, "banks_per_tier", "architecture.stacked_sram"
        ),
        bank_capacity_bytes=_int(
            sram_raw, "bank_capacity_bytes", "architecture.stacked_sram"
        ),
        line_bytes=_int(sram_raw, "line_bytes", "architecture.stacked_sram"),
        read_ports_per_bank=_int(
            sram_raw, "read_ports_per_bank", "architecture.stacked_sram"
        ),
        write_ports_per_bank=_int(
            sram_raw, "write_ports_per_bank", "architecture.stacked_sram"
        ),
        read_latency_cycles=_int(
            sram_raw, "read_latency_cycles", "architecture.stacked_sram"
        ),
        write_latency_cycles=_int(
            sram_raw, "write_latency_cycles", "architecture.stacked_sram"
        ),
        read_bytes_per_cycle=_int(
            sram_raw, "read_bytes_per_cycle", "architecture.stacked_sram"
        ),
        write_bytes_per_cycle=_int(
            sram_raw, "write_bytes_per_cycle", "architecture.stacked_sram"
        ),
    )

    def parse_link(key: str) -> LinkSpec:
        data = _mapping(
            _required(arch_raw, key, "architecture"), f"architecture.{key}"
        )
        _strict_keys(
            data,
            {"groups", "latency_cycles", "bytes_per_cycle"},
            f"architecture.{key}",
        )
        return LinkSpec(
            groups=_int(data, "groups", f"architecture.{key}"),
            latency_cycles=_int(data, "latency_cycles", f"architecture.{key}"),
            bytes_per_cycle=_int(data, "bytes_per_cycle", f"architecture.{key}"),
        )

    buffer_raw = _mapping(
        _required(arch_raw, "global_buffer", "architecture"),
        "architecture.global_buffer",
    )
    _strict_keys(
        buffer_raw,
        {
            "capacity_bytes",
            "read_ports",
            "write_ports",
            "read_latency_cycles",
            "write_latency_cycles",
            "read_bytes_per_cycle",
            "write_bytes_per_cycle",
        },
        "architecture.global_buffer",
    )
    global_buffer = BufferSpec(
        capacity_bytes=_int(
            buffer_raw, "capacity_bytes", "architecture.global_buffer"
        ),
        read_ports=_int(buffer_raw, "read_ports", "architecture.global_buffer"),
        write_ports=_int(
            buffer_raw, "write_ports", "architecture.global_buffer"
        ),
        read_latency_cycles=_int(
            buffer_raw, "read_latency_cycles", "architecture.global_buffer"
        ),
        write_latency_cycles=_int(
            buffer_raw, "write_latency_cycles", "architecture.global_buffer"
        ),
        read_bytes_per_cycle=_int(
            buffer_raw, "read_bytes_per_cycle", "architecture.global_buffer"
        ),
        write_bytes_per_cycle=_int(
            buffer_raw, "write_bytes_per_cycle", "architecture.global_buffer"
        ),
    )

    reduction_raw = _mapping(
        _required(arch_raw, "reduction", "architecture"),
        "architecture.reduction",
    )
    _strict_keys(
        reduction_raw,
        {"units", "lanes_per_unit", "latency_cycles"},
        "architecture.reduction",
    )
    reduction = ReductionSpec(
        units=_int(reduction_raw, "units", "architecture.reduction"),
        lanes_per_unit=_int(
            reduction_raw, "lanes_per_unit", "architecture.reduction"
        ),
        latency_cycles=_int(
            reduction_raw, "latency_cycles", "architecture.reduction"
        ),
    )

    architecture = ArchitectureSpec(
        name=str(_required(arch_raw, "name", "architecture")),
        frequency_hz=_int(arch_raw, "frequency_hz", "architecture"),
        placement=ComputePlacement(
            str(_required(arch_raw, "compute_placement", "architecture"))
        ),
        arrays=arrays,
        sram=sram,
        vertical_link=parse_link("vertical_link"),
        global_buffer=global_buffer,
        noc=parse_link("noc"),
        reduction=reduction,
        bank_groups=_int(arch_raw, "bank_groups", "architecture"),
    )

    mapping_raw = _mapping(_required(root, "mapping", "root"), "mapping")
    _strict_keys(
        mapping_raw,
        {
            "tile_m",
            "tile_n",
            "tile_k",
            "partition_dim",
            "prefetch_distance_cycles",
        },
        "mapping",
    )
    mapping = MappingSpec(
        tile_m=_int(mapping_raw, "tile_m", "mapping"),
        tile_n=_int(mapping_raw, "tile_n", "mapping"),
        tile_k=_int(mapping_raw, "tile_k", "mapping"),
        partition_dim=str(_required(mapping_raw, "partition_dim", "mapping")),
        prefetch_distance_cycles=_int(
            mapping_raw, "prefetch_distance_cycles", "mapping"
        ),
    )

    backend_raw = _mapping(_required(root, "backend", "root"), "backend")
    if "kind" not in backend_raw:
        raise ConfigurationError("missing required key backend.kind")
    backend = BackendConfig(
        kind=str(backend_raw["kind"]),
        options={key: value for key, value in backend_raw.items() if key != "kind"},
    )
    return CModelConfig(
        architecture=architecture, mapping=mapping, backend=backend
    )
