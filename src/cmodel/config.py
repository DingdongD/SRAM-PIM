"""Strict configuration schema for the architecture C-model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class RepositorySpec:
    path: str
    expected_commit: str


@dataclass(frozen=True, slots=True)
class ScaleSimSpec:
    repository: RepositorySpec
    python_executable: str
    module: str
    architecture_config: str
    run_name: str
    timeout_seconds: int
    trace_word_bytes: int


@dataclass(frozen=True, slots=True)
class PersistentBackendSpec:
    repository: RepositorySpec
    executable: str
    config_path: str
    protocol_version: int
    timeout_seconds: int
    ticks_numerator: int
    ticks_denominator: int


@dataclass(frozen=True, slots=True)
class SRAMMacroBackendSpec:
    kind: str
    repository: RepositorySpec
    executable: str
    config_path: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class BackendsSpec:
    scalesim: ScaleSimSpec
    ramulator2: PersistentBackendSpec
    booksim2: PersistentBackendSpec
    sram_macro: SRAMMacroBackendSpec


@dataclass(frozen=True, slots=True)
class ArraySpec:
    count: int
    rows: int
    cols: int
    dataflow: str


@dataclass(frozen=True, slots=True)
class SRAMSpec:
    tiers: int
    banks_per_tier: int
    bank_capacity_bytes: int
    line_bytes: int
    read_ports_per_bank: int
    write_ports_per_bank: int

    @property
    def total_banks(self) -> int:
        return self.tiers * self.banks_per_tier

    @property
    def total_capacity_bytes(self) -> int:
        return self.total_banks * self.bank_capacity_bytes


@dataclass(frozen=True, slots=True)
class VerticalLinkSpec:
    groups: int
    latency_cycles: int
    bytes_per_cycle: int


@dataclass(frozen=True, slots=True)
class BufferSpec:
    capacity_bytes: int
    read_ports: int
    write_ports: int


@dataclass(frozen=True, slots=True)
class VectorSpec:
    lanes: int
    bytes_per_cycle: int
    max_cycles_per_element: int
    sub_cycles_per_element: int
    exp_cycles_per_element: int
    add_cycles_per_element: int
    reciprocal_cycles_per_element: int
    multiply_cycles_per_element: int
    rsqrt_cycles_per_element: int
    affine_cycles_per_element: int


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    units: int
    lanes_per_unit: int
    cycles_per_stage: int


@dataclass(frozen=True, slots=True)
class NoCSpec:
    nodes: int
    flit_bytes: int
    endpoint_for_global_buffer: int
    endpoint_for_dram: int
    endpoint_for_bank_group: tuple[int, ...]
    vc_activation: int
    vc_weight: int
    vc_output: int
    vc_partial_sum: int
    vc_control: int


@dataclass(frozen=True, slots=True)
class DRAMSpec:
    read_request_type: int
    write_request_type: int
    transaction_bytes: int
    source_id: int


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    name: str
    frequency_hz: int
    compute_placement: str
    bank_groups: int
    arrays: ArraySpec
    sram: SRAMSpec
    vertical_link: VerticalLinkSpec
    global_buffer: BufferSpec
    vector: VectorSpec
    reduction: ReductionSpec
    noc: NoCSpec
    dram: DRAMSpec


@dataclass(frozen=True, slots=True)
class MappingSpec:
    tile_m: int
    tile_n: int
    tile_k: int
    partition_dim: str


@dataclass(frozen=True, slots=True)
class AddressSpec:
    input_base: int
    weight_base: int
    output_base: int
    temporary_base: int
    kv_cache_base: int
    region_bytes: int
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    max_cycles: int


@dataclass(frozen=True, slots=True)
class CModelConfig:
    architecture: ArchitectureSpec
    mapping: MappingSpec
    addresses: AddressSpec
    backends: BackendsSpec
    simulation: SimulationSpec


def load_cmodel_config(path: str | Path) -> CModelConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        root = _dict(yaml.safe_load(handle), "root")

    _keys(root, {"architecture", "mapping", "addresses", "backends", "simulation"}, "root")
    architecture = _parse_architecture(_dict(root["architecture"], "architecture"))
    mapping = _parse_mapping(_dict(root["mapping"], "mapping"))
    addresses = _parse_addresses(_dict(root["addresses"], "addresses"))
    backends = _parse_backends(_dict(root["backends"], "backends"))
    simulation_raw = _dict(root["simulation"], "simulation")
    _keys(simulation_raw, {"max_cycles"}, "simulation")
    simulation = SimulationSpec(_int(simulation_raw["max_cycles"], "simulation.max_cycles"))
    config = CModelConfig(architecture, mapping, addresses, backends, simulation)
    _validate(config)
    return config


def _parse_architecture(raw: dict[str, Any]) -> ArchitectureSpec:
    _keys(
        raw,
        {
            "name", "frequency_hz", "compute_placement", "bank_groups", "arrays",
            "stacked_sram", "vertical_link", "global_buffer", "vector", "reduction",
            "noc", "dram",
        },
        "architecture",
    )
    arrays_raw = _dict(raw["arrays"], "architecture.arrays")
    _keys(arrays_raw, {"count", "rows", "cols", "dataflow"}, "architecture.arrays")
    arrays = ArraySpec(
        _int(arrays_raw["count"], "architecture.arrays.count"),
        _int(arrays_raw["rows"], "architecture.arrays.rows"),
        _int(arrays_raw["cols"], "architecture.arrays.cols"),
        _str(arrays_raw["dataflow"], "architecture.arrays.dataflow"),
    )

    sram_raw = _dict(raw["stacked_sram"], "architecture.stacked_sram")
    _keys(
        sram_raw,
        {"tiers", "banks_per_tier", "bank_capacity_bytes", "line_bytes", "read_ports_per_bank", "write_ports_per_bank"},
        "architecture.stacked_sram",
    )
    sram = SRAMSpec(
        _int(sram_raw["tiers"], "architecture.stacked_sram.tiers"),
        _int(sram_raw["banks_per_tier"], "architecture.stacked_sram.banks_per_tier"),
        _int(sram_raw["bank_capacity_bytes"], "architecture.stacked_sram.bank_capacity_bytes"),
        _int(sram_raw["line_bytes"], "architecture.stacked_sram.line_bytes"),
        _int(sram_raw["read_ports_per_bank"], "architecture.stacked_sram.read_ports_per_bank"),
        _int(sram_raw["write_ports_per_bank"], "architecture.stacked_sram.write_ports_per_bank"),
    )

    vlink_raw = _dict(raw["vertical_link"], "architecture.vertical_link")
    _keys(vlink_raw, {"groups", "latency_cycles", "bytes_per_cycle"}, "architecture.vertical_link")
    vlink = VerticalLinkSpec(
        _int(vlink_raw["groups"], "architecture.vertical_link.groups"),
        _int(vlink_raw["latency_cycles"], "architecture.vertical_link.latency_cycles"),
        _int(vlink_raw["bytes_per_cycle"], "architecture.vertical_link.bytes_per_cycle"),
    )

    buffer_raw = _dict(raw["global_buffer"], "architecture.global_buffer")
    _keys(buffer_raw, {"capacity_bytes", "read_ports", "write_ports"}, "architecture.global_buffer")
    buffer = BufferSpec(
        _int(buffer_raw["capacity_bytes"], "architecture.global_buffer.capacity_bytes"),
        _int(buffer_raw["read_ports"], "architecture.global_buffer.read_ports"),
        _int(buffer_raw["write_ports"], "architecture.global_buffer.write_ports"),
    )

    vector_raw = _dict(raw["vector"], "architecture.vector")
    vector_keys = {
        "lanes", "bytes_per_cycle", "max_cycles_per_element", "sub_cycles_per_element",
        "exp_cycles_per_element", "add_cycles_per_element", "reciprocal_cycles_per_element",
        "multiply_cycles_per_element", "rsqrt_cycles_per_element", "affine_cycles_per_element",
    }
    _keys(vector_raw, vector_keys, "architecture.vector")
    vector = VectorSpec(*(_int(vector_raw[key], f"architecture.vector.{key}") for key in (
        "lanes", "bytes_per_cycle", "max_cycles_per_element", "sub_cycles_per_element",
        "exp_cycles_per_element", "add_cycles_per_element", "reciprocal_cycles_per_element",
        "multiply_cycles_per_element", "rsqrt_cycles_per_element", "affine_cycles_per_element",
    )))

    reduction_raw = _dict(raw["reduction"], "architecture.reduction")
    _keys(reduction_raw, {"units", "lanes_per_unit", "cycles_per_stage"}, "architecture.reduction")
    reduction = ReductionSpec(
        _int(reduction_raw["units"], "architecture.reduction.units"),
        _int(reduction_raw["lanes_per_unit"], "architecture.reduction.lanes_per_unit"),
        _int(reduction_raw["cycles_per_stage"], "architecture.reduction.cycles_per_stage"),
    )

    noc_raw = _dict(raw["noc"], "architecture.noc")
    _keys(
        noc_raw,
        {"nodes", "flit_bytes", "endpoint_for_global_buffer", "endpoint_for_dram", "endpoint_for_bank_group", "vc_activation", "vc_weight", "vc_output", "vc_partial_sum", "vc_control"},
        "architecture.noc",
    )
    endpoints = _int_tuple(noc_raw["endpoint_for_bank_group"], "architecture.noc.endpoint_for_bank_group")
    noc = NoCSpec(
        _int(noc_raw["nodes"], "architecture.noc.nodes"),
        _int(noc_raw["flit_bytes"], "architecture.noc.flit_bytes"),
        _int(noc_raw["endpoint_for_global_buffer"], "architecture.noc.endpoint_for_global_buffer"),
        _int(noc_raw["endpoint_for_dram"], "architecture.noc.endpoint_for_dram"),
        endpoints,
        _int(noc_raw["vc_activation"], "architecture.noc.vc_activation"),
        _int(noc_raw["vc_weight"], "architecture.noc.vc_weight"),
        _int(noc_raw["vc_output"], "architecture.noc.vc_output"),
        _int(noc_raw["vc_partial_sum"], "architecture.noc.vc_partial_sum"),
        _int(noc_raw["vc_control"], "architecture.noc.vc_control"),
    )

    dram_raw = _dict(raw["dram"], "architecture.dram")
    _keys(dram_raw, {"read_request_type", "write_request_type", "transaction_bytes", "source_id"}, "architecture.dram")
    dram = DRAMSpec(
        _int(dram_raw["read_request_type"], "architecture.dram.read_request_type"),
        _int(dram_raw["write_request_type"], "architecture.dram.write_request_type"),
        _int(dram_raw["transaction_bytes"], "architecture.dram.transaction_bytes"),
        _int(dram_raw["source_id"], "architecture.dram.source_id"),
    )

    return ArchitectureSpec(
        _str(raw["name"], "architecture.name"),
        _int(raw["frequency_hz"], "architecture.frequency_hz"),
        _str(raw["compute_placement"], "architecture.compute_placement"),
        _int(raw["bank_groups"], "architecture.bank_groups"),
        arrays, sram, vlink, buffer, vector, reduction, noc, dram,
    )


def _parse_mapping(raw: dict[str, Any]) -> MappingSpec:
    _keys(raw, {"tile_m", "tile_n", "tile_k", "partition_dim"}, "mapping")
    return MappingSpec(
        _int(raw["tile_m"], "mapping.tile_m"),
        _int(raw["tile_n"], "mapping.tile_n"),
        _int(raw["tile_k"], "mapping.tile_k"),
        _str(raw["partition_dim"], "mapping.partition_dim"),
    )


def _parse_addresses(raw: dict[str, Any]) -> AddressSpec:
    keys = {"input_base", "weight_base", "output_base", "temporary_base", "kv_cache_base", "region_bytes", "alignment_bytes"}
    _keys(raw, keys, "addresses")
    return AddressSpec(*(_int(raw[key], f"addresses.{key}") for key in (
        "input_base", "weight_base", "output_base", "temporary_base", "kv_cache_base", "region_bytes", "alignment_bytes"
    )))


def _parse_backends(raw: dict[str, Any]) -> BackendsSpec:
    _keys(raw, {"scalesim", "ramulator2", "booksim2", "sram_macro"}, "backends")

    scalesim_raw = _dict(raw["scalesim"], "backends.scalesim")
    _keys(scalesim_raw, {"repository", "python_executable", "module", "architecture_config", "run_name", "timeout_seconds", "trace_word_bytes"}, "backends.scalesim")
    scalesim = ScaleSimSpec(
        _repository(_dict(scalesim_raw["repository"], "backends.scalesim.repository"), "backends.scalesim.repository"),
        _str(scalesim_raw["python_executable"], "backends.scalesim.python_executable"),
        _str(scalesim_raw["module"], "backends.scalesim.module"),
        _str(scalesim_raw["architecture_config"], "backends.scalesim.architecture_config"),
        _str(scalesim_raw["run_name"], "backends.scalesim.run_name"),
        _int(scalesim_raw["timeout_seconds"], "backends.scalesim.timeout_seconds"),
        _int(scalesim_raw["trace_word_bytes"], "backends.scalesim.trace_word_bytes"),
    )

    ramulator = _persistent(_dict(raw["ramulator2"], "backends.ramulator2"), "backends.ramulator2")
    booksim = _persistent(_dict(raw["booksim2"], "backends.booksim2"), "backends.booksim2")

    macro_raw = _dict(raw["sram_macro"], "backends.sram_macro")
    _keys(macro_raw, {"kind", "repository", "executable", "config_path", "timeout_seconds"}, "backends.sram_macro")
    macro = SRAMMacroBackendSpec(
        _str(macro_raw["kind"], "backends.sram_macro.kind"),
        _repository(_dict(macro_raw["repository"], "backends.sram_macro.repository"), "backends.sram_macro.repository"),
        _str(macro_raw["executable"], "backends.sram_macro.executable"),
        _str(macro_raw["config_path"], "backends.sram_macro.config_path"),
        _int(macro_raw["timeout_seconds"], "backends.sram_macro.timeout_seconds"),
    )
    return BackendsSpec(scalesim, ramulator, booksim, macro)


def _persistent(raw: dict[str, Any], path: str) -> PersistentBackendSpec:
    _keys(raw, {"repository", "executable", "config_path", "protocol_version", "timeout_seconds", "ticks_numerator", "ticks_denominator"}, path)
    return PersistentBackendSpec(
        _repository(_dict(raw["repository"], f"{path}.repository"), f"{path}.repository"),
        _str(raw["executable"], f"{path}.executable"),
        _str(raw["config_path"], f"{path}.config_path"),
        _int(raw["protocol_version"], f"{path}.protocol_version"),
        _int(raw["timeout_seconds"], f"{path}.timeout_seconds"),
        _int(raw["ticks_numerator"], f"{path}.ticks_numerator"),
        _int(raw["ticks_denominator"], f"{path}.ticks_denominator"),
    )


def _repository(raw: dict[str, Any], path: str) -> RepositorySpec:
    _keys(raw, {"path", "expected_commit"}, path)
    return RepositorySpec(_str(raw["path"], f"{path}.path"), _str(raw["expected_commit"], f"{path}.expected_commit"))


def _validate(config: CModelConfig) -> None:
    arch = config.architecture
    positive = (
        arch.frequency_hz, arch.bank_groups, arch.arrays.count, arch.arrays.rows, arch.arrays.cols,
        arch.sram.tiers, arch.sram.banks_per_tier, arch.sram.bank_capacity_bytes, arch.sram.line_bytes,
        arch.sram.read_ports_per_bank, arch.sram.write_ports_per_bank, arch.vertical_link.groups,
        arch.vertical_link.latency_cycles, arch.vertical_link.bytes_per_cycle, arch.global_buffer.capacity_bytes,
        arch.global_buffer.read_ports, arch.global_buffer.write_ports, arch.vector.lanes, arch.vector.bytes_per_cycle,
        arch.vector.max_cycles_per_element, arch.vector.sub_cycles_per_element, arch.vector.exp_cycles_per_element,
        arch.vector.add_cycles_per_element, arch.vector.reciprocal_cycles_per_element,
        arch.vector.multiply_cycles_per_element, arch.vector.rsqrt_cycles_per_element, arch.vector.affine_cycles_per_element,
        arch.reduction.units, arch.reduction.lanes_per_unit, arch.reduction.cycles_per_stage, arch.noc.nodes,
        arch.noc.flit_bytes, arch.dram.transaction_bytes, config.mapping.tile_m, config.mapping.tile_n,
        config.mapping.tile_k, config.addresses.region_bytes, config.addresses.alignment_bytes, config.simulation.max_cycles,
        config.backends.scalesim.timeout_seconds, config.backends.scalesim.trace_word_bytes,
        config.backends.ramulator2.protocol_version, config.backends.ramulator2.timeout_seconds,
        config.backends.ramulator2.ticks_numerator, config.backends.ramulator2.ticks_denominator,
        config.backends.booksim2.protocol_version, config.backends.booksim2.timeout_seconds,
        config.backends.booksim2.ticks_numerator, config.backends.booksim2.ticks_denominator,
        config.backends.sram_macro.timeout_seconds,
    )
    if any(value <= 0 for value in positive):
        raise ConfigurationError("all architecture, mapping, and address capacity fields must be positive")
    if arch.compute_placement not in {"centralized_logic", "bank_local_logic"}:
        raise ConfigurationError("compute_placement must be centralized_logic or bank_local_logic")
    if arch.arrays.dataflow not in {"ws", "os", "is"}:
        raise ConfigurationError("array dataflow must be ws, os, or is")
    if config.mapping.partition_dim not in {"N", "K"}:
        raise ConfigurationError("partition_dim must be N or K")
    if arch.sram.total_banks % arch.bank_groups:
        raise ConfigurationError("total SRAM banks must be divisible by bank_groups")
    if arch.compute_placement == "bank_local_logic" and arch.arrays.count != arch.bank_groups:
        raise ConfigurationError("bank_local_logic requires one array per bank group")
    if len(arch.noc.endpoint_for_bank_group) != arch.bank_groups:
        raise ConfigurationError("endpoint_for_bank_group length must equal bank_groups")
    for endpoint in (arch.noc.endpoint_for_global_buffer, arch.noc.endpoint_for_dram, *arch.noc.endpoint_for_bank_group):
        if endpoint < 0 or endpoint >= arch.noc.nodes:
            raise ConfigurationError("NoC endpoint index is out of range")
    if arch.sram.bank_capacity_bytes % arch.sram.line_bytes:
        raise ConfigurationError("SRAM bank capacity must be a multiple of line_bytes")
    vcs = (arch.noc.vc_activation, arch.noc.vc_weight, arch.noc.vc_output, arch.noc.vc_partial_sum, arch.noc.vc_control)
    if any(vc < 0 for vc in vcs):
        raise ConfigurationError("NoC VC indices cannot be negative")
    if len(set(vcs)) != len(vcs):
        raise ConfigurationError("NoC traffic classes must use distinct VCs")
    if arch.dram.read_request_type < 0 or arch.dram.write_request_type < 0 or arch.dram.source_id < 0:
        raise ConfigurationError("DRAM request type IDs and source_id cannot be negative")
    bases = (
        config.addresses.input_base,
        config.addresses.weight_base,
        config.addresses.output_base,
        config.addresses.temporary_base,
        config.addresses.kv_cache_base,
    )
    if any(base < 0 or base % config.addresses.alignment_bytes for base in bases):
        raise ConfigurationError("address region bases must be non-negative and aligned")
    ordered = sorted(bases)
    for left, right in zip(ordered, ordered[1:]):
        if left + config.addresses.region_bytes > right:
            raise ConfigurationError("address regions overlap")
    if config.backends.sram_macro.kind not in {"cacti", "destiny"}:
        raise ConfigurationError("sram_macro.kind must be cacti or destiny")


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _keys(raw: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConfigurationError(f"{path} keys mismatch: missing={missing}, extra={extra}")


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer")
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value


def _int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a list")
    result = tuple(_int(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ConfigurationError(f"{path} must be non-empty")
    return result
