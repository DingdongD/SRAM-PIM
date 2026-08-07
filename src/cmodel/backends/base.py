"""Shared result types for strict external backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorDemand:
    cycle: int
    operand: str
    address: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class ScaleSimResult:
    cycles: int
    utilization: float
    demands: tuple[TensorDemand, ...]
    report_path: str
    trace_directory: str


@dataclass(frozen=True, slots=True)
class SRAMMacroResult:
    access_time_ns: float
    cycle_time_ns: float
    read_energy_pj: float
    write_energy_pj: float
    leakage_mw: float
    area_mm2: float


@dataclass(frozen=True, slots=True)
class BackendCompletion:
    request_id: int
    cycle: int


@dataclass(frozen=True, slots=True)
class NoCCompletion:
    packet_id: int
    cycle: int
    flits: int
    hops: int
