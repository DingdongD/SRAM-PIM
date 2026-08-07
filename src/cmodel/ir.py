"""Execution IR shared by operator lowering and the cycle scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import AddressSpec
from .errors import MappingError


class MicroOpKind(str, Enum):
    BARRIER = "barrier"
    DRAM_READ = "dram_read"
    DRAM_WRITE = "dram_write"
    NOC_SEND = "noc_send"
    SRAM_READ = "sram_read"
    SRAM_WRITE = "sram_write"
    VLINK_SEND = "vlink_send"
    BUFFER_READ = "buffer_read"
    BUFFER_WRITE = "buffer_write"
    ARRAY_STEP = "array_step"
    VECTOR = "vector"
    REDUCE = "reduce"


@dataclass(frozen=True, slots=True)
class Tensor:
    name: str
    address: int
    nbytes: int
    region: str


@dataclass(frozen=True, slots=True)
class MicroOp:
    uid: int
    name: str
    kind: MicroOpKind
    deps: tuple[int, ...]
    resource_id: str | None
    latency_cycles: int
    issue_interval_cycles: int
    nbytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MicroProgram:
    operations: tuple[MicroOp, ...]
    terminals: tuple[int, ...]
    tensors: tuple[Tensor, ...]
    metadata: dict[str, Any]


class ProgramBuilder:
    def __init__(self) -> None:
        self.operations: list[MicroOp] = []
        self.tensors: list[Tensor] = []
        self.next_uid = 0

    def add(
        self,
        *,
        name: str,
        kind: MicroOpKind,
        deps: tuple[int, ...],
        resource_id: str | None,
        latency_cycles: int,
        issue_interval_cycles: int,
        nbytes: int,
        metadata: dict[str, Any],
    ) -> int:
        if latency_cycles < 0 or issue_interval_cycles < 0 or nbytes < 0:
            raise MappingError("micro-op latency, issue interval, and bytes cannot be negative")
        if resource_id is None and (latency_cycles != 0 or issue_interval_cycles != 0):
            raise MappingError("resource-free micro-ops must be zero-cycle barriers")
        uid = self.next_uid
        self.next_uid += 1
        self.operations.append(
            MicroOp(uid, name, kind, deps, resource_id, latency_cycles, issue_interval_cycles, nbytes, metadata)
        )
        return uid

    def barrier(self, name: str, deps: tuple[int, ...]) -> int:
        return self.add(
            name=name,
            kind=MicroOpKind.BARRIER,
            deps=deps,
            resource_id=None,
            latency_cycles=0,
            issue_interval_cycles=0,
            nbytes=0,
            metadata={},
        )

    def build(self, terminals: tuple[int, ...], metadata: dict[str, Any]) -> MicroProgram:
        known = {operation.uid for operation in self.operations}
        for operation in self.operations:
            for dependency in operation.deps:
                if dependency not in known:
                    raise MappingError(f"micro-op {operation.uid} depends on unknown uid {dependency}")
        for terminal in terminals:
            if terminal not in known:
                raise MappingError(f"terminal uid {terminal} does not exist")
        return MicroProgram(tuple(self.operations), terminals, tuple(self.tensors), metadata)


class AddressAllocator:
    def __init__(self, spec: AddressSpec, builder: ProgramBuilder):
        self.spec = spec
        self.builder = builder
        self.cursor = {
            "input": spec.input_base,
            "weight": spec.weight_base,
            "output": spec.output_base,
            "temporary": spec.temporary_base,
            "kv_cache": spec.kv_cache_base,
        }
        self.base = dict(self.cursor)

    def allocate(self, name: str, nbytes: int, region: str) -> Tensor:
        if nbytes <= 0:
            raise MappingError(f"tensor {name} size must be positive")
        if region not in self.cursor:
            raise MappingError(f"unknown address region {region}")
        cursor = self.cursor[region]
        aligned = _align(cursor, self.spec.alignment_bytes)
        limit = self.base[region] + self.spec.region_bytes
        if aligned + nbytes > limit:
            raise MappingError(f"address region {region} overflow while allocating {name}")
        tensor = Tensor(name=name, address=aligned, nbytes=nbytes, region=region)
        self.cursor[region] = aligned + nbytes
        self.builder.tensors.append(tensor)
        return tensor


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
