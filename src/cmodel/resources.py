"""Deterministic resource graph and cycle scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .architecture import ArchitectureSpec
from .errors import DeadlockError, ResourceError
from .ir import MicroOp, MicroOpKind, MicroProgram


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    resource_id: str
    kind: str
    capacity: int
    base_latency_cycles: int = 0
    bytes_per_cycle: int | None = None

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ResourceError("resource_id must be non-empty")
        if self.capacity <= 0:
            raise ResourceError(f"resource {self.resource_id} capacity must be positive")
        if self.base_latency_cycles < 0:
            raise ResourceError("base latency cannot be negative")
        if self.bytes_per_cycle is not None and self.bytes_per_cycle <= 0:
            raise ResourceError("bytes_per_cycle must be positive")

    def service_cycles(self, op: MicroOp) -> int:
        if op.duration_cycles > 0:
            return op.duration_cycles
        transfer = 0
        if op.nbytes:
            if self.bytes_per_cycle is None:
                raise ResourceError(
                    f"resource {self.resource_id} has no bandwidth for {op.nbytes} bytes"
                )
            transfer = math.ceil(op.nbytes / self.bytes_per_cycle)
        duration = self.base_latency_cycles + transfer
        if duration <= 0:
            raise ResourceError(
                f"resource operation {op.uid} has zero service time on {self.resource_id}"
            )
        return duration


@dataclass(slots=True)
class _ResourceState:
    spec: ResourceSpec
    slots: list[tuple[int, int | None]]

    @classmethod
    def create(cls, spec: ResourceSpec) -> "_ResourceState":
        return cls(spec=spec, slots=[(0, None) for _ in range(spec.capacity)])

    def preview(self, earliest: int) -> tuple[int, int, int | None]:
        slot_index = min(
            range(len(self.slots)), key=lambda i: (self.slots[i][0], i)
        )
        available, blocker = self.slots[slot_index]
        return max(earliest, available), slot_index, blocker

    def reserve(
        self,
        *,
        earliest: int,
        duration: int,
        event_uid: int,
    ) -> tuple[int, int, int | None, int]:
        start, slot_index, blocker = self.preview(earliest)
        resource_ready = self.slots[slot_index][0]
        finish = start + duration
        self.slots[slot_index] = (finish, event_uid)
        return start, finish, blocker, resource_ready


@dataclass(frozen=True, slots=True)
class ScheduledOp:
    uid: int
    name: str
    kind: MicroOpKind
    resource_id: str | None
    resource_kind: str | None
    start_cycle: int
    finish_cycle: int
    dependency_ready_cycle: int
    release_ready_cycle: int
    resource_ready_cycle: int
    resource_blocker_uid: int | None
    critical_predecessor_uid: int | None
    nbytes: int

    @property
    def duration_cycles(self) -> int:
        return self.finish_cycle - self.start_cycle

    @property
    def resource_wait_cycles(self) -> int:
        logical_ready = max(self.dependency_ready_cycle, self.release_ready_cycle)
        return max(0, self.start_cycle - logical_ready)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    total_cycles: int
    scheduled: Mapping[int, ScheduledOp]
    critical_path: tuple[int, ...]
    diagnosis: Mapping[str, int | float | str]
    resource_busy_cycles: Mapping[str, int]

    def to_dict(self, frequency_hz: int) -> dict:
        total_ns = self.total_cycles * 1e9 / frequency_hz
        return {
            "latency": {
                "total_cycles": self.total_cycles,
                "total_ns": total_ns,
            },
            "diagnosis": dict(self.diagnosis),
            "resource_busy_cycles": dict(self.resource_busy_cycles),
            "critical_path": list(self.critical_path),
            "events": [
                {
                    "uid": event.uid,
                    "name": event.name,
                    "kind": event.kind.value,
                    "resource": event.resource_id,
                    "start": event.start_cycle,
                    "finish": event.finish_cycle,
                    "resource_wait": event.resource_wait_cycles,
                    "nbytes": event.nbytes,
                }
                for event in sorted(self.scheduled.values(), key=lambda e: e.uid)
            ],
        }


class ResourceGraph:
    def __init__(self, specs: Iterable[ResourceSpec]):
        spec_list = list(specs)
        ids = [spec.resource_id for spec in spec_list]
        if len(ids) != len(set(ids)):
            raise ResourceError("duplicate resource identifier")
        self.specs = {spec.resource_id: spec for spec in spec_list}

    @classmethod
    def from_architecture(cls, arch: ArchitectureSpec) -> "ResourceGraph":
        specs: list[ResourceSpec] = []
        for bank in range(arch.sram.total_banks):
            specs.append(
                ResourceSpec(
                    resource_id=f"sram.bank.{bank}.read",
                    kind="sram_read",
                    capacity=arch.sram.read_ports_per_bank,
                    base_latency_cycles=arch.sram.read_latency_cycles,
                    bytes_per_cycle=arch.sram.read_bytes_per_cycle,
                )
            )
            specs.append(
                ResourceSpec(
                    resource_id=f"sram.bank.{bank}.write",
                    kind="sram_write",
                    capacity=arch.sram.write_ports_per_bank,
                    base_latency_cycles=arch.sram.write_latency_cycles,
                    bytes_per_cycle=arch.sram.write_bytes_per_cycle,
                )
            )
        for group in range(arch.vertical_link.groups):
            specs.append(
                ResourceSpec(
                    resource_id=f"vlink.{group}",
                    kind="vertical_link",
                    capacity=1,
                    base_latency_cycles=arch.vertical_link.latency_cycles,
                    bytes_per_cycle=arch.vertical_link.bytes_per_cycle,
                )
            )
        specs.extend(
            [
                ResourceSpec(
                    resource_id="gbuf.read",
                    kind="global_buffer_read",
                    capacity=arch.global_buffer.read_ports,
                    base_latency_cycles=arch.global_buffer.read_latency_cycles,
                    bytes_per_cycle=arch.global_buffer.read_bytes_per_cycle,
                ),
                ResourceSpec(
                    resource_id="gbuf.write",
                    kind="global_buffer_write",
                    capacity=arch.global_buffer.write_ports,
                    base_latency_cycles=arch.global_buffer.write_latency_cycles,
                    bytes_per_cycle=arch.global_buffer.write_bytes_per_cycle,
                ),
            ]
        )
        for group in range(arch.bank_groups):
            specs.append(
                ResourceSpec(
                    resource_id=f"lbuf.{group}.read",
                    kind="local_buffer_read",
                    capacity=arch.global_buffer.read_ports,
                    base_latency_cycles=arch.global_buffer.read_latency_cycles,
                    bytes_per_cycle=arch.global_buffer.read_bytes_per_cycle,
                )
            )
            specs.append(
                ResourceSpec(
                    resource_id=f"lbuf.{group}.write",
                    kind="local_buffer_write",
                    capacity=arch.global_buffer.write_ports,
                    base_latency_cycles=arch.global_buffer.write_latency_cycles,
                    bytes_per_cycle=arch.global_buffer.write_bytes_per_cycle,
                )
            )
        for group in range(arch.noc.groups):
            specs.append(
                ResourceSpec(
                    resource_id=f"noc.{group}",
                    kind="noc",
                    capacity=1,
                    base_latency_cycles=arch.noc.latency_cycles,
                    bytes_per_cycle=arch.noc.bytes_per_cycle,
                )
            )
        for array_id in range(arch.arrays.count):
            specs.append(
                ResourceSpec(
                    resource_id=f"array.{array_id}",
                    kind="systolic_array",
                    capacity=1,
                )
            )
        for reducer in range(arch.reduction.units):
            specs.append(
                ResourceSpec(
                    resource_id=f"reduce.{reducer}",
                    kind="reduction",
                    capacity=1,
                    base_latency_cycles=arch.reduction.latency_cycles,
                    bytes_per_cycle=arch.reduction.lanes_per_unit * 4,
                )
            )
        return cls(specs)


class EventSimulator:
    """Offline deterministic list scheduler with exact resource attribution."""

    def __init__(self, resource_graph: ResourceGraph):
        self.resource_graph = resource_graph

    def run(self, program: MicroProgram) -> SimulationResult:
        by_id = {op.uid: op for op in program.operations}
        pending = set(by_id)
        scheduled: dict[int, ScheduledOp] = {}
        states = {
            resource_id: _ResourceState.create(spec)
            for resource_id, spec in self.resource_graph.specs.items()
        }
        busy_cycles: dict[str, int] = {resource_id: 0 for resource_id in states}

        while pending:
            candidates: list[
                tuple[int, int, MicroOp, int, int, int, int | None, int | None]
            ] = []
            for uid in sorted(pending):
                op = by_id[uid]
                if any(dep not in scheduled for dep in op.deps):
                    continue
                if op.release_anchor is not None and op.release_anchor not in scheduled:
                    continue

                dep_ready = max(
                    (scheduled[dep].finish_cycle for dep in op.deps), default=0
                )
                dep_pred = None
                if op.deps:
                    dep_pred = max(
                        op.deps,
                        key=lambda dep: (scheduled[dep].finish_cycle, dep),
                    )
                release_ready = 0
                release_pred = None
                if op.release_anchor is not None:
                    release_pred = op.release_anchor
                    release_ready = (
                        scheduled[op.release_anchor].finish_cycle
                        + op.release_offset_cycles
                    )
                earliest = max(dep_ready, release_ready)
                logical_pred = dep_pred
                if release_ready > dep_ready:
                    logical_pred = release_pred
                elif release_ready == dep_ready and release_pred is not None:
                    logical_pred = max(
                        (p for p in (dep_pred, release_pred) if p is not None),
                        default=None,
                    )

                if op.resource_id is None:
                    start = earliest
                    resource_ready = 0
                    blocker = None
                else:
                    if op.resource_id not in states:
                        raise ResourceError(
                            f"micro-op {op.uid} requests unknown resource {op.resource_id}"
                        )
                    start, _, blocker = states[op.resource_id].preview(earliest)
                    slot_index = min(
                        range(len(states[op.resource_id].slots)),
                        key=lambda i: (states[op.resource_id].slots[i][0], i),
                    )
                    resource_ready = states[op.resource_id].slots[slot_index][0]
                candidates.append(
                    (
                        start,
                        uid,
                        op,
                        dep_ready,
                        release_ready,
                        resource_ready,
                        blocker,
                        logical_pred,
                    )
                )

            if not candidates:
                unresolved = sorted(pending)
                raise DeadlockError(
                    f"micro-operation graph cannot make progress; pending={unresolved[:20]}"
                )

            (
                _,
                uid,
                op,
                dep_ready,
                release_ready,
                resource_ready,
                blocker,
                logical_pred,
            ) = min(candidates, key=lambda item: (item[0], item[1]))
            earliest = max(dep_ready, release_ready)

            if op.resource_id is None:
                start = earliest
                finish = start
                resource_kind = None
                critical_pred = logical_pred
            else:
                state = states[op.resource_id]
                duration = state.spec.service_cycles(op)
                start, finish, blocker, resource_ready = state.reserve(
                    earliest=earliest,
                    duration=duration,
                    event_uid=uid,
                )
                busy_cycles[op.resource_id] += duration
                resource_kind = state.spec.kind
                critical_pred = blocker if resource_ready > earliest else logical_pred

            scheduled[uid] = ScheduledOp(
                uid=uid,
                name=op.name,
                kind=op.kind,
                resource_id=op.resource_id,
                resource_kind=resource_kind,
                start_cycle=start,
                finish_cycle=finish,
                dependency_ready_cycle=dep_ready,
                release_ready_cycle=release_ready,
                resource_ready_cycle=resource_ready,
                resource_blocker_uid=blocker,
                critical_predecessor_uid=critical_pred,
                nbytes=op.nbytes,
            )
            pending.remove(uid)

        terminal = max(
            program.terminal_uids,
            key=lambda uid: (scheduled[uid].finish_cycle, uid),
        )
        total_cycles = scheduled[terminal].finish_cycle
        critical_path = self._critical_path(terminal, scheduled)
        diagnosis = self._diagnose(total_cycles, critical_path, scheduled)
        return SimulationResult(
            total_cycles=total_cycles,
            scheduled=scheduled,
            critical_path=critical_path,
            diagnosis=diagnosis,
            resource_busy_cycles=busy_cycles,
        )

    @staticmethod
    def _critical_path(
        terminal_uid: int,
        scheduled: Mapping[int, ScheduledOp],
    ) -> tuple[int, ...]:
        reversed_path: list[int] = []
        current: int | None = terminal_uid
        visited: set[int] = set()
        while current is not None:
            if current in visited:
                raise DeadlockError("cycle detected in reconstructed critical path")
            visited.add(current)
            reversed_path.append(current)
            current = scheduled[current].critical_predecessor_uid
        return tuple(reversed(reversed_path))

    @staticmethod
    def _diagnose(
        total_cycles: int,
        critical_path: tuple[int, ...],
        scheduled: Mapping[int, ScheduledOp],
    ) -> dict[str, int | float | str]:
        attribution: dict[str, int] = {}
        previous_uid: int | None = None
        for uid in critical_path:
            event = scheduled[uid]
            predecessor_finish = (
                scheduled[previous_uid].finish_cycle if previous_uid is not None else 0
            )
            gap = max(0, event.start_cycle - predecessor_finish)
            if gap:
                if event.release_ready_cycle >= event.resource_ready_cycle:
                    gap_key = "release_wait"
                else:
                    gap_key = f"{event.resource_kind or 'resource'}_queue"
                attribution[gap_key] = attribution.get(gap_key, 0) + gap
            duration_key = event.resource_kind or "dependency"
            attribution[duration_key] = (
                attribution.get(duration_key, 0) + event.duration_cycles
            )
            previous_uid = uid

        attributed_cycles = sum(attribution.values())
        if attributed_cycles != total_cycles:
            attribution["unattributed"] = attribution.get("unattributed", 0) + (
                total_cycles - attributed_cycles
            )

        ranked = sorted(attribution.items(), key=lambda item: (-item[1], item[0]))
        primary = ranked[0][0] if ranked else "none"
        primary_cycles = ranked[0][1] if ranked else 0
        diagnosis: dict[str, int | float | str] = {
            "primary_bottleneck": primary,
            "primary_cycles_on_critical_path": primary_cycles,
            "primary_critical_path_share": (
                primary_cycles / total_cycles if total_cycles else 0.0
            ),
        }
        for key, value in ranked:
            diagnosis[f"critical_path_cycles.{key}"] = value
        return diagnosis
