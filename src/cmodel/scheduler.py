"""Cycle scheduler coupling local resources with Ramulator2 and BookSim2."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from .backends.base import SRAMMacroResult
from .config import CModelConfig
from .errors import DeadlockError, ResourceError
from .ir import MicroOp, MicroOpKind, MicroProgram


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    resource_id: str
    capacity: int


@dataclass(slots=True)
class _Slot:
    next_issue: int
    last_uid: int | None


@dataclass(frozen=True, slots=True)
class EventRecord:
    uid: int
    name: str
    kind: str
    resource: str | None
    ready_cycle: int
    start_cycle: int
    finish_cycle: int
    queue_wait_cycles: int
    nbytes: int
    critical_predecessor: int | None


@dataclass(frozen=True, slots=True)
class SimulationResult:
    total_cycles: int
    events: Mapping[int, EventRecord]
    critical_path: tuple[int, ...]
    counters: Mapping[str, int | float]
    energy: Mapping[str, float | list[str]]

    def to_dict(self, frequency_hz: int) -> dict:
        return {
            "latency": {
                "total_cycles": self.total_cycles,
                "total_ns": self.total_cycles * 1e9 / frequency_hz,
            },
            "critical_path": list(self.critical_path),
            "counters": dict(self.counters),
            "energy": dict(self.energy),
            "events": [
                {
                    "uid": event.uid,
                    "name": event.name,
                    "kind": event.kind,
                    "resource": event.resource,
                    "ready": event.ready_cycle,
                    "start": event.start_cycle,
                    "finish": event.finish_cycle,
                    "queue_wait": event.queue_wait_cycles,
                    "nbytes": event.nbytes,
                }
                for event in sorted(self.events.values(), key=lambda item: item.uid)
            ],
        }


class CycleScheduler:
    def __init__(self, config: CModelConfig, macro: SRAMMacroResult, ramulator2, booksim2):
        self.config = config
        self.macro = macro
        self.ramulator2 = ramulator2
        self.booksim2 = booksim2
        self.resources = _resource_specs(config)

    def run(self, program: MicroProgram) -> SimulationResult:
        operations = {operation.uid: operation for operation in program.operations}
        pending = set(operations)
        local_heap: list[tuple[int, int]] = []
        external_finish: dict[int, list[int]] = {}
        done: dict[int, int] = {}
        records: dict[int, EventRecord] = {}
        started_external: set[int] = set()
        slots = {
            resource_id: [_Slot(0, None) for _ in range(spec.capacity)]
            for resource_id, spec in self.resources.items()
        }
        dram_issued = 0
        dram_completed = 0
        noc_issued = 0
        noc_completed = 0
        sram_reads = 0
        sram_writes = 0
        buffer_reads = 0
        buffer_writes = 0
        queue_wait_by_kind: dict[str, int] = {}
        busy_by_kind: dict[str, int] = {}

        cycle = 0
        while len(done) != len(operations):
            if cycle > self.config.simulation.max_cycles:
                remaining = sorted(uid for uid in operations if uid not in done)
                raise DeadlockError(f"simulation exceeded max_cycles; remaining={remaining[:20]}")

            while local_heap and local_heap[0][0] <= cycle:
                finish, uid = heapq.heappop(local_heap)
                if uid in done:
                    raise ResourceError(f"local micro-op {uid} completed twice")
                done[uid] = finish

            if cycle in external_finish:
                for uid in external_finish[cycle]:
                    if uid in done:
                        raise ResourceError(f"external micro-op {uid} completed twice")
                    done[uid] = cycle

            progress = True
            while progress:
                progress = False
                for uid in sorted(tuple(pending)):
                    operation = operations[uid]
                    if operation.kind is not MicroOpKind.BARRIER:
                        continue
                    if all(dependency in done for dependency in operation.deps):
                        ready, predecessor = _dependency_ready(operation, done)
                        records[uid] = EventRecord(
                            uid, operation.name, operation.kind.value, None,
                            ready, max(cycle, ready), max(cycle, ready), max(0, cycle - ready),
                            0, predecessor,
                        )
                        done[uid] = max(cycle, ready)
                        pending.remove(uid)
                        progress = True

            for uid in sorted(tuple(pending)):
                operation = operations[uid]
                if operation.kind is MicroOpKind.BARRIER:
                    continue
                if not all(dependency in done for dependency in operation.deps):
                    continue
                ready, predecessor = _dependency_ready(operation, done)
                if ready > cycle:
                    continue

                if operation.kind in {MicroOpKind.DRAM_READ, MicroOpKind.DRAM_WRITE}:
                    if uid in started_external:
                        continue
                    accepted = self.ramulator2.submit(
                        request_id=uid,
                        cycle=cycle,
                        request_type=int(operation.metadata["request_type"]),
                        address=int(operation.metadata["address"]),
                        source_id=int(operation.metadata["source_id"]),
                        nbytes=operation.nbytes,
                    )
                    if not accepted:
                        continue
                    started_external.add(uid)
                    pending.remove(uid)
                    dram_issued += 1
                    records[uid] = EventRecord(
                        uid, operation.name, operation.kind.value, "dram", ready, cycle, -1,
                        cycle - ready, operation.nbytes, predecessor,
                    )
                    _accumulate(queue_wait_by_kind, "dram", cycle - ready)
                    continue

                if operation.kind is MicroOpKind.NOC_SEND:
                    if uid in started_external:
                        continue
                    accepted = self.booksim2.submit(
                        packet_id=uid,
                        cycle=cycle,
                        src=int(operation.metadata["src"]),
                        dst=int(operation.metadata["dst"]),
                        vc=int(operation.metadata["vc"]),
                        flits=int(operation.metadata["flits"]),
                        traffic_class=int(operation.metadata["traffic_class"]),
                    )
                    if not accepted:
                        continue
                    started_external.add(uid)
                    pending.remove(uid)
                    noc_issued += 1
                    records[uid] = EventRecord(
                        uid, operation.name, operation.kind.value, "noc", ready, cycle, -1,
                        cycle - ready, operation.nbytes, predecessor,
                    )
                    _accumulate(queue_wait_by_kind, "noc", cycle - ready)
                    continue

                if operation.resource_id is None:
                    raise ResourceError(f"non-barrier micro-op {uid} has no resource")
                if operation.resource_id not in slots:
                    raise ResourceError(f"unknown local resource {operation.resource_id}")
                slot_index = _available_slot(slots[operation.resource_id], cycle)
                if slot_index is None:
                    continue
                slot = slots[operation.resource_id][slot_index]
                resource_predecessor = slot.last_uid if slot.next_issue > ready else predecessor
                start = cycle
                finish = cycle + operation.latency_cycles
                slot.next_issue = cycle + operation.issue_interval_cycles
                slot.last_uid = uid
                pending.remove(uid)
                records[uid] = EventRecord(
                    uid, operation.name, operation.kind.value, operation.resource_id,
                    ready, start, finish, start - ready, operation.nbytes, resource_predecessor,
                )
                heapq.heappush(local_heap, (finish, uid))
                _accumulate(queue_wait_by_kind, operation.kind.value, start - ready)
                _accumulate(busy_by_kind, operation.kind.value, operation.latency_cycles)
                if operation.kind is MicroOpKind.SRAM_READ:
                    sram_reads += 1
                elif operation.kind is MicroOpKind.SRAM_WRITE:
                    sram_writes += 1
                elif operation.kind is MicroOpKind.BUFFER_READ:
                    buffer_reads += 1
                elif operation.kind is MicroOpKind.BUFFER_WRITE:
                    buffer_writes += 1

            dram_completions = self.ramulator2.tick(cycle)
            for completion in dram_completions:
                uid = completion.request_id
                if uid not in records:
                    raise ResourceError(f"Ramulator2 completed non-issued uid {uid}")
                record = records[uid]
                records[uid] = EventRecord(
                    record.uid, record.name, record.kind, record.resource,
                    record.ready_cycle, record.start_cycle, completion.cycle,
                    record.queue_wait_cycles, record.nbytes, record.critical_predecessor,
                )
                _append_finish(external_finish, completion.cycle, uid)
                _accumulate(busy_by_kind, "dram", completion.cycle - record.start_cycle)
                dram_completed += 1

            noc_completions = self.booksim2.tick(cycle)
            for completion in noc_completions:
                uid = completion.packet_id
                if uid not in records:
                    raise ResourceError(f"BookSim2 completed non-issued uid {uid}")
                record = records[uid]
                records[uid] = EventRecord(
                    record.uid, record.name, record.kind, record.resource,
                    record.ready_cycle, record.start_cycle, completion.cycle,
                    record.queue_wait_cycles, record.nbytes, record.critical_predecessor,
                )
                _append_finish(external_finish, completion.cycle, uid)
                _accumulate(busy_by_kind, "noc", completion.cycle - record.start_cycle)
                noc_completed += 1

            cycle += 1

        if dram_issued != dram_completed:
            raise ResourceError(f"DRAM request conservation failed: issued={dram_issued}, completed={dram_completed}")
        if noc_issued != noc_completed:
            raise ResourceError(f"NoC packet conservation failed: issued={noc_issued}, completed={noc_completed}")
        if self.ramulator2.pending:
            raise ResourceError("Ramulator2 still has pending requests after simulation")
        if self.booksim2.pending:
            raise ResourceError("BookSim2 still has pending packets after simulation")

        total_cycles = max(done[uid] for uid in program.terminals)
        critical_path = _critical_path(program.terminals, records)
        frequency = self.config.architecture.frequency_hz
        total_ns = total_cycles * 1e9 / frequency
        macro_instances = self.config.architecture.sram.total_banks + 1 + self.config.architecture.bank_groups
        dynamic_read_pj = (sram_reads + buffer_reads) * self.macro.read_energy_pj
        dynamic_write_pj = (sram_writes + buffer_writes) * self.macro.write_energy_pj
        leakage_pj = self.macro.leakage_mw * macro_instances * total_ns
        counters: dict[str, int | float] = {
            "dram_requests_issued": dram_issued,
            "dram_requests_completed": dram_completed,
            "noc_packets_issued": noc_issued,
            "noc_packets_completed": noc_completed,
            "sram_reads": sram_reads,
            "sram_writes": sram_writes,
            "buffer_reads": buffer_reads,
            "buffer_writes": buffer_writes,
        }
        for key in sorted(queue_wait_by_kind):
            counters[f"queue_wait_cycles.{key}"] = queue_wait_by_kind[key]
        for key in sorted(busy_by_kind):
            counters[f"busy_cycles.{key}"] = busy_by_kind[key]
        energy: dict[str, float | list[str]] = {
            "sram_dynamic_read_pj": dynamic_read_pj,
            "sram_dynamic_write_pj": dynamic_write_pj,
            "sram_and_buffer_leakage_pj": leakage_pj,
            "modeled_total_pj": dynamic_read_pj + dynamic_write_pj + leakage_pj,
            "unmodeled_components": ["systolic_compute", "vector_compute", "reduction_compute", "noc", "dram", "vertical_link"],
        }
        return SimulationResult(total_cycles, records, critical_path, counters, energy)


def _resource_specs(config: CModelConfig) -> dict[str, ResourceSpec]:
    arch = config.architecture
    specs: dict[str, ResourceSpec] = {}
    for bank in range(arch.sram.total_banks):
        specs[f"sram.bank.{bank}.read"] = ResourceSpec(f"sram.bank.{bank}.read", arch.sram.read_ports_per_bank)
        specs[f"sram.bank.{bank}.write"] = ResourceSpec(f"sram.bank.{bank}.write", arch.sram.write_ports_per_bank)
    for link in range(arch.vertical_link.groups):
        specs[f"vlink.{link}"] = ResourceSpec(f"vlink.{link}", 1)
    specs["gbuf.read"] = ResourceSpec("gbuf.read", arch.global_buffer.read_ports)
    specs["gbuf.write"] = ResourceSpec("gbuf.write", arch.global_buffer.write_ports)
    for group in range(arch.bank_groups):
        specs[f"lbuf.{group}.read"] = ResourceSpec(f"lbuf.{group}.read", arch.global_buffer.read_ports)
        specs[f"lbuf.{group}.write"] = ResourceSpec(f"lbuf.{group}.write", arch.global_buffer.write_ports)
    for array in range(arch.arrays.count):
        specs[f"array.{array}"] = ResourceSpec(f"array.{array}", 1)
    specs["vector.0"] = ResourceSpec("vector.0", 1)
    for reducer in range(arch.reduction.units):
        specs[f"reduce.{reducer}"] = ResourceSpec(f"reduce.{reducer}", 1)
    return specs


def _dependency_ready(operation: MicroOp, done: Mapping[int, int]) -> tuple[int, int | None]:
    if not operation.deps:
        return 0, None
    predecessor = max(operation.deps, key=lambda uid: (done[uid], uid))
    return done[predecessor], predecessor


def _available_slot(slots: list[_Slot], cycle: int) -> int | None:
    candidates = [index for index, slot in enumerate(slots) if slot.next_issue <= cycle]
    if not candidates:
        return None
    return min(candidates, key=lambda index: (slots[index].next_issue, index))


def _append_finish(mapping: dict[int, list[int]], cycle: int, uid: int) -> None:
    if cycle not in mapping:
        mapping[cycle] = []
    mapping[cycle].append(uid)


def _accumulate(mapping: dict[str, int], key: str, value: int) -> None:
    if key not in mapping:
        mapping[key] = 0
    mapping[key] += value


def _critical_path(terminals: tuple[int, ...], records: Mapping[int, EventRecord]) -> tuple[int, ...]:
    terminal = max(terminals, key=lambda uid: (records[uid].finish_cycle, uid))
    reverse: list[int] = []
    seen: set[int] = set()
    current: int | None = terminal
    while current is not None:
        if current in seen:
            raise ResourceError("critical predecessor cycle detected")
        seen.add(current)
        reverse.append(current)
        current = records[current].critical_predecessor
    return tuple(reversed(reverse))
