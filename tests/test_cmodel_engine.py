from __future__ import annotations

import numpy as np
import pytest

from src.cmodel.architecture import (
    ArchitectureSpec,
    ArraySpec,
    BufferSpec,
    ComputePlacement,
    LinkSpec,
    ReductionSpec,
    StackedSRAMSpec,
)
from src.cmodel.backends.base import SystolicBackend
from src.cmodel.errors import MappingError
from src.cmodel.functional import integer_gemm_reference
from src.cmodel.ir import (
    GemmOp,
    MappingSpec,
    MicroOpKind,
    MicroProgramBuilder,
    Operand,
    OperandDemand,
    SystolicInvocation,
    SystolicResult,
)
from src.cmodel.lowering import ArchitectureLowerer
from src.cmodel.resources import EventSimulator, ResourceGraph, ResourceSpec


class DeterministicBackend(SystolicBackend):
    def __init__(self) -> None:
        self.invocations: list[SystolicInvocation] = []

    def run(self, invocation: SystolicInvocation) -> SystolicResult:
        self.invocations.append(invocation)
        return SystolicResult(
            backend_name="deterministic-test",
            backend_version="1",
            config_digest="abcd1234",
            total_cycles=4,
            utilization=0.5,
            demands=(
                OperandDemand(0, Operand.INPUT, 0, 1),
                OperandDemand(0, Operand.WEIGHT, 0, 1),
                OperandDemand(1, Operand.INPUT, 1, 1),
                OperandDemand(1, Operand.WEIGHT, 1, 1),
                OperandDemand(3, Operand.OUTPUT, 0, 4),
            ),
        )


def make_arch(
    placement: ComputePlacement, partition_groups: int = 2
) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=placement.value,
        frequency_hz=1_000_000_000,
        placement=placement,
        arrays=ArraySpec(
            count=(
                1
                if placement is ComputePlacement.CENTRALIZED_LOGIC
                else partition_groups
            ),
            rows=8,
            cols=8,
            dataflow="ws",
        ),
        sram=StackedSRAMSpec(
            tiers=1,
            banks_per_tier=8,
            bank_capacity_bytes=4096,
            line_bytes=16,
            read_ports_per_bank=1,
            write_ports_per_bank=1,
            read_latency_cycles=2,
            write_latency_cycles=2,
            read_bytes_per_cycle=16,
            write_bytes_per_cycle=16,
        ),
        vertical_link=LinkSpec(groups=2, latency_cycles=1, bytes_per_cycle=16),
        global_buffer=BufferSpec(
            capacity_bytes=4096,
            read_ports=1,
            write_ports=1,
            read_latency_cycles=1,
            write_latency_cycles=1,
            read_bytes_per_cycle=16,
            write_bytes_per_cycle=16,
        ),
        noc=LinkSpec(groups=2, latency_cycles=1, bytes_per_cycle=16),
        reduction=ReductionSpec(units=1, lanes_per_unit=8, latency_cycles=1),
        bank_groups=partition_groups,
    )


def test_resource_contention_is_serialized_and_attributed() -> None:
    graph = ResourceGraph([ResourceSpec("r0", "test_resource", capacity=1)])
    builder = MicroProgramBuilder()
    first = builder.add(
        name="first",
        kind=MicroOpKind.ARRAY_STEP,
        resource_id="r0",
        duration_cycles=5,
    )
    second = builder.add(
        name="second",
        kind=MicroOpKind.ARRAY_STEP,
        resource_id="r0",
        duration_cycles=3,
    )
    done = builder.add(
        name="done",
        kind=MicroOpKind.BARRIER,
        resource_id=None,
        duration_cycles=0,
        deps=(first, second),
    )
    result = EventSimulator(graph).run(builder.build((done,)))
    assert result.total_cycles == 8
    assert result.scheduled[second].start_cycle == 5
    assert result.scheduled[second].resource_blocker_uid == first
    assert "test_resource" in result.diagnosis["primary_bottleneck"]
    attributed = sum(
        value
        for key, value in result.diagnosis.items()
        if key.startswith("critical_path_cycles.")
    )
    assert attributed == result.total_cycles


def test_same_backend_drives_both_architectures() -> None:
    op = GemmOp("g", 4, 4, 4)
    mapping = MappingSpec(
        4, 4, 4, partition_dim="N", prefetch_distance_cycles=4
    )

    central_backend = DeterministicBackend()
    central_arch = make_arch(ComputePlacement.CENTRALIZED_LOGIC)
    central_program = ArchitectureLowerer(
        central_arch, mapping, central_backend
    ).lower_gemm(op)
    central = EventSimulator(ResourceGraph.from_architecture(central_arch)).run(
        central_program
    )

    local_backend = DeterministicBackend()
    local_arch = make_arch(ComputePlacement.BANK_LOCAL_LOGIC)
    local_program = ArchitectureLowerer(
        local_arch, mapping, local_backend
    ).lower_gemm(op)
    local = EventSimulator(ResourceGraph.from_architecture(local_arch)).run(
        local_program
    )

    assert central_backend.invocations
    assert local_backend.invocations
    assert all(
        inv.array_rows == 8 and inv.array_cols == 8
        for inv in central_backend.invocations
    )
    assert all(
        inv.array_rows == 8 and inv.array_cols == 8
        for inv in local_backend.invocations
    )

    central_resources = {event.resource_id for event in central.scheduled.values()}
    local_resources = {event.resource_id for event in local.scheduled.values()}
    assert "gbuf.read" in central_resources
    assert any(
        resource and resource.startswith("lbuf.") for resource in local_resources
    )
    assert any(
        resource and resource.startswith("noc.") for resource in local_resources
    )
    assert central.total_cycles > 0
    assert local.total_cycles > 0


def test_k_partition_emits_global_reduction() -> None:
    op = GemmOp("gk", 4, 4, 8)
    mapping = MappingSpec(
        4, 4, 8, partition_dim="K", prefetch_distance_cycles=4
    )
    arch = make_arch(ComputePlacement.BANK_LOCAL_LOGIC)
    program = ArchitectureLowerer(
        arch, mapping, DeterministicBackend()
    ).lower_gemm(op)
    assert any(item.kind is MicroOpKind.GLOBAL_REDUCE for item in program.operations)
    result = EventSimulator(ResourceGraph.from_architecture(arch)).run(program)
    assert result.total_cycles > 0
    assert any(
        event.resource_kind == "reduction" for event in result.scheduled.values()
    )


def test_integer_reference_is_bit_exact_and_rejects_overflow() -> None:
    op = GemmOp("ref", 2, 2, 2, accumulator_bits=32)
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.array([[5, 6], [7, 8]], dtype=np.int8)
    result = integer_gemm_reference(op, a, b)
    np.testing.assert_array_equal(
        result, np.array([[19, 22], [43, 50]], dtype=np.int32)
    )

    overflow_op = GemmOp("overflow", 1, 1, 2, accumulator_bits=8)
    with pytest.raises(OverflowError):
        integer_gemm_reference(
            overflow_op,
            np.array([[127, 127]], dtype=np.int8),
            np.array([[127], [127]], dtype=np.int8),
        )


def test_oversized_workload_fails_without_spill_fallback() -> None:
    arch = make_arch(ComputePlacement.CENTRALIZED_LOGIC)
    mapping = MappingSpec(4, 4, 4, partition_dim="N")
    op = GemmOp("too_large", 256, 256, 256)
    with pytest.raises(MappingError, match="no DRAM/spill fallback"):
        ArchitectureLowerer(
            arch, mapping, DeterministicBackend()
        ).lower_gemm(op)
