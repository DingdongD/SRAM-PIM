from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.cmodel.backends.base import BackendCompletion, NoCCompletion, SRAMMacroResult, ScaleSimResult, TensorDemand
from src.cmodel.config import (
    AddressSpec,
    ArchitectureSpec,
    ArraySpec,
    BackendsSpec,
    BufferSpec,
    CModelConfig,
    DRAMSpec,
    MappingSpec,
    NoCSpec,
    PersistentBackendSpec,
    ReductionSpec,
    RepositorySpec,
    SRAMMacroBackendSpec,
    SRAMSpec,
    ScaleSimSpec,
    SimulationSpec,
    VectorSpec,
    VerticalLinkSpec,
)
from src.cmodel.lowering import ArchitectureLowerer
from src.cmodel.operators import AttentionMode, AttentionOp, Conv2dOp, GemmOp, LayerNormOp, SoftmaxOp
from src.cmodel.scheduler import CycleScheduler


class FakeScaleSim:
    def run_gemm(self, *, op_id: str, m: int, n: int, k: int, input_base: int, weight_base: int, output_base: int) -> ScaleSimResult:
        demands = (
            TensorDemand(0, "input", input_base, 1),
            TensorDemand(0, "weight", weight_base, 1),
            TensorDemand(2, "output", output_base, 4),
        )
        return ScaleSimResult(3, 0.75, demands, f"{op_id}.csv", f"{op_id}.trace")


class FakeRamulator:
    def __init__(self) -> None:
        self.current_cycle = 0
        self.pending: dict[int, int] = {}

    def submit(self, *, request_id: int, cycle: int, request_type: int, address: int, source_id: int, nbytes: int) -> bool:
        assert cycle == self.current_cycle
        assert request_id not in self.pending
        assert request_type in {0, 1}
        assert address >= 0 and source_id >= 0 and nbytes > 0
        self.pending[request_id] = cycle + 2
        return True

    def tick(self, cycle: int) -> tuple[BackendCompletion, ...]:
        assert cycle == self.current_cycle
        completed = tuple(
            BackendCompletion(request_id, cycle + 1)
            for request_id, ready in tuple(self.pending.items())
            if ready <= cycle
        )
        for item in completed:
            del self.pending[item.request_id]
        self.current_cycle += 1
        return completed

    def close(self) -> None:
        assert not self.pending


class FakeBookSim:
    def __init__(self) -> None:
        self.current_cycle = 0
        self.pending: dict[int, tuple[int, int]] = {}

    def submit(self, *, packet_id: int, cycle: int, src: int, dst: int, vc: int, flits: int, traffic_class: int) -> bool:
        assert cycle == self.current_cycle
        assert packet_id not in self.pending
        assert src >= 0 and dst >= 0 and vc >= 0 and flits > 0 and traffic_class == 0
        self.pending[packet_id] = (cycle + flits + 1, flits)
        return True

    def tick(self, cycle: int) -> tuple[NoCCompletion, ...]:
        assert cycle == self.current_cycle
        completed = tuple(
            NoCCompletion(packet_id, cycle + 1, flits, 2)
            for packet_id, (ready, flits) in tuple(self.pending.items())
            if ready <= cycle
        )
        for item in completed:
            del self.pending[item.packet_id]
        self.current_cycle += 1
        return completed

    def close(self) -> None:
        assert not self.pending


def config(placement: str = "centralized_logic", partition: str = "N") -> CModelConfig:
    repo = RepositorySpec("/tmp/repo", "0123456789abcdef0123456789abcdef01234567")
    persistent = PersistentBackendSpec(repo, "/tmp/tool", "/tmp/config", 1, 10, 1, 1)
    backends = BackendsSpec(
        ScaleSimSpec(repo, "python3", "scalesim.scale", "/tmp/scale.cfg", "strict_cmodel", 10, 1),
        persistent,
        persistent,
        SRAMMacroBackendSpec("cacti", repo, "/tmp/cacti", "/tmp/cacti.cfg", 10),
    )
    architecture = ArchitectureSpec(
        "test",
        1_000_000_000,
        placement,
        2,
        ArraySpec(2, 8, 8, "ws"),
        SRAMSpec(1, 8, 1 << 20, 64, 1, 1),
        VerticalLinkSpec(2, 1, 64),
        BufferSpec(1 << 20, 2, 2),
        VectorSpec(16, 64, 1, 1, 2, 1, 2, 1, 2, 1),
        ReductionSpec(1, 16, 1),
        NoCSpec(8, 16, 6, 7, (0, 1), 0, 1, 2, 3, 4),
        DRAMSpec(0, 1, 64, 0),
    )
    return CModelConfig(
        architecture,
        MappingSpec(4, 4, 4, partition),
        AddressSpec(0x10000000, 0x20000000, 0x30000000, 0x40000000, 0x50000000, 1 << 27, 64),
        backends,
        SimulationSpec(2_000_000),
    )


def macro() -> SRAMMacroResult:
    return SRAMMacroResult(1.0, 1.0, 2.0, 3.0, 0.1, 0.01)


@pytest.mark.parametrize(
    "operation",
    [
        GemmOp("g", 4, 4, 4, 8, 8, 32, 32),
        Conv2dOp("c", 1, 2, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 8, 8, 32, 32),
        SoftmaxOp("s", 2, 4, 32),
        LayerNormOp("l", 2, 4, 32, 32, 1e-5),
        AttentionOp("a", 1, 2, 2, 4, 2, 2, AttentionMode.PREFILL, True, 8, 8, 32, 32),
        AttentionOp("d", 1, 1, 3, 4, 2, 2, AttentionMode.DECODE, True, 8, 8, 32, 32),
    ],
)
def test_all_operator_lowerings_schedule(operation) -> None:
    cfg = config()
    lowerer = ArchitectureLowerer(cfg, FakeScaleSim(), macro())
    program = lowerer.lower(operation)
    result = CycleScheduler(cfg, macro(), FakeRamulator(), FakeBookSim()).run(program)
    assert result.total_cycles > 0
    assert result.counters["dram_requests_issued"] == result.counters["dram_requests_completed"]
    assert result.counters["noc_packets_issued"] == result.counters["noc_packets_completed"]


def test_bank_local_k_reduces_before_writing_output() -> None:
    cfg = config("bank_local_logic", "K")
    program = ArchitectureLowerer(cfg, FakeScaleSim(), macro()).lower(
        GemmOp("gk", 4, 4, 8, 8, 8, 32, 32)
    )
    reduce_uids = {op.uid for op in program.operations if op.name.startswith("gk.reduce.")}
    assert reduce_uids
    reduced_writes = [op for op in program.operations if ".reduced." in op.name and op.kind.value == "sram_write"]
    assert reduced_writes
    for write in reduced_writes:
        assert write.deps
    assert not any(op.name == "compute.output.sram" for op in program.operations)
    partial_noc = [op for op in program.operations if ".partial." in op.name and op.kind.value == "noc_send"]
    assert partial_noc
    assert all(op.metadata["vc"] == cfg.architecture.noc.vc_partial_sum for op in partial_noc)


def test_layernorm_parameters_precede_affine() -> None:
    cfg = config()
    program = ArchitectureLowerer(cfg, FakeScaleSim(), macro()).lower(
        LayerNormOp("ln", 2, 4, 32, 32, 1e-5)
    )
    params = next(op for op in program.operations if op.name == "ln.params")
    affine = next(op for op in program.operations if op.name == "ln.affine")
    dependency_closure = set(affine.deps)
    frontier = list(affine.deps)
    by_uid = {op.uid: op for op in program.operations}
    while frontier:
        uid = frontier.pop()
        for dep in by_uid[uid].deps:
            if dep not in dependency_closure:
                dependency_closure.add(dep)
                frontier.append(dep)
    assert params.uid in dependency_closure


def test_prefill_requires_matching_query_and_kv_lengths() -> None:
    with pytest.raises(Exception):
        AttentionOp("bad", 1, 2, 3, 4, 2, 2, AttentionMode.PREFILL, True, 8, 8, 32, 32)


def test_source_has_no_forbidden_config_access_patterns() -> None:
    root = Path(__file__).parents[1]
    paths = [root / "src" / "cmodel", root / "cmodel_main.py", root / "compare_cmodel.py"]
    forbidden = ("getattr(", ".get(", "try:", "except ")
    for path in paths:
        files = tuple(path.rglob("*.py")) if path.is_dir() else (path,)
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{token} found in {file_path}"


def test_committed_configs_are_strict_and_comparable() -> None:
    from src.cmodel.config import load_cmodel_config

    root = Path(__file__).parents[1]
    centralized = load_cmodel_config(root / "configs" / "cmodel_centralized.yaml")
    bank_local = load_cmodel_config(root / "configs" / "cmodel_bank_local.yaml")
    assert centralized.architecture.compute_placement == "centralized_logic"
    assert bank_local.architecture.compute_placement == "bank_local_logic"
    assert centralized.mapping == bank_local.mapping
    assert centralized.addresses == bank_local.addresses
    assert centralized.backends == bank_local.backends
    assert centralized.architecture.arrays == bank_local.architecture.arrays
    assert centralized.architecture.sram == bank_local.architecture.sram
    assert centralized.architecture.noc == bank_local.architecture.noc


def test_vector_buffer_overflow_fails_instead_of_spilling() -> None:
    from src.cmodel.errors import MappingError

    cfg = config()
    tiny = replace(cfg.architecture, global_buffer=BufferSpec(64, 1, 1))
    strict = replace(cfg, architecture=tiny)
    with pytest.raises(MappingError):
        ArchitectureLowerer(strict, FakeScaleSim(), macro()).lower(SoftmaxOp("too_big", 4, 8, 32))
