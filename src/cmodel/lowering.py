"""Architecture-aware lowering from GEMM to an explicit micro-operation DAG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .architecture import ArchitectureSpec, ComputePlacement
from .backends.base import SystolicBackend
from .errors import BackendOutputError, MappingError
from .ir import (
    GemmOp,
    MappingSpec,
    MicroOpKind,
    MicroProgram,
    MicroProgramBuilder,
    Operand,
    OperandDemand,
    SystolicInvocation,
    SystolicResult,
)

_INPUT_BASE = 0x1000_0000
_WEIGHT_BASE = 0x2000_0000
_OUTPUT_BASE = 0x3000_0000


@dataclass(frozen=True, slots=True)
class _Tile:
    m0: int
    n0: int
    k0: int
    m: int
    n: int
    k: int


class ArchitectureLowerer:
    """Use one systolic backend while changing only placement/data paths."""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        mapping: MappingSpec,
        backend: SystolicBackend,
    ) -> None:
        self.arch = architecture
        self.mapping = mapping
        self.backend = backend
        self.builder = MicroProgramBuilder()
        self._backend_manifests: set[tuple[str, str]] = set()

    def lower_gemm(self, op: GemmOp) -> MicroProgram:
        self.builder = MicroProgramBuilder()
        self._backend_manifests = set()
        self._validate_residency(op)
        root = self._barrier(f"{op.op_id}.start", ())
        if self.arch.placement is ComputePlacement.CENTRALIZED_LOGIC:
            terminals, runs = self._centralized(op, root)
        elif self.mapping.partition_dim == "N":
            terminals, runs = self._bank_local_n(op, root)
        else:
            terminals, runs = self._bank_local_k(op, root)
        final = self._barrier(f"{op.op_id}.complete", terminals)
        return self.builder.build(
            (final,),
            {
                "op_id": op.op_id,
                "placement": self.arch.placement.value,
                "mac_count": op.mac_count,
                "backend_runs": runs,
                "backend_manifests": [
                    {"name": name, "version": version}
                    for name, version in sorted(self._backend_manifests)
                ],
            },
        )

    def _centralized(self, op: GemmOp, root: int) -> tuple[list[int], int]:
        terminals: list[int] = []
        runs = 0
        array_cursor = 0
        for m0 in range(0, op.m, self.mapping.tile_m):
            m = min(self.mapping.tile_m, op.m - m0)
            for n0 in range(0, op.n, self.mapping.tile_n):
                n = min(self.mapping.tile_n, op.n - n0)
                previous = root
                for k0 in range(0, op.k, self.mapping.tile_k):
                    k = min(self.mapping.tile_k, op.k - k0)
                    tile = _Tile(m0, n0, k0, m, n, k)
                    array_id = array_cursor % self.arch.arrays.count
                    array_cursor += 1
                    result = self._invoke(op, tile, f"c_a{array_id}")
                    runs += 1
                    start = self._barrier(
                        f"{op.op_id}.central.m{m0}.n{n0}.k{k0}.start",
                        (previous,),
                    )
                    previous = self._execution(
                        op=op,
                        tile=tile,
                        result=result,
                        array_id=array_id,
                        start=start,
                        read_path=lambda demand, address: self._central_read(
                            op.op_id, demand, address, start
                        ),
                        output_path=lambda demand, address, source: self._central_output(
                            op.op_id,
                            demand,
                            address,
                            source,
                            final=(k0 + k == op.k),
                        ),
                    )
                terminals.append(previous)
        return terminals, runs

    def _bank_local_n(self, op: GemmOp, root: int) -> tuple[list[int], int]:
        terminals: list[int] = []
        runs = 0
        for m0 in range(0, op.m, self.mapping.tile_m):
            m = min(self.mapping.tile_m, op.m - m0)
            for n0 in range(0, op.n, self.mapping.tile_n):
                n_total = min(self.mapping.tile_n, op.n - n0)
                previous = root
                for k0 in range(0, op.k, self.mapping.tile_k):
                    k = min(self.mapping.tile_k, op.k - k0)
                    start = self._barrier(
                        f"{op.op_id}.npart.m{m0}.n{n0}.k{k0}.start",
                        (previous,),
                    )
                    shared_input: dict[tuple[int, int, int], int] = {}
                    group_terms: list[int] = []
                    local_n0 = n0
                    for group, n_local in enumerate(
                        self._split(n_total, self.arch.bank_groups)
                    ):
                        if n_local == 0:
                            continue
                        tile = _Tile(m0, local_n0, k0, m, n_local, k)
                        result = self._invoke(op, tile, f"npart_g{group}")
                        runs += 1

                        def read_path(
                            demand: OperandDemand,
                            address: int,
                            *,
                            group_id: int = group,
                        ) -> int:
                            if demand.operand is Operand.INPUT:
                                key = (demand.cycle, address, demand.nbytes)
                                multicast = shared_input.get(key)
                                if multicast is None:
                                    multicast = self._multicast_source(
                                        op.op_id, demand, address, start
                                    )
                                    shared_input[key] = multicast
                                return self._local_delivery(
                                    op.op_id,
                                    group_id,
                                    address,
                                    demand.nbytes,
                                    multicast,
                                )
                            return self._local_read(
                                op.op_id, group_id, demand, address, start
                            )

                        group_terms.append(
                            self._execution(
                                op=op,
                                tile=tile,
                                result=result,
                                array_id=group,
                                start=start,
                                read_path=read_path,
                                output_path=lambda demand, address, source, group_id=group: self._local_output(
                                    op.op_id,
                                    group_id,
                                    demand,
                                    address,
                                    source,
                                    final=(k0 + k == op.k),
                                ),
                            )
                        )
                        local_n0 += n_local
                    if not group_terms:
                        raise MappingError("N partition produced no active bank groups")
                    previous = self._barrier(
                        f"{op.op_id}.npart.m{m0}.n{n0}.k{k0}.join",
                        group_terms,
                    )
                terminals.append(previous)
        return terminals, runs

    def _bank_local_k(self, op: GemmOp, root: int) -> tuple[list[int], int]:
        terminals: list[int] = []
        runs = 0
        for m0 in range(0, op.m, self.mapping.tile_m):
            m = min(self.mapping.tile_m, op.m - m0)
            for n0 in range(0, op.n, self.mapping.tile_n):
                n = min(self.mapping.tile_n, op.n - n0)
                previous = root
                for k0 in range(0, op.k, self.mapping.tile_k):
                    k_total = min(self.mapping.tile_k, op.k - k0)
                    start = self._barrier(
                        f"{op.op_id}.kpart.m{m0}.n{n0}.k{k0}.start",
                        (previous,),
                    )
                    group_terms: list[int] = []
                    local_k0 = k0
                    for group, k_local in enumerate(
                        self._split(k_total, self.arch.bank_groups)
                    ):
                        if k_local == 0:
                            continue
                        tile = _Tile(m0, n0, local_k0, m, n, k_local)
                        result = self._invoke(op, tile, f"kpart_g{group}")
                        runs += 1
                        group_terms.append(
                            self._execution(
                                op=op,
                                tile=tile,
                                result=result,
                                array_id=group,
                                start=start,
                                read_path=lambda demand, address, group_id=group: self._local_read(
                                    op.op_id, group_id, demand, address, start
                                ),
                                output_path=lambda demand, address, source, group_id=group: self._local_output(
                                    op.op_id,
                                    group_id,
                                    demand,
                                    address,
                                    source,
                                    final=False,
                                ),
                            )
                        )
                        local_k0 += k_local
                    if not group_terms:
                        raise MappingError("K partition produced no active bank groups")
                    ready = self._barrier(
                        f"{op.op_id}.kpart.m{m0}.n{n0}.k{k0}.partials",
                        group_terms,
                    )
                    reduced = self._reduce(
                        op.op_id,
                        m,
                        n,
                        len(group_terms),
                        ready,
                    )
                    previous = (
                        self._reduced_write(op, m0, n0, m, n, reduced)
                        if k0 + k_total == op.k
                        else reduced
                    )
                terminals.append(previous)
        return terminals, runs

    def _execution(
        self,
        *,
        op: GemmOp,
        tile: _Tile,
        result: SystolicResult,
        array_id: int,
        start: int,
        read_path: Callable[[OperandDemand, int], int],
        output_path: Callable[[OperandDemand, int, int], int],
    ) -> int:
        feed: dict[int, list[int]] = {}
        outputs: dict[int, list[tuple[OperandDemand, int]]] = {}
        for demand in result.demands:
            address = self._address(op, tile, demand)
            if demand.operand in {Operand.INPUT, Operand.WEIGHT}:
                feed.setdefault(demand.cycle, []).append(read_path(demand, address))
            else:
                outputs.setdefault(demand.cycle, []).append((demand, address))
        steps = self._array_steps(
            f"{op.op_id}.m{tile.m0}.n{tile.n0}.k{tile.k0}",
            array_id,
            result.total_cycles,
            start,
            feed,
        )
        terminals: list[int] = []
        for cycle, demands in sorted(outputs.items()):
            source = steps[min(cycle, len(steps) - 1)]
            terminals.extend(
                output_path(demand, address, source)
                for demand, address in demands
            )
        return self._barrier(
            f"{op.op_id}.m{tile.m0}.n{tile.n0}.k{tile.k0}.done",
            terminals or (steps[-1],),
        )

    def _array_steps(
        self,
        name: str,
        array_id: int,
        total_cycles: int,
        start: int,
        feed: dict[int, list[int]],
    ) -> list[int]:
        if total_cycles <= 0:
            raise BackendOutputError("backend returned no array cycles")
        previous = start
        steps: list[int] = []
        for cycle in range(total_cycles):
            step = self.builder.add(
                name=f"{name}.cycle{cycle}",
                kind=MicroOpKind.ARRAY_STEP,
                resource_id=f"array.{array_id}",
                duration_cycles=1,
                deps=(previous, *feed.get(cycle, ())),
                release_anchor=start,
                release_offset_cycles=cycle,
                metadata={"nominal_cycle": cycle},
            )
            steps.append(step)
            previous = step
        return steps

    def _central_read(
        self, op_id: str, demand: OperandDemand, address: int, start: int
    ) -> int:
        bank = self.arch.sram.bank_for_address(address)
        read = self._transfer(
            f"{op_id}.{demand.operand.value}.sram_read",
            MicroOpKind.SRAM_READ,
            f"sram.bank.{bank}.read",
            demand.nbytes,
            (),
            start,
            self._release(demand.cycle),
            address,
        )
        link = self._transfer(
            f"{op_id}.{demand.operand.value}.vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{bank % self.arch.vertical_link.groups}",
            demand.nbytes,
            (read,),
        )
        write = self._transfer(
            f"{op_id}.{demand.operand.value}.gbuf_write",
            MicroOpKind.BUFFER_WRITE,
            "gbuf.write",
            demand.nbytes,
            (link,),
        )
        return self._transfer(
            f"{op_id}.{demand.operand.value}.gbuf_read",
            MicroOpKind.BUFFER_READ,
            "gbuf.read",
            demand.nbytes,
            (write,),
        )

    def _central_output(
        self,
        op_id: str,
        demand: OperandDemand,
        address: int,
        source: int,
        *,
        final: bool,
    ) -> int:
        write = self._transfer(
            f"{op_id}.psum_write",
            MicroOpKind.BUFFER_WRITE,
            "gbuf.write",
            demand.nbytes,
            (source,),
            address=address,
        )
        if not final:
            return write
        bank = self.arch.sram.bank_for_address(address)
        read = self._transfer(
            f"{op_id}.output.gbuf_read",
            MicroOpKind.BUFFER_READ,
            "gbuf.read",
            demand.nbytes,
            (write,),
        )
        link = self._transfer(
            f"{op_id}.output.vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{bank % self.arch.vertical_link.groups}",
            demand.nbytes,
            (read,),
        )
        return self._transfer(
            f"{op_id}.output.sram_write",
            MicroOpKind.SRAM_WRITE,
            f"sram.bank.{bank}.write",
            demand.nbytes,
            (link,),
            address=address,
        )

    def _local_read(
        self,
        op_id: str,
        group: int,
        demand: OperandDemand,
        address: int,
        start: int,
    ) -> int:
        bank = self.arch.local_bank_for_address(address, group)
        read = self._transfer(
            f"{op_id}.g{group}.sram_read",
            MicroOpKind.SRAM_READ,
            f"sram.bank.{bank}.read",
            demand.nbytes,
            (),
            start,
            self._release(demand.cycle),
            address,
        )
        link = self._transfer(
            f"{op_id}.g{group}.vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{group % self.arch.vertical_link.groups}",
            demand.nbytes,
            (read,),
        )
        write = self._transfer(
            f"{op_id}.g{group}.lbuf_write",
            MicroOpKind.BUFFER_WRITE,
            f"lbuf.{group}.write",
            demand.nbytes,
            (link,),
        )
        return self._transfer(
            f"{op_id}.g{group}.lbuf_read",
            MicroOpKind.BUFFER_READ,
            f"lbuf.{group}.read",
            demand.nbytes,
            (write,),
        )

    def _multicast_source(
        self, op_id: str, demand: OperandDemand, address: int, start: int
    ) -> int:
        bank = self.arch.sram.bank_for_address(address)
        read = self._transfer(
            f"{op_id}.multicast.sram_read",
            MicroOpKind.SRAM_READ,
            f"sram.bank.{bank}.read",
            demand.nbytes,
            (),
            start,
            self._release(demand.cycle),
            address,
        )
        link = self._transfer(
            f"{op_id}.multicast.vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{bank % self.arch.vertical_link.groups}",
            demand.nbytes,
            (read,),
        )
        return self._transfer(
            f"{op_id}.multicast.noc",
            MicroOpKind.NOC_SEND,
            f"noc.{bank % self.arch.noc.groups}",
            demand.nbytes,
            (link,),
        )

    def _local_delivery(
        self,
        op_id: str,
        group: int,
        address: int,
        nbytes: int,
        source: int,
    ) -> int:
        write = self._transfer(
            f"{op_id}.g{group}.multicast_lbuf_write",
            MicroOpKind.BUFFER_WRITE,
            f"lbuf.{group}.write",
            nbytes,
            (source,),
            address=address,
        )
        return self._transfer(
            f"{op_id}.g{group}.multicast_lbuf_read",
            MicroOpKind.BUFFER_READ,
            f"lbuf.{group}.read",
            nbytes,
            (write,),
        )

    def _local_output(
        self,
        op_id: str,
        group: int,
        demand: OperandDemand,
        address: int,
        source: int,
        *,
        final: bool,
    ) -> int:
        write = self._transfer(
            f"{op_id}.g{group}.local_output",
            MicroOpKind.BUFFER_WRITE,
            f"lbuf.{group}.write",
            demand.nbytes,
            (source,),
            address=address,
        )
        if not final:
            return write
        bank = self.arch.local_bank_for_address(address, group)
        read = self._transfer(
            f"{op_id}.g{group}.output_lbuf_read",
            MicroOpKind.BUFFER_READ,
            f"lbuf.{group}.read",
            demand.nbytes,
            (write,),
        )
        link = self._transfer(
            f"{op_id}.g{group}.output_vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{group % self.arch.vertical_link.groups}",
            demand.nbytes,
            (read,),
        )
        return self._transfer(
            f"{op_id}.g{group}.output_sram_write",
            MicroOpKind.SRAM_WRITE,
            f"sram.bank.{bank}.write",
            demand.nbytes,
            (link,),
            address=address,
        )

    def _reduce(
        self, op_id: str, m: int, n: int, partials: int, ready: int
    ) -> int:
        if partials <= 1:
            return ready
        nbytes = m * n * 4
        previous = ready
        for level in range(math.ceil(math.log2(partials))):
            noc = self._transfer(
                f"{op_id}.reduce.level{level}.noc",
                MicroOpKind.NOC_SEND,
                f"noc.{level % self.arch.noc.groups}",
                nbytes,
                (previous,),
            )
            previous = self._transfer(
                f"{op_id}.reduce.level{level}",
                MicroOpKind.GLOBAL_REDUCE,
                f"reduce.{level % self.arch.reduction.units}",
                nbytes,
                (noc,),
            )
        return previous

    def _reduced_write(
        self,
        op: GemmOp,
        m0: int,
        n0: int,
        m: int,
        n: int,
        source: int,
    ) -> int:
        address = _OUTPUT_BASE + (m0 * op.n + n0) * (op.accumulator_bits // 8)
        nbytes = m * n * (op.accumulator_bits // 8)
        bank = self.arch.sram.bank_for_address(address)
        link = self._transfer(
            f"{op.op_id}.reduced.vlink",
            MicroOpKind.VLINK_SEND,
            f"vlink.{bank % self.arch.vertical_link.groups}",
            nbytes,
            (source,),
        )
        return self._transfer(
            f"{op.op_id}.reduced.sram_write",
            MicroOpKind.SRAM_WRITE,
            f"sram.bank.{bank}.write",
            nbytes,
            (link,),
            address=address,
        )

    def _invoke(self, op: GemmOp, tile: _Tile, suffix: str) -> SystolicResult:
        self._validate_tile_capacity(op, tile)
        invocation = SystolicInvocation(
            op_id=(
                f"{op.op_id}_{suffix}_m{tile.m0}_{tile.m}_n{tile.n0}_{tile.n}"
                f"_k{tile.k0}_{tile.k}"
            ),
            m=tile.m,
            n=tile.n,
            k=tile.k,
            array_rows=self.arch.arrays.rows,
            array_cols=self.arch.arrays.cols,
            dataflow=self.arch.arrays.dataflow,
            input_bits=op.input_bits,
            weight_bits=op.weight_bits,
            accumulator_bits=op.accumulator_bits,
        )
        result = self.backend.run(invocation)
        self._backend_manifests.add((result.backend_name, result.backend_version))
        for demand in result.demands:
            if (
                demand.operand in {Operand.INPUT, Operand.WEIGHT}
                and demand.cycle >= result.total_cycles
            ):
                raise BackendOutputError(
                    f"backend demand cycle {demand.cycle} exceeds compute window "
                    f"[0, {result.total_cycles}) for {invocation.op_id}"
                )
        return result

    def _address(self, op: GemmOp, tile: _Tile, demand: OperandDemand) -> int:
        offset = self._local_offset(demand)
        if demand.operand is Operand.INPUT:
            width = op.input_bits // 8
            row, col = divmod(offset // width, tile.k)
            return _INPUT_BASE + (
                ((tile.m0 + row % tile.m) * op.k + tile.k0 + col % tile.k) * width
            )
        if demand.operand is Operand.WEIGHT:
            width = op.weight_bits // 8
            row, col = divmod(offset // width, tile.n)
            return _WEIGHT_BASE + (
                ((tile.k0 + row % tile.k) * op.n + tile.n0 + col % tile.n) * width
            )
        width = op.accumulator_bits // 8
        row, col = divmod(offset // width, tile.n)
        return _OUTPUT_BASE + (
            ((tile.m0 + row % tile.m) * op.n + tile.n0 + col % tile.n) * width
        )

    def _transfer(
        self,
        name: str,
        kind: MicroOpKind,
        resource: str,
        nbytes: int,
        deps: Sequence[int],
        anchor: int | None = None,
        offset: int = 0,
        address: int | None = None,
    ) -> int:
        metadata = {} if address is None else {"address": address}
        return self.builder.add(
            name=name,
            kind=kind,
            resource_id=resource,
            duration_cycles=0,
            deps=deps,
            release_anchor=anchor,
            release_offset_cycles=offset,
            nbytes=nbytes,
            metadata=metadata,
        )

    def _barrier(self, name: str, deps: Sequence[int]) -> int:
        return self.builder.add(
            name=name,
            kind=MicroOpKind.BARRIER,
            resource_id=None,
            duration_cycles=0,
            deps=deps,
        )

    def _release(self, demand_cycle: int) -> int:
        return max(0, demand_cycle - self.mapping.prefetch_distance_cycles)

    def _validate_residency(self, op: GemmOp) -> None:
        required = (
            op.m * op.k * (op.input_bits // 8)
            + op.k * op.n * (op.weight_bits // 8)
            + op.m * op.n * (op.accumulator_bits // 8)
        )
        if required > self.arch.sram.total_capacity_bytes:
            raise MappingError(
                "workload tensors exceed declared stacked-SRAM capacity and the "
                "strict C-model has no DRAM/spill fallback: "
                f"required={required}, capacity={self.arch.sram.total_capacity_bytes}"
            )

    def _validate_tile_capacity(self, op: GemmOp, tile: _Tile) -> None:
        working_set = (
            tile.m * tile.k * (op.input_bits // 8)
            + tile.k * tile.n * (op.weight_bits // 8)
            + tile.m * tile.n * (op.accumulator_bits // 8)
        )
        if working_set > self.arch.global_buffer.capacity_bytes:
            name = (
                "global buffer"
                if self.arch.placement is ComputePlacement.CENTRALIZED_LOGIC
                else "per-group local buffer"
            )
            raise MappingError(
                f"tile working set {working_set} exceeds declared {name} capacity "
                f"{self.arch.global_buffer.capacity_bytes}; strict mode does not "
                "retile automatically"
            )

    @staticmethod
    def _local_offset(demand: OperandDemand) -> int:
        base = {
            Operand.INPUT: _INPUT_BASE,
            Operand.WEIGHT: _WEIGHT_BASE,
            Operand.OUTPUT: _OUTPUT_BASE,
        }[demand.operand]
        return demand.address - base if demand.address >= base else demand.address

    @staticmethod
    def _split(extent: int, parts: int) -> list[int]:
        base, remainder = divmod(extent, parts)
        return [base + (1 if index < remainder else 0) for index in range(parts)]
