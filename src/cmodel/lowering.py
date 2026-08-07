"""Architecture-aware lowering for GEMM, CONV, Attention, Softmax, and LayerNorm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .backends.base import SRAMMacroResult, ScaleSimResult, TensorDemand
from .config import CModelConfig
from .errors import MappingError
from .ir import AddressAllocator, MicroOpKind, MicroProgram, ProgramBuilder, Tensor
from .operators import AttentionMode, AttentionOp, Conv2dOp, GemmOp, LayerNormOp, Operator, SoftmaxOp


@dataclass(frozen=True, slots=True)
class MacroTiming:
    access_cycles: int
    issue_cycles: int
    read_energy_pj: float
    write_energy_pj: float
    leakage_mw: float
    area_mm2: float

    @classmethod
    def from_result(cls, result: SRAMMacroResult, frequency_hz: int) -> "MacroTiming":
        access = max(1, math.ceil(result.access_time_ns * frequency_hz / 1e9))
        issue = max(1, math.ceil(result.cycle_time_ns * frequency_hz / 1e9))
        return cls(access, issue, result.read_energy_pj, result.write_energy_pj, result.leakage_mw, result.area_mm2)


@dataclass(frozen=True, slots=True)
class _Tile:
    m0: int
    n0: int
    k0: int
    m: int
    n: int
    k: int


class ArchitectureLowerer:
    def __init__(self, config: CModelConfig, scalesim, macro: SRAMMacroResult):
        self.config = config
        self.arch = config.architecture
        self.mapping = config.mapping
        self.scalesim = scalesim
        self.macro = MacroTiming.from_result(macro, self.arch.frequency_hz)
        self.builder = ProgramBuilder()
        self.allocator = AddressAllocator(config.addresses, self.builder)
        self.backend_runs = 0

    def lower(self, operation: Operator) -> MicroProgram:
        self.builder = ProgramBuilder()
        self.allocator = AddressAllocator(self.config.addresses, self.builder)
        self.backend_runs = 0
        root = self.builder.barrier(f"{operation.op_id}.start", ())
        if isinstance(operation, GemmOp):
            terminal = self._lower_standalone_gemm(operation, root)
        elif isinstance(operation, Conv2dOp):
            terminal = self._lower_conv(operation, root)
        elif isinstance(operation, SoftmaxOp):
            terminal = self._lower_softmax(operation, root)
        elif isinstance(operation, LayerNormOp):
            terminal = self._lower_layernorm(operation, root)
        elif isinstance(operation, AttentionOp):
            terminal = self._lower_attention(operation, root)
        else:
            raise MappingError(f"unsupported operator type {type(operation).__name__}")
        complete = self.builder.barrier(f"{operation.op_id}.complete", (terminal,))
        resident_bytes = sum(tensor.nbytes for tensor in self.builder.tensors)
        if resident_bytes > self.arch.sram.total_capacity_bytes:
            raise MappingError(
                f"strict residency exceeds stacked SRAM capacity: {resident_bytes} > {self.arch.sram.total_capacity_bytes}"
            )
        return self.builder.build(
            (complete,),
            {
                "operator": type(operation).__name__,
                "backend_runs": self.backend_runs,
                "resident_bytes": resident_bytes,
                "macro": {
                    "access_cycles": self.macro.access_cycles,
                    "issue_cycles": self.macro.issue_cycles,
                    "read_energy_pj": self.macro.read_energy_pj,
                    "write_energy_pj": self.macro.write_energy_pj,
                    "leakage_mw": self.macro.leakage_mw,
                    "area_mm2": self.macro.area_mm2,
                },
            },
        )

    def _lower_standalone_gemm(self, op: GemmOp, root: int) -> int:
        input_tensor = self.allocator.allocate(
            f"{op.op_id}.A", op.m * op.k * (op.input_bits // 8), "input"
        )
        weight_tensor = self.allocator.allocate(
            f"{op.op_id}.B", op.k * op.n * (op.weight_bits // 8), "weight"
        )
        output_tensor = self.allocator.allocate(
            f"{op.op_id}.C", op.m * op.n * (op.output_bits // 8), "output"
        )
        loaded_a = self._load_tensor(input_tensor, root, "activation", None)
        loaded_b = self._load_tensor(weight_tensor, root, "weight", None)
        ready = self.builder.barrier(f"{op.op_id}.operands_ready", (loaded_a, loaded_b))
        computed = self._gemm(
            op_id=op.op_id,
            m=op.m,
            n=op.n,
            k=op.k,
            input_bits=op.input_bits,
            weight_bits=op.weight_bits,
            output_bits=op.output_bits,
            input_tensor=input_tensor,
            weight_tensor=weight_tensor,
            output_tensor=output_tensor,
            ready=ready,
        )
        return self._store_tensor(output_tensor, computed, "output", None)

    def _lower_conv(self, op: Conv2dOp, root: int) -> int:
        input_bytes = op.batch * op.in_channels * op.input_h * op.input_w * (op.input_bits // 8)
        weight_bytes = (
            op.out_channels * (op.in_channels // op.groups) * op.kernel_h * op.kernel_w * (op.weight_bits // 8)
        )
        output_bytes = op.batch * op.out_channels * op.output_h * op.output_w * (op.output_bits // 8)
        input_tensor = self.allocator.allocate(f"{op.op_id}.input", input_bytes, "input")
        weight_tensor = self.allocator.allocate(f"{op.op_id}.weight", weight_bytes, "weight")
        output_tensor = self.allocator.allocate(f"{op.op_id}.output", output_bytes, "output")
        im2col_elems = op.batch * op.output_h * op.output_w * op.in_channels * op.kernel_h * op.kernel_w
        im2col = self.allocator.allocate(
            f"{op.op_id}.im2col", im2col_elems * (op.input_bits // 8), "temporary"
        )
        loaded_input = self._load_tensor(input_tensor, root, "activation", None)
        loaded_weight = self._load_tensor(weight_tensor, root, "weight", None)
        ready = self.builder.barrier(f"{op.op_id}.loaded", (loaded_input, loaded_weight))
        im2col_ready = self._vector_transform(
            f"{op.op_id}.im2col_transform",
            input_tensor,
            im2col,
            ready,
            elements=im2col_elems,
            cycles_per_element=self.arch.vector.affine_cycles_per_element,
        )
        group_terms: list[int] = []
        rows = op.batch * op.output_h * op.output_w
        in_per_group = op.in_channels // op.groups
        out_per_group = op.out_channels // op.groups
        k_per_group = in_per_group * op.kernel_h * op.kernel_w
        for group in range(op.groups):
            input_group = Tensor(
                f"{im2col.name}.g{group}",
                im2col.address + group * rows * k_per_group * (op.input_bits // 8),
                rows * k_per_group * (op.input_bits // 8),
                im2col.region,
            )
            weight_group = Tensor(
                f"{weight_tensor.name}.g{group}",
                weight_tensor.address + group * k_per_group * out_per_group * (op.weight_bits // 8),
                k_per_group * out_per_group * (op.weight_bits // 8),
                weight_tensor.region,
            )
            output_group = Tensor(
                f"{output_tensor.name}.g{group}",
                output_tensor.address + group * rows * out_per_group * (op.output_bits // 8),
                rows * out_per_group * (op.output_bits // 8),
                output_tensor.region,
            )
            group_terms.append(
                self._gemm(
                    op_id=f"{op.op_id}.g{group}",
                    m=rows,
                    n=out_per_group,
                    k=k_per_group,
                    input_bits=op.input_bits,
                    weight_bits=op.weight_bits,
                    output_bits=op.output_bits,
                    input_tensor=input_group,
                    weight_tensor=weight_group,
                    output_tensor=output_group,
                    ready=im2col_ready,
                )
            )
        computed = self.builder.barrier(f"{op.op_id}.groups_done", tuple(group_terms))
        return self._store_tensor(output_tensor, computed, "output", None)

    def _lower_softmax(self, op: SoftmaxOp, root: int) -> int:
        nbytes = op.rows * op.cols * (op.element_bits // 8)
        source = self.allocator.allocate(f"{op.op_id}.input", nbytes, "input")
        output = self.allocator.allocate(f"{op.op_id}.output", nbytes, "output")
        temp0 = self.allocator.allocate(f"{op.op_id}.shifted", nbytes, "temporary")
        temp1 = self.allocator.allocate(f"{op.op_id}.exp", nbytes, "temporary")
        loaded = self._load_tensor(source, root, "activation", None)
        maximum = self._reduce_tensor(
            f"{op.op_id}.max", source, loaded, op.rows, op.cols, self.arch.vector.max_cycles_per_element
        )
        shifted = self._vector_transform(
            f"{op.op_id}.sub", source, temp0, maximum, op.rows * op.cols, self.arch.vector.sub_cycles_per_element
        )
        exponent = self._vector_transform(
            f"{op.op_id}.exp", temp0, temp1, shifted, op.rows * op.cols, self.arch.vector.exp_cycles_per_element
        )
        summed = self._reduce_tensor(
            f"{op.op_id}.sum", temp1, exponent, op.rows, op.cols, self.arch.vector.add_cycles_per_element
        )
        reciprocal = self._vector_only(
            f"{op.op_id}.reciprocal", summed, op.rows, self.arch.vector.reciprocal_cycles_per_element
        )
        normalized = self._vector_transform(
            f"{op.op_id}.normalize", temp1, output, reciprocal, op.rows * op.cols, self.arch.vector.multiply_cycles_per_element
        )
        return self._store_tensor(output, normalized, "output", None)

    def _lower_layernorm(self, op: LayerNormOp, root: int) -> int:
        elem_bytes = op.element_bits // 8
        param_bytes = op.parameter_bits // 8
        source = self.allocator.allocate(f"{op.op_id}.input", op.rows * op.cols * elem_bytes, "input")
        gamma = self.allocator.allocate(f"{op.op_id}.gamma", op.cols * param_bytes, "weight")
        beta = self.allocator.allocate(f"{op.op_id}.beta", op.cols * param_bytes, "weight")
        centered = self.allocator.allocate(f"{op.op_id}.centered", op.rows * op.cols * elem_bytes, "temporary")
        output = self.allocator.allocate(f"{op.op_id}.output", op.rows * op.cols * elem_bytes, "output")
        loaded = self.builder.barrier(
            f"{op.op_id}.loaded",
            (
                self._load_tensor(source, root, "activation", None),
                self._load_tensor(gamma, root, "weight", None),
                self._load_tensor(beta, root, "weight", None),
            ),
        )
        mean = self._reduce_tensor(
            f"{op.op_id}.mean", source, loaded, op.rows, op.cols, self.arch.vector.add_cycles_per_element
        )
        center = self._vector_transform(
            f"{op.op_id}.center", source, centered, mean, op.rows * op.cols, self.arch.vector.sub_cycles_per_element
        )
        variance = self._reduce_tensor(
            f"{op.op_id}.variance", centered, center, op.rows, op.cols, self.arch.vector.multiply_cycles_per_element + self.arch.vector.add_cycles_per_element
        )
        inv_std = self._vector_only(
            f"{op.op_id}.rsqrt", variance, op.rows, self.arch.vector.rsqrt_cycles_per_element
        )
        param_reads = self.builder.barrier(
            f"{op.op_id}.params",
            (
                self._read_tensor_to_global_buffer(gamma, inv_std, "weight"),
                self._read_tensor_to_global_buffer(beta, inv_std, "weight"),
            ),
        )
        affine = self._vector_transform(
            f"{op.op_id}.affine", centered, output, param_reads, op.rows * op.cols, self.arch.vector.affine_cycles_per_element
        )
        done = self.builder.barrier(f"{op.op_id}.affine_done", (affine,))
        return self._store_tensor(output, done, "output", None)

    def _lower_attention(self, op: AttentionOp, root: int) -> int:
        in_bytes = op.input_bits // 8
        weight_bytes = op.weight_bits // 8
        act_bytes = op.activation_bits // 8
        tokens = op.batch * op.query_tokens
        source = self.allocator.allocate(f"{op.op_id}.input", tokens * op.model_dim * in_bytes, "input")
        q_weight = self.allocator.allocate(f"{op.op_id}.Wq", op.model_dim * op.model_dim * weight_bytes, "weight")
        k_weight = self.allocator.allocate(f"{op.op_id}.Wk", op.model_dim * op.model_dim * weight_bytes, "weight")
        v_weight = self.allocator.allocate(f"{op.op_id}.Wv", op.model_dim * op.model_dim * weight_bytes, "weight")
        o_weight = self.allocator.allocate(f"{op.op_id}.Wo", op.model_dim * op.model_dim * weight_bytes, "weight")
        q = self.allocator.allocate(f"{op.op_id}.Q", tokens * op.model_dim * act_bytes, "temporary")
        new_k = self.allocator.allocate(f"{op.op_id}.Knew", tokens * op.model_dim * act_bytes, "temporary")
        new_v = self.allocator.allocate(f"{op.op_id}.Vnew", tokens * op.model_dim * act_bytes, "temporary")
        kv_cache_k = self.allocator.allocate(
            f"{op.op_id}.Kcache", op.batch * op.kv_tokens * op.model_dim * act_bytes, "kv_cache"
        )
        kv_cache_v = self.allocator.allocate(
            f"{op.op_id}.Vcache", op.batch * op.kv_tokens * op.model_dim * act_bytes, "kv_cache"
        )
        score = self.allocator.allocate(
            f"{op.op_id}.score", op.batch * op.heads * op.query_tokens * op.kv_tokens * act_bytes, "temporary"
        )
        probability = self.allocator.allocate(
            f"{op.op_id}.probability", score.nbytes, "temporary"
        )
        context = self.allocator.allocate(
            f"{op.op_id}.context", tokens * op.model_dim * act_bytes, "temporary"
        )
        output = self.allocator.allocate(f"{op.op_id}.output", tokens * op.model_dim * act_bytes, "output")

        load_terms = [self._load_tensor(source, root, "activation", None)]
        for weight in (q_weight, k_weight, v_weight, o_weight):
            load_terms.append(self._load_tensor(weight, root, "weight", None))
        if op.mode is AttentionMode.DECODE:
            load_terms.append(self._load_tensor(kv_cache_k, root, "activation", None))
            load_terms.append(self._load_tensor(kv_cache_v, root, "activation", None))
        loaded = self.builder.barrier(f"{op.op_id}.loaded", tuple(load_terms))
        q_ready = self._projection(f"{op.op_id}.q", source, q_weight, q, op, loaded)
        k_ready = self._projection(f"{op.op_id}.k", source, k_weight, new_k, op, loaded)
        v_ready = self._projection(f"{op.op_id}.v", source, v_weight, new_v, op, loaded)
        projections = self.builder.barrier(f"{op.op_id}.qkv_ready", (q_ready, k_ready, v_ready))

        if op.mode is AttentionMode.PREFILL:
            cache_k_ready = self._copy_tensor(new_k, kv_cache_k, projections, "activation")
            cache_v_ready = self._copy_tensor(new_v, kv_cache_v, projections, "activation")
        else:
            cache_k_ready = self._append_tensor(new_k, kv_cache_k, projections, "activation")
            cache_v_ready = self._append_tensor(new_v, kv_cache_v, projections, "activation")
        cache_ready = self.builder.barrier(f"{op.op_id}.cache_ready", (cache_k_ready, cache_v_ready))

        qk = self._gemm(
            op_id=f"{op.op_id}.qk",
            m=op.batch * op.heads * op.query_tokens,
            n=op.kv_tokens,
            k=op.head_dim,
            input_bits=op.activation_bits,
            weight_bits=op.activation_bits,
            output_bits=op.activation_bits,
            input_tensor=q,
            weight_tensor=kv_cache_k,
            output_tensor=score,
            ready=cache_ready,
        )
        scaled = self._vector_inplace(
            f"{op.op_id}.scale", score, qk, op.batch * op.heads * op.query_tokens * op.kv_tokens, self.arch.vector.multiply_cycles_per_element
        )
        masked = scaled
        if op.causal:
            masked = self._vector_inplace(
                f"{op.op_id}.causal_mask", score, scaled, op.batch * op.heads * op.query_tokens * op.kv_tokens, self.arch.vector.affine_cycles_per_element
            )
        softmax_ready = self._softmax_tensor(
            f"{op.op_id}.softmax", score, probability, masked,
            op.batch * op.heads * op.query_tokens, op.kv_tokens,
        )
        pv = self._gemm(
            op_id=f"{op.op_id}.pv",
            m=op.batch * op.heads * op.query_tokens,
            n=op.head_dim,
            k=op.kv_tokens,
            input_bits=op.activation_bits,
            weight_bits=op.activation_bits,
            output_bits=op.activation_bits,
            input_tensor=probability,
            weight_tensor=kv_cache_v,
            output_tensor=context,
            ready=softmax_ready,
        )
        projected = self._gemm(
            op_id=f"{op.op_id}.out",
            m=tokens,
            n=op.model_dim,
            k=op.model_dim,
            input_bits=op.activation_bits,
            weight_bits=op.weight_bits,
            output_bits=op.activation_bits,
            input_tensor=context,
            weight_tensor=o_weight,
            output_tensor=output,
            ready=pv,
        )
        return self._store_tensor(output, projected, "output", None)

    def _projection(self, name: str, source: Tensor, weight: Tensor, output: Tensor, op: AttentionOp, ready: int) -> int:
        return self._gemm(
            op_id=name,
            m=op.batch * op.query_tokens,
            n=op.model_dim,
            k=op.model_dim,
            input_bits=op.input_bits,
            weight_bits=op.weight_bits,
            output_bits=op.activation_bits,
            input_tensor=source,
            weight_tensor=weight,
            output_tensor=output,
            ready=ready,
        )

    def _gemm(
        self,
        *,
        op_id: str,
        m: int,
        n: int,
        k: int,
        input_bits: int,
        weight_bits: int,
        output_bits: int,
        input_tensor: Tensor,
        weight_tensor: Tensor,
        output_tensor: Tensor,
        ready: int,
    ) -> int:
        if self.arch.compute_placement == "centralized_logic":
            return self._gemm_centralized(
                op_id, m, n, k, input_bits, weight_bits, output_bits,
                input_tensor, weight_tensor, output_tensor, ready,
            )
        if self.mapping.partition_dim == "N":
            return self._gemm_bank_local_n(
                op_id, m, n, k, input_bits, weight_bits, output_bits,
                input_tensor, weight_tensor, output_tensor, ready,
            )
        return self._gemm_bank_local_k(
            op_id, m, n, k, input_bits, weight_bits, output_bits,
            input_tensor, weight_tensor, output_tensor, ready,
        )

    def _gemm_centralized(self, op_id: str, m: int, n: int, k: int, input_bits: int, weight_bits: int, output_bits: int, input_tensor: Tensor, weight_tensor: Tensor, output_tensor: Tensor, ready: int) -> int:
        terminals: list[int] = []
        array_cursor = 0
        for m0 in range(0, m, self.mapping.tile_m):
            mt = min(self.mapping.tile_m, m - m0)
            for n0 in range(0, n, self.mapping.tile_n):
                nt = min(self.mapping.tile_n, n - n0)
                array_id = array_cursor % self.arch.arrays.count
                array_cursor += 1
                previous = ready
                for k0 in range(0, k, self.mapping.tile_k):
                    kt = min(self.mapping.tile_k, k - k0)
                    tile = _Tile(m0, n0, k0, mt, nt, kt)
                    final_k = k0 + kt == k
                    previous = self._run_tile(
                        op_id, m, n, k, input_bits, weight_bits, output_bits,
                        input_tensor, weight_tensor, output_tensor, tile, array_id, None,
                        previous, final_k,
                    )
                terminals.append(previous)
        return self.builder.barrier(f"{op_id}.gemm_done", tuple(terminals))

    def _gemm_bank_local_n(self, op_id: str, m: int, n: int, k: int, input_bits: int, weight_bits: int, output_bits: int, input_tensor: Tensor, weight_tensor: Tensor, output_tensor: Tensor, ready: int) -> int:
        group_terms: list[int] = []
        n_ranges = _split_ranges(n, self.arch.bank_groups)
        for group, (n0_group, n_group) in enumerate(n_ranges):
            if n_group == 0:
                continue
            group_ready = ready
            for m0 in range(0, m, self.mapping.tile_m):
                mt = min(self.mapping.tile_m, m - m0)
                for n_local in range(0, n_group, self.mapping.tile_n):
                    nt = min(self.mapping.tile_n, n_group - n_local)
                    previous = group_ready
                    for k0 in range(0, k, self.mapping.tile_k):
                        kt = min(self.mapping.tile_k, k - k0)
                        tile = _Tile(m0, n0_group + n_local, k0, mt, nt, kt)
                        previous = self._run_tile(
                            op_id, m, n, k, input_bits, weight_bits, output_bits,
                            input_tensor, weight_tensor, output_tensor, tile, group, group,
                            previous, k0 + kt == k,
                        )
                    group_terms.append(previous)
        if not group_terms:
            raise MappingError("N partition produced no active bank groups")
        return self.builder.barrier(f"{op_id}.n_partition_done", tuple(group_terms))

    def _gemm_bank_local_k(self, op_id: str, m: int, n: int, k: int, input_bits: int, weight_bits: int, output_bits: int, input_tensor: Tensor, weight_tensor: Tensor, output_tensor: Tensor, ready: int) -> int:
        terminals: list[int] = []
        for m0 in range(0, m, self.mapping.tile_m):
            mt = min(self.mapping.tile_m, m - m0)
            for n0 in range(0, n, self.mapping.tile_n):
                nt = min(self.mapping.tile_n, n - n0)
                partial_terms: list[int] = []
                for group, (k0_group, k_group) in enumerate(_split_ranges(k, self.arch.bank_groups)):
                    if k_group == 0:
                        continue
                    previous = ready
                    for k_local in range(0, k_group, self.mapping.tile_k):
                        kt = min(self.mapping.tile_k, k_group - k_local)
                        tile = _Tile(m0, n0, k0_group + k_local, mt, nt, kt)
                        previous = self._run_tile(
                            op_id, m, n, k, input_bits, weight_bits, output_bits,
                            input_tensor, weight_tensor, output_tensor, tile, group, group,
                            previous, False,
                        )
                    partial_terms.append(
                        self._partial_to_reducer(
                            op_id, m0, n0, group, mt * nt, output_bits, previous
                        )
                    )
                if not partial_terms:
                    raise MappingError("K partition produced no active bank groups")
                partials = self.builder.barrier(f"{op_id}.partials.m{m0}.n{n0}", tuple(partial_terms))
                elements = mt * nt
                stages = math.ceil(math.log2(len(partial_terms))) if len(partial_terms) > 1 else 1
                duration = stages * self.arch.reduction.cycles_per_stage * math.ceil(elements / self.arch.reduction.lanes_per_unit)
                reduced = self.builder.add(
                    name=f"{op_id}.reduce.m{m0}.n{n0}", kind=MicroOpKind.REDUCE,
                    deps=(partials,), resource_id="reduce.0", latency_cycles=max(1, duration),
                    issue_interval_cycles=max(1, duration), nbytes=elements * (output_bits // 8),
                    metadata={"partials": len(partial_terms)},
                )
                terminals.append(
                    self._write_reduced_tile(
                        op_id, output_tensor, m0, n0, mt, nt, n, output_bits, reduced
                    )
                )
        return self.builder.barrier(f"{op_id}.k_partition_done", tuple(terminals))

    def _partial_to_reducer(
        self,
        op_id: str,
        m0: int,
        n0: int,
        group: int,
        elements: int,
        output_bits: int,
        dep: int,
    ) -> int:
        nbytes = elements * (output_bits // 8)
        local = self._buffer_write(
            f"lbuf.{group}", f"{op_id}.partial.m{m0}.n{n0}.g{group}.local", dep, nbytes
        )
        noc = self._noc(
            f"{op_id}.partial.m{m0}.n{n0}.g{group}.noc",
            local,
            self.arch.noc.endpoint_for_bank_group[group],
            self.arch.noc.endpoint_for_global_buffer,
            self._vc("partial_sum"),
            nbytes,
        )
        return self._buffer_write(
            "gbuf", f"{op_id}.partial.m{m0}.n{n0}.g{group}.global", noc, nbytes
        )

    def _write_reduced_tile(
        self,
        op_id: str,
        output_tensor: Tensor,
        m0: int,
        n0: int,
        m: int,
        n: int,
        n_total: int,
        output_bits: int,
        dep: int,
    ) -> int:
        element_bytes = output_bits // 8
        terms: list[int] = []
        for row in range(m):
            nbytes = n * element_bytes
            address = output_tensor.address + ((m0 + row) * n_total + n0) * element_bytes
            buffered = self._buffer_write(
                "gbuf", f"{op_id}.reduced.m{m0}.n{n0}.row{row}.gbuf", dep, nbytes
            )
            bank = self._bank(address, None)
            bank_group = self._bank_group(bank)
            noc = self._noc(
                f"{op_id}.reduced.m{m0}.n{n0}.row{row}.noc",
                buffered,
                self.arch.noc.endpoint_for_global_buffer,
                self.arch.noc.endpoint_for_bank_group[bank_group],
                self._vc("output"),
                nbytes,
            )
            vlink = self._vlink(
                f"{op_id}.reduced.m{m0}.n{n0}.row{row}.vlink", noc, bank_group, nbytes
            )
            terms.append(
                self._sram_write(
                    f"{op_id}.reduced.m{m0}.n{n0}.row{row}.sram",
                    vlink, bank, address, nbytes,
                )
            )
        return self.builder.barrier(f"{op_id}.reduced.m{m0}.n{n0}.written", tuple(terms))

    def _run_tile(self, op_id: str, m_total: int, n_total: int, k_total: int, input_bits: int, weight_bits: int, output_bits: int, input_tensor: Tensor, weight_tensor: Tensor, output_tensor: Tensor, tile: _Tile, array_id: int, group: int | None, ready: int, write_output: bool) -> int:
        result: ScaleSimResult = self.scalesim.run_gemm(
            op_id=f"{op_id}.m{tile.m0}.n{tile.n0}.k{tile.k0}",
            m=tile.m, n=tile.n, k=tile.k,
            input_base=0, weight_base=0, output_base=0,
        )
        self.backend_runs += 1
        feed: dict[int, list[int]] = {}
        outputs: dict[int, list[tuple[int, int]]] = {}
        input_bytes = input_bits // 8
        weight_bytes = weight_bits // 8
        output_bytes = output_bits // 8
        for demand in result.demands:
            if demand.operand == "input":
                address = _translate_a(demand.address, input_bytes, tile, k_total, input_tensor.address)
                event = self._compute_read(address, input_bytes, ready, "activation", group)
                _append(feed, demand.cycle, event)
            elif demand.operand == "weight":
                address = _translate_b(demand.address, weight_bytes, tile, n_total, weight_tensor.address)
                event = self._compute_read(address, weight_bytes, ready, "weight", group)
                _append(feed, demand.cycle, event)
            elif demand.operand == "output" and write_output:
                address = _translate_c(demand.address, output_bytes, tile, n_total, output_tensor.address)
                _append_pair(outputs, demand.cycle, (address, output_bytes))
            elif demand.operand != "output":
                raise MappingError(f"unknown SCALE-Sim operand {demand.operand}")
        previous = ready
        steps: list[int] = []
        for cycle in range(result.cycles):
            deps = (previous, *tuple(feed[cycle])) if cycle in feed else (previous,)
            step = self.builder.add(
                name=f"{op_id}.array{array_id}.cycle{cycle}", kind=MicroOpKind.ARRAY_STEP,
                deps=deps, resource_id=f"array.{array_id}", latency_cycles=1,
                issue_interval_cycles=1, nbytes=0, metadata={"nominal_cycle": cycle},
            )
            steps.append(step)
            previous = step
        output_terms: list[int] = []
        if write_output:
            for cycle in sorted(outputs):
                source = steps[min(cycle, len(steps) - 1)]
                for address, nbytes in outputs[cycle]:
                    output_terms.append(self._compute_write(address, nbytes, source, "output", group))
        return self.builder.barrier(
            f"{op_id}.tile.m{tile.m0}.n{tile.n0}.k{tile.k0}.done",
            tuple(output_terms) if output_terms else (steps[-1],),
        )

    def _load_tensor(self, tensor: Tensor, dep: int, role: str, group: int | None) -> int:
        terms: list[int] = []
        transaction = self.arch.dram.transaction_bytes
        for offset in range(0, tensor.nbytes, transaction):
            nbytes = min(transaction, tensor.nbytes - offset)
            address = tensor.address + offset
            bank = self._bank(address, group)
            bank_group = self._bank_group(bank)
            dram = self.builder.add(
                name=f"{tensor.name}.dram_read.{offset}", kind=MicroOpKind.DRAM_READ,
                deps=(dep,), resource_id="dram", latency_cycles=0, issue_interval_cycles=0,
                nbytes=nbytes,
                metadata={
                    "request_type": self.arch.dram.read_request_type,
                    "address": address,
                    "source_id": self.arch.dram.source_id,
                },
            )
            noc = self._noc(
                f"{tensor.name}.load_noc.{offset}", dram,
                self.arch.noc.endpoint_for_dram, self.arch.noc.endpoint_for_bank_group[bank_group],
                self._vc(role), nbytes,
            )
            vlink = self._vlink(f"{tensor.name}.load_vlink.{offset}", noc, bank_group, nbytes)
            terms.append(self._sram_write(f"{tensor.name}.load_sram.{offset}", vlink, bank, address, nbytes))
        return self.builder.barrier(f"{tensor.name}.loaded", tuple(terms))

    def _store_tensor(self, tensor: Tensor, dep: int, role: str, group: int | None) -> int:
        terms: list[int] = []
        transaction = self.arch.dram.transaction_bytes
        for offset in range(0, tensor.nbytes, transaction):
            nbytes = min(transaction, tensor.nbytes - offset)
            address = tensor.address + offset
            bank = self._bank(address, group)
            bank_group = self._bank_group(bank)
            read = self._sram_read(f"{tensor.name}.store_sram.{offset}", dep, bank, address, nbytes)
            vlink = self._vlink(f"{tensor.name}.store_vlink.{offset}", read, bank_group, nbytes)
            noc = self._noc(
                f"{tensor.name}.store_noc.{offset}", vlink,
                self.arch.noc.endpoint_for_bank_group[bank_group], self.arch.noc.endpoint_for_dram,
                self._vc(role), nbytes,
            )
            terms.append(
                self.builder.add(
                    name=f"{tensor.name}.dram_write.{offset}", kind=MicroOpKind.DRAM_WRITE,
                    deps=(noc,), resource_id="dram", latency_cycles=0, issue_interval_cycles=0,
                    nbytes=nbytes,
                    metadata={
                        "request_type": self.arch.dram.write_request_type,
                        "address": address,
                        "source_id": self.arch.dram.source_id,
                    },
                )
            )
        return self.builder.barrier(f"{tensor.name}.stored", tuple(terms))

    def _compute_read(self, address: int, nbytes: int, dep: int, role: str, group: int | None) -> int:
        bank = self._bank(address, group if role == "weight" and group is not None else None)
        bank_group = self._bank_group(bank)
        read = self._sram_read(f"compute.{role}.sram", dep, bank, address, nbytes)
        vlink = self._vlink(f"compute.{role}.vlink", read, bank_group, nbytes)
        if self.arch.compute_placement == "centralized_logic":
            noc = self._noc(
                f"compute.{role}.noc", vlink,
                self.arch.noc.endpoint_for_bank_group[bank_group], self.arch.noc.endpoint_for_global_buffer,
                self._vc(role), nbytes,
            )
            write = self._buffer_write("gbuf", f"compute.{role}.gbuf_write", noc, nbytes)
            return self._buffer_read("gbuf", f"compute.{role}.gbuf_read", write, nbytes)
        if group is None:
            raise MappingError("bank-local compute read requires a bank group")
        delivered = vlink
        if bank_group != group:
            delivered = self._noc(
                f"compute.{role}.multicast", vlink,
                self.arch.noc.endpoint_for_bank_group[bank_group], self.arch.noc.endpoint_for_bank_group[group],
                self._vc(role), nbytes,
            )
        write = self._buffer_write(f"lbuf.{group}", f"compute.{role}.lbuf_write", delivered, nbytes)
        return self._buffer_read(f"lbuf.{group}", f"compute.{role}.lbuf_read", write, nbytes)

    def _compute_write(self, address: int, nbytes: int, dep: int, role: str, group: int | None) -> int:
        if self.arch.compute_placement == "centralized_logic":
            write = self._buffer_write("gbuf", "compute.output.gbuf_write", dep, nbytes)
            bank = self._bank(address, None)
            bank_group = self._bank_group(bank)
            noc = self._noc(
                "compute.output.noc", write,
                self.arch.noc.endpoint_for_global_buffer, self.arch.noc.endpoint_for_bank_group[bank_group],
                self._vc(role), nbytes,
            )
            vlink = self._vlink("compute.output.vlink", noc, bank_group, nbytes)
            return self._sram_write("compute.output.sram", vlink, bank, address, nbytes)
        if group is None:
            raise MappingError("bank-local compute write requires a bank group")
        write = self._buffer_write(f"lbuf.{group}", "compute.output.lbuf_write", dep, nbytes)
        bank = self._bank(address, group)
        vlink = self._vlink("compute.output.vlink", write, group, nbytes)
        return self._sram_write("compute.output.sram", vlink, bank, address, nbytes)

    def _vector_transform(self, name: str, source: Tensor, output: Tensor, dep: int, elements: int, cycles_per_element: int) -> int:
        loaded = self._read_tensor_to_global_buffer(source, dep, "activation")
        vector = self._vector_only(name, loaded, elements, cycles_per_element)
        return self._write_global_buffer_to_tensor(output, vector, "output")

    def _vector_inplace(self, name: str, tensor: Tensor, dep: int, elements: int, cycles_per_element: int) -> int:
        loaded = self._read_tensor_to_global_buffer(tensor, dep, "activation")
        vector = self._vector_only(name, loaded, elements, cycles_per_element)
        return self._write_global_buffer_to_tensor(tensor, vector, "output")

    def _reduce_tensor(self, name: str, tensor: Tensor, dep: int, rows: int, cols: int, cycles_per_element: int) -> int:
        loaded = self._read_tensor_to_global_buffer(tensor, dep, "activation")
        elements = rows * cols
        duration = max(1, math.ceil(elements / self.arch.reduction.lanes_per_unit) * cycles_per_element)
        return self.builder.add(
            name=name, kind=MicroOpKind.REDUCE, deps=(loaded,), resource_id="reduce.0",
            latency_cycles=duration, issue_interval_cycles=duration,
            nbytes=tensor.nbytes, metadata={"rows": rows, "cols": cols},
        )

    def _softmax_tensor(self, name: str, source: Tensor, output: Tensor, dep: int, rows: int, cols: int) -> int:
        maximum = self._reduce_tensor(f"{name}.max", source, dep, rows, cols, self.arch.vector.max_cycles_per_element)
        shifted = self._vector_only(f"{name}.sub", maximum, rows * cols, self.arch.vector.sub_cycles_per_element)
        exponent = self._vector_only(f"{name}.exp", shifted, rows * cols, self.arch.vector.exp_cycles_per_element)
        summed = self._vector_only(f"{name}.sum", exponent, rows * cols, self.arch.vector.add_cycles_per_element)
        reciprocal = self._vector_only(f"{name}.reciprocal", summed, rows, self.arch.vector.reciprocal_cycles_per_element)
        normalized = self._vector_only(f"{name}.normalize", reciprocal, rows * cols, self.arch.vector.multiply_cycles_per_element)
        return self._write_global_buffer_to_tensor(output, normalized, "output")

    def _vector_only(self, name: str, dep: int, elements: int, cycles_per_element: int) -> int:
        duration = max(1, math.ceil(elements / self.arch.vector.lanes) * cycles_per_element)
        return self.builder.add(
            name=name, kind=MicroOpKind.VECTOR, deps=(dep,), resource_id="vector.0",
            latency_cycles=duration, issue_interval_cycles=duration, nbytes=0,
            metadata={"elements": elements},
        )

    def _copy_tensor(self, source: Tensor, output: Tensor, dep: int, role: str) -> int:
        loaded = self._read_tensor_to_global_buffer(source, dep, role)
        return self._write_global_buffer_to_tensor(output, loaded, role)

    def _append_tensor(self, source: Tensor, cache: Tensor, dep: int, role: str) -> int:
        loaded = self._read_tensor_to_global_buffer(source, dep, role)
        append_bytes = source.nbytes
        address = cache.address + cache.nbytes - append_bytes
        view = Tensor(f"{cache.name}.append", address, append_bytes, cache.region)
        return self._write_global_buffer_to_tensor(view, loaded, role)

    def _read_tensor_to_global_buffer(self, tensor: Tensor, dep: int, role: str) -> int:
        if tensor.nbytes > self.arch.global_buffer.capacity_bytes:
            raise MappingError(
                f"tensor {tensor.name} requires {tensor.nbytes} bytes in the global buffer, "
                f"capacity is {self.arch.global_buffer.capacity_bytes}"
            )
        terms: list[int] = []
        for offset in range(0, tensor.nbytes, self.arch.sram.line_bytes):
            nbytes = min(self.arch.sram.line_bytes, tensor.nbytes - offset)
            address = tensor.address + offset
            bank = self._bank(address, None)
            group = self._bank_group(bank)
            read = self._sram_read(f"{tensor.name}.read.{offset}", dep, bank, address, nbytes)
            vlink = self._vlink(f"{tensor.name}.vlink.{offset}", read, group, nbytes)
            noc = self._noc(
                f"{tensor.name}.noc.{offset}", vlink,
                self.arch.noc.endpoint_for_bank_group[group], self.arch.noc.endpoint_for_global_buffer,
                self._vc(role), nbytes,
            )
            terms.append(self._buffer_write("gbuf", f"{tensor.name}.gbuf.{offset}", noc, nbytes))
        return self.builder.barrier(f"{tensor.name}.gbuf_ready", tuple(terms))

    def _write_global_buffer_to_tensor(self, tensor: Tensor, dep: int, role: str) -> int:
        if tensor.nbytes > self.arch.global_buffer.capacity_bytes:
            raise MappingError(
                f"tensor {tensor.name} requires {tensor.nbytes} bytes in the global buffer, "
                f"capacity is {self.arch.global_buffer.capacity_bytes}"
            )
        terms: list[int] = []
        for offset in range(0, tensor.nbytes, self.arch.sram.line_bytes):
            nbytes = min(self.arch.sram.line_bytes, tensor.nbytes - offset)
            address = tensor.address + offset
            bank = self._bank(address, None)
            group = self._bank_group(bank)
            read = self._buffer_read("gbuf", f"{tensor.name}.gbuf_read.{offset}", dep, nbytes)
            noc = self._noc(
                f"{tensor.name}.noc_write.{offset}", read,
                self.arch.noc.endpoint_for_global_buffer, self.arch.noc.endpoint_for_bank_group[group],
                self._vc(role), nbytes,
            )
            vlink = self._vlink(f"{tensor.name}.vlink_write.{offset}", noc, group, nbytes)
            terms.append(self._sram_write(f"{tensor.name}.write.{offset}", vlink, bank, address, nbytes))
        return self.builder.barrier(f"{tensor.name}.written", tuple(terms))

    def _sram_read(self, name: str, dep: int, bank: int, address: int, nbytes: int) -> int:
        return self.builder.add(
            name=name, kind=MicroOpKind.SRAM_READ, deps=(dep,), resource_id=f"sram.bank.{bank}.read",
            latency_cycles=self.macro.access_cycles, issue_interval_cycles=self.macro.issue_cycles,
            nbytes=nbytes, metadata={"address": address},
        )

    def _sram_write(self, name: str, dep: int, bank: int, address: int, nbytes: int) -> int:
        return self.builder.add(
            name=name, kind=MicroOpKind.SRAM_WRITE, deps=(dep,), resource_id=f"sram.bank.{bank}.write",
            latency_cycles=self.macro.access_cycles, issue_interval_cycles=self.macro.issue_cycles,
            nbytes=nbytes, metadata={"address": address},
        )

    def _vlink(self, name: str, dep: int, bank_group: int, nbytes: int) -> int:
        link = bank_group % self.arch.vertical_link.groups
        duration = self.arch.vertical_link.latency_cycles + math.ceil(nbytes / self.arch.vertical_link.bytes_per_cycle)
        return self.builder.add(
            name=name, kind=MicroOpKind.VLINK_SEND, deps=(dep,), resource_id=f"vlink.{link}",
            latency_cycles=duration, issue_interval_cycles=duration, nbytes=nbytes,
            metadata={"bank_group": bank_group},
        )

    def _noc(self, name: str, dep: int, src: int, dst: int, vc: int, nbytes: int) -> int:
        flits = math.ceil(nbytes / self.arch.noc.flit_bytes)
        return self.builder.add(
            name=name, kind=MicroOpKind.NOC_SEND, deps=(dep,), resource_id="noc",
            latency_cycles=0, issue_interval_cycles=0, nbytes=nbytes,
            metadata={"src": src, "dst": dst, "vc": vc, "flits": flits, "traffic_class": 0},
        )

    def _buffer_write(self, prefix: str, name: str, dep: int, nbytes: int) -> int:
        if nbytes > self.arch.global_buffer.capacity_bytes:
            raise MappingError(
                f"buffer transfer {name} requires {nbytes} bytes, capacity is {self.arch.global_buffer.capacity_bytes}"
            )
        return self.builder.add(
            name=name, kind=MicroOpKind.BUFFER_WRITE, deps=(dep,), resource_id=f"{prefix}.write",
            latency_cycles=self.macro.access_cycles, issue_interval_cycles=self.macro.issue_cycles,
            nbytes=nbytes, metadata={},
        )

    def _buffer_read(self, prefix: str, name: str, dep: int, nbytes: int) -> int:
        if nbytes > self.arch.global_buffer.capacity_bytes:
            raise MappingError(
                f"buffer transfer {name} requires {nbytes} bytes, capacity is {self.arch.global_buffer.capacity_bytes}"
            )
        return self.builder.add(
            name=name, kind=MicroOpKind.BUFFER_READ, deps=(dep,), resource_id=f"{prefix}.read",
            latency_cycles=self.macro.access_cycles, issue_interval_cycles=self.macro.issue_cycles,
            nbytes=nbytes, metadata={},
        )

    def _bank(self, address: int, group: int | None) -> int:
        line = address // self.arch.sram.line_bytes
        if group is None:
            return line % self.arch.sram.total_banks
        banks_per_group = self.arch.sram.total_banks // self.arch.bank_groups
        return group * banks_per_group + line % banks_per_group

    def _bank_group(self, bank: int) -> int:
        banks_per_group = self.arch.sram.total_banks // self.arch.bank_groups
        return bank // banks_per_group

    def _vc(self, role: str) -> int:
        if role == "activation":
            return self.arch.noc.vc_activation
        if role == "weight":
            return self.arch.noc.vc_weight
        if role == "output":
            return self.arch.noc.vc_output
        if role == "partial_sum":
            return self.arch.noc.vc_partial_sum
        if role == "control":
            return self.arch.noc.vc_control
        raise MappingError(f"unknown NoC traffic role {role}")


def _split_ranges(total: int, parts: int) -> tuple[tuple[int, int], ...]:
    base = total // parts
    remainder = total % parts
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        ranges.append((cursor, size))
        cursor += size
    return tuple(ranges)


def _append(mapping: dict[int, list[int]], key: int, value: int) -> None:
    if key not in mapping:
        mapping[key] = []
    mapping[key].append(value)


def _append_pair(mapping: dict[int, list[tuple[int, int]]], key: int, value: tuple[int, int]) -> None:
    if key not in mapping:
        mapping[key] = []
    mapping[key].append(value)


def _translate_a(local_address: int, element_bytes: int, tile: _Tile, full_k: int, base: int) -> int:
    index = local_address // element_bytes
    row = index // tile.k
    col = index % tile.k
    if row >= tile.m:
        raise MappingError("SCALE-Sim input trace exceeds tile bounds")
    return base + ((tile.m0 + row) * full_k + tile.k0 + col) * element_bytes


def _translate_b(local_address: int, element_bytes: int, tile: _Tile, full_n: int, base: int) -> int:
    index = local_address // element_bytes
    row = index // tile.n
    col = index % tile.n
    if row >= tile.k:
        raise MappingError("SCALE-Sim weight trace exceeds tile bounds")
    return base + ((tile.k0 + row) * full_n + tile.n0 + col) * element_bytes


def _translate_c(local_address: int, element_bytes: int, tile: _Tile, full_n: int, base: int) -> int:
    index = local_address // element_bytes
    row = index // tile.n
    col = index % tile.n
    if row >= tile.m:
        raise MappingError("SCALE-Sim output trace exceeds tile bounds")
    return base + ((tile.m0 + row) * full_n + tile.n0 + col) * element_bytes
