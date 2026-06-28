import heapq
from dataclasses import dataclass, field
from src.config import SimConfig
from src.trace_ir import TraceCommand, OpCode
from src.memory_object import MemoryObject, ObjType, ObjState
from src.memory_manager import MemoryManager
from src.dram_model import create_dram_model
from src.energy_model import EnergyModel
from src.resource_model import ResourceModel
from src.utils.math_utils import ceil_div
from src.utils.location import parse_sram_loc, global_bank_ids, parse_src_ids


def _obj_type_from_str(s: str) -> ObjType:
    mapping = {
        "WEIGHT": ObjType.WEIGHT, "ACTIVATION": ObjType.ACTIVATION,
        "STATE": ObjType.STATE, "PSUM": ObjType.PSUM, "LUT": ObjType.LUT,
        "META": ObjType.META, "OUTPUT": ObjType.OUTPUT,
        "COORD": ObjType.COORD, "DIST": ObjType.DIST,
    }
    return mapping.get(s, ObjType.ACTIVATION)


@dataclass(order=True)
class Event:
    finish_cycle: int
    cmd_id: int
    cmd: TraceCommand = field(compare=False)


class Simulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.mem_mgr = MemoryManager(config.sram_pim)
        self.dram = create_dram_model(config.dram, config.system.frequency_hz)
        self.energy = EnergyModel(config.energy, config.sram_pim,
                                  config.dram, config.system.frequency_hz)
        self.resource = ResourceModel(config.sram_pim)
        self.strict = config.system.correctness_mode == "strict"
        self.cycle = 0
        self.commands = []
        self.completed = set()
        self.event_queue: list[Event] = []
        self.latency_breakdown = {
            "dram_load_cycles": 0,
            "dram_store_cycles": 0,
            "pim_compute_cycles": 0,
            "pim_reduce_cycles": 0,
            "stall_dependency_cycles": 0,
            "bank_conflict_stall_cycles": 0,
            "spill_writeback_cycles": 0,
            "final_writeback_cycles": 0,
        }
        self.correctness = {
            "illegal_read_unresident_object": 0,
            "missing_input_object": 0,
            "illegal_read_power_gated_object": 0,
            "illegal_evict_dirty_without_writeback": 0,
            "capacity_overcommit_events": 0,
            "dependency_violations": 0,
            "deadlock_events": 0,
            "final_dirty_objects": 0,
        }
        self.final_dirty_objects: list[dict] = []
        self.final_resident_objects: list[dict] = []

    def load_trace(self, commands: list):
        self.commands = list(commands)

    def _deps_ready(self, cmd: TraceCommand) -> bool:
        return all(d in self.completed for d in cmd.deps)

    # ------------------------------------------------------------------ #
    # Bank resolution (P0-08)
    # ------------------------------------------------------------------ #
    def _banks_of_object(self, oid: str) -> list:
        obj = self.mem_mgr.objects.get(oid)
        if obj is None or obj.sram_tile < 0:
            return []
        bpt = self.config.sram_pim.banks_per_tile
        return [obj.sram_tile * bpt + b for b in obj.sram_banks]

    def _get_banks(self, cmd: TraceCommand) -> list:
        if "banks" in cmd.attrs:
            return list(cmd.attrs["banks"])

        banks = set()
        bpt = self.config.sram_pim.banks_per_tile

        if cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE,
                      OpCode.PIM_NL}:
            for iid in parse_src_ids(cmd.src):
                if iid in self.mem_mgr.objects:
                    banks.update(self._banks_of_object(iid))
            if cmd.object_id in self.mem_mgr.objects:
                banks.update(self._banks_of_object(cmd.object_id))
            elif cmd.dst and cmd.dst.startswith("SRAM:"):
                loc = parse_sram_loc(cmd.dst)
                banks.update(global_bank_ids(loc, bpt))
            return sorted(banks)

        if cmd.op == OpCode.DMA_LOAD:
            if cmd.dst and cmd.dst.startswith("SRAM:"):
                loc = parse_sram_loc(cmd.dst)
                return global_bank_ids(loc, bpt)
            if cmd.object_id in self.mem_mgr.objects:
                return self._banks_of_object(cmd.object_id)

        if cmd.op == OpCode.DMA_STORE:
            if cmd.object_id in self.mem_mgr.objects:
                return self._banks_of_object(cmd.object_id)
            if cmd.src and cmd.src.startswith("SRAM:"):
                loc = parse_sram_loc(cmd.src)
                return global_bank_ids(loc, bpt)

        for loc_str in [cmd.dst, cmd.src]:
            if loc_str and loc_str.startswith("SRAM:"):
                try:
                    loc = parse_sram_loc(loc_str)
                    return global_bank_ids(loc, bpt)
                except Exception:
                    pass
        return []

    # ------------------------------------------------------------------ #
    # P0-C: Input validation with auto_reload support
    # ------------------------------------------------------------------ #
    def _ensure_inputs_ready(self, cmd: TraceCommand) -> int:
        extra_lat = 0
        mode = self.config.system.correctness_mode

        for iid in parse_src_ids(cmd.src):
            obj = self.mem_mgr.objects.get(iid)

            if obj is None:
                self.correctness["missing_input_object"] += 1
                if mode in {"strict", "auto_reload"}:
                    raise RuntimeError(
                        f"cmd {cmd.cmd_id} ({cmd.op.value}): "
                        f"missing input object '{iid}'")
                continue

            if obj.power_state != "active":
                self.correctness["illegal_read_power_gated_object"] += 1
                if mode in {"strict", "auto_reload"}:
                    raise RuntimeError(
                        f"cmd {cmd.cmd_id} ({cmd.op.value}): "
                        f"input '{iid}' is {obj.power_state}")
                continue

            if obj.valid_in_sram:
                continue

            self.correctness["illegal_read_unresident_object"] += 1

            if mode == "strict":
                raise RuntimeError(
                    f"cmd {cmd.cmd_id} ({cmd.op.value}): "
                    f"input '{iid}' not valid in SRAM (state={obj.state.value})")

            if mode == "auto_reload":
                if not obj.valid_in_dram:
                    raise RuntimeError(
                        f"cmd {cmd.cmd_id} ({cmd.op.value}): "
                        f"input '{iid}' not in SRAM and DRAM copy invalid")
                extra_lat += self._blocking_reload(obj)

            # mode == "warn": just continue

        return extra_lat

    # ------------------------------------------------------------------ #
    # Command dispatch
    # ------------------------------------------------------------------ #
    def _issue_command(self, cmd: TraceCommand) -> int:
        self.energy.add_command()

        if cmd.op == OpCode.SRAM_ALLOC:
            return self._issue_alloc(cmd)
        elif cmd.op == OpCode.DMA_LOAD:
            return self._issue_dma_load(cmd)
        elif cmd.op == OpCode.DMA_STORE:
            return self._issue_dma_store(cmd)
        elif cmd.op == OpCode.PIM_MAC:
            return self._issue_pim_mac(cmd)
        elif cmd.op == OpCode.PIM_EW_OP:
            return self._issue_pim_ew(cmd)
        elif cmd.op == OpCode.PIM_REDUCE:
            return self._issue_pim_reduce(cmd)
        elif cmd.op == OpCode.PIM_NL:
            return self._issue_pim_nl(cmd)
        elif cmd.op == OpCode.SRAM_FREE:
            return self._issue_free(cmd)
        elif cmd.op == OpCode.BARRIER:
            return 0
        elif cmd.op == OpCode.PIM_WRITEBACK:
            return self._issue_pim_writeback(cmd)
        elif cmd.op == OpCode.POWER_SET:
            return self._issue_power_set(cmd)
        return 0

    def _issue_alloc(self, cmd: TraceCommand) -> int:
        loc = parse_sram_loc(cmd.dst)
        tile, banks = loc.tile, list(loc.banks)
        obj_type_str = cmd.attrs.get("type", "ACTIVATION")
        pinned = bool(cmd.attrs.get("pinned", False))
        preloaded = bool(cmd.attrs.get("preloaded", False))

        obj = MemoryObject(
            object_id=cmd.object_id,
            obj_type=_obj_type_from_str(obj_type_str),
            bytes=cmd.bytes,
            precision="int8",
            pinned=pinned,
        )
        self.mem_mgr.register_object(obj)

        ok = self.mem_mgr.allocate(cmd.object_id, tile, banks,
                                   make_valid=preloaded)
        if not ok:
            extra = self._allocate_or_spill(cmd.object_id, tile, banks)
            if extra > 0:
                self.latency_breakdown["spill_writeback_cycles"] += extra
            if preloaded:
                obj = self.mem_mgr.objects[cmd.object_id]
                obj.valid_in_sram = True
                obj.state = ObjState.VALID_CLEAN
            return extra
        return 0

    def _issue_dma_load(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.begin_loading()
        lat = self.dram.get_read_latency(cmd.bytes)
        self.energy.add_dram_read(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        self.latency_breakdown["dram_load_cycles"] += lat
        return lat

    def _issue_dma_store(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.begin_spilling()
        lat = self.dram.get_write_latency(cmd.bytes)
        self.energy.add_dram_write(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        self.latency_breakdown["dram_store_cycles"] += lat
        return lat

    def _issue_pim_mac(self, cmd: TraceCommand) -> int:
        reload_lat = self._ensure_inputs_ready(cmd)
        mac_count = cmd.attrs.get("mac_count", 0)

        output_banks = self._get_output_banks(cmd)
        active_banks = max(1, len(output_banks))
        effective_lanes = active_banks * self.config.sram_pim.pim.lanes_per_bank
        compute_steps = ceil_div(mac_count, effective_lanes) if mac_count > 0 else 1
        lat = max(
            self.config.sram_pim.pim.mac_latency_cycles,
            compute_steps * self.config.sram_pim.pim.mac_issue_interval_cycles
        )

        self._account_pim_data_energy(cmd, mac_count)
        self.energy.add_pim_mac(mac_count)

        # P0-A: _ensure_output_object returns extra alloc/spill latency
        alloc_lat = self._ensure_output_object(cmd, ObjType.PSUM, "int32")
        lat += alloc_lat + reload_lat

        obj = self.mem_mgr.objects[cmd.object_id]
        obj.begin_producing()

        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_ew(self, cmd: TraceCommand) -> int:
        reload_lat = self._ensure_inputs_ready(cmd)
        count = cmd.attrs.get("count", 0)

        active_banks = max(1, len(self._get_output_banks(cmd)))
        effective_lanes = active_banks * self.config.sram_pim.pim.lanes_per_bank
        lat = max(1, ceil_div(count, effective_lanes))

        self._account_pim_data_energy(cmd, count)
        self.energy.add_pim_ew(count)

        # P0-D: Use attrs for output type/precision
        out_type = _obj_type_from_str(cmd.attrs.get("out_type", "ACTIVATION"))
        out_prec = cmd.attrs.get("out_precision", "int8")
        alloc_lat = self._ensure_output_object(cmd, out_type, out_prec)
        lat += alloc_lat + reload_lat

        obj = self.mem_mgr.objects[cmd.object_id]
        obj.begin_producing()

        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_reduce(self, cmd: TraceCommand) -> int:
        reload_lat = self._ensure_inputs_ready(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.reduce_latency_cycles
        self.energy.add_pim_reduce(count)

        out_type = _obj_type_from_str(cmd.attrs.get("out_type", "PSUM"))
        out_prec = cmd.attrs.get("out_precision", "int32")
        alloc_lat = self._ensure_output_object(cmd, out_type, out_prec)
        lat += alloc_lat + reload_lat

        obj = self.mem_mgr.objects[cmd.object_id]
        obj.begin_producing()

        self.latency_breakdown["pim_reduce_cycles"] += lat
        return lat

    def _issue_pim_nl(self, cmd: TraceCommand) -> int:
        reload_lat = self._ensure_inputs_ready(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.nonlinear_latency_cycles
        self.energy.add_pim_nl(count)

        out_type = _obj_type_from_str(cmd.attrs.get("out_type", "ACTIVATION"))
        out_prec = cmd.attrs.get("out_precision", "int8")
        alloc_lat = self._ensure_output_object(cmd, out_type, out_prec)
        lat += alloc_lat + reload_lat

        obj = self.mem_mgr.objects[cmd.object_id]
        obj.begin_producing()

        return lat

    def _issue_pim_writeback(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            n_accesses = ceil_div(cmd.bytes, self.config.sram_pim.word_bytes)
            self.energy.add_sram_write(n_accesses)
        return 1

    def _issue_free(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj and obj.dirty_in_sram:
            self.correctness["illegal_evict_dirty_without_writeback"] += 1
            if self.strict:
                raise RuntimeError(
                    f"cmd {cmd.cmd_id}: Cannot free dirty object "
                    f"'{cmd.object_id}' without DMA_STORE")
            lat = self._blocking_writeback(obj)
            self.mem_mgr.free(cmd.object_id)
            return lat
        if obj:
            self.mem_mgr.free(cmd.object_id)
        return 0

    def _issue_power_set(self, cmd: TraceCommand) -> int:
        target = cmd.attrs.get("state", "active")
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj is None:
            return 0
        if target == "power_gated":
            if obj.dirty_in_sram:
                if self.strict:
                    raise RuntimeError(
                        f"Cannot power-gate dirty object {obj.object_id}")
                self._blocking_writeback(obj)
            obj.power_gate()
            return self.config.sram_pim.power_state.wakeup_latency_cycles
        elif target == "clock_gated":
            obj.power_state = "clock_gated"
            return 0
        elif target == "active":
            if obj.power_state == "power_gated":
                obj.wakeup()
                return self.config.sram_pim.power_state.wakeup_latency_cycles
            obj.wakeup()
            return 0
        return 0

    # ------------------------------------------------------------------ #
    # P0-03: Completion handler
    # ------------------------------------------------------------------ #
    def _complete_command(self, cmd: TraceCommand) -> None:
        if cmd.op == OpCode.DMA_LOAD:
            obj = self.mem_mgr.objects.get(cmd.object_id)
            if obj:
                obj.commit_load()
                n_accesses = ceil_div(cmd.bytes, self.config.sram_pim.word_bytes)
                self.energy.add_sram_write(n_accesses)

        elif cmd.op == OpCode.DMA_STORE:
            obj = self.mem_mgr.objects.get(cmd.object_id)
            if obj:
                n_accesses = ceil_div(cmd.bytes, self.config.sram_pim.word_bytes)
                self.energy.add_sram_read(n_accesses)
                obj.writeback_complete()

        elif cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP,
                        OpCode.PIM_REDUCE, OpCode.PIM_NL}:
            obj = self.mem_mgr.objects.get(cmd.object_id)
            if obj:
                obj.commit_produce()

        elif cmd.op == OpCode.PIM_WRITEBACK:
            obj = self.mem_mgr.objects.get(cmd.object_id)
            if obj:
                obj.mark_dirty()

    # ------------------------------------------------------------------ #
    # P0-A: Output object allocation with spill support
    # ------------------------------------------------------------------ #
    def _ensure_output_object(self, cmd: TraceCommand,
                              default_type: ObjType,
                              default_precision: str) -> int:
        extra_lat = 0

        # P0-D: Resolve output metadata from attrs
        out_type = _obj_type_from_str(
            cmd.attrs.get("out_type", default_type.name))
        out_precision = cmd.attrs.get("out_precision", default_precision)
        out_bytes = cmd.attrs.get("out_bytes", cmd.bytes or 1024)

        if cmd.object_id not in self.mem_mgr.objects:
            out_obj = MemoryObject(
                object_id=cmd.object_id,
                obj_type=out_type,
                bytes=out_bytes,
                precision=out_precision,
                valid_in_dram=False,
                valid_in_sram=False,
            )
            self.mem_mgr.register_object(out_obj)
        else:
            out_obj = self.mem_mgr.objects[cmd.object_id]

        obj = self.mem_mgr.objects[cmd.object_id]
        if obj.sram_tile < 0:
            if not (cmd.dst and cmd.dst.startswith("SRAM:")):
                if self.strict:
                    raise RuntimeError(
                        f"PIM output {cmd.object_id} has no SRAM dst")
                return 0

            loc = parse_sram_loc(cmd.dst)
            ok = self.mem_mgr.allocate(
                cmd.object_id, loc.tile, list(loc.banks), make_valid=False)
            if not ok:
                extra_lat += self._allocate_or_spill(
                    cmd.object_id, loc.tile, list(loc.banks))

        # Final placement check
        obj = self.mem_mgr.objects[cmd.object_id]
        if obj.sram_tile < 0:
            if self.strict:
                raise RuntimeError(
                    f"PIM output {cmd.object_id} allocation failed")

        if extra_lat > 0:
            self.latency_breakdown["spill_writeback_cycles"] += extra_lat

        return extra_lat

    def _get_output_banks(self, cmd: TraceCommand) -> list:
        if cmd.object_id in self.mem_mgr.objects:
            return self._banks_of_object(cmd.object_id)
        if cmd.dst and cmd.dst.startswith("SRAM:"):
            loc = parse_sram_loc(cmd.dst)
            return global_bank_ids(loc, self.config.sram_pim.banks_per_tile)
        return []

    # ------------------------------------------------------------------ #
    # Spill / writeback / reload helpers
    # ------------------------------------------------------------------ #
    def _allocate_or_spill(self, object_id: str, tile: int,
                           banks: list) -> int:
        obj = self.mem_mgr.objects[object_id]
        needed = obj.bytes - self.mem_mgr.get_free_bytes(tile)
        if needed <= 0:
            needed = obj.bytes
        victims = self.mem_mgr.find_eviction_candidate(tile, needed)

        if not victims:
            self.correctness["capacity_overcommit_events"] += 1
            if self.strict:
                raise RuntimeError(
                    f"No eviction candidate for {object_id}, "
                    f"need {needed} bytes on tile {tile}")
            obj.place_in_sram(tile, banks)
            return 0

        extra_lat = 0
        for vid in victims:
            victim = self.mem_mgr.objects[vid]
            if victim.dirty_in_sram:
                extra_lat += self._blocking_writeback(victim)
                self.mem_mgr.stats["writeback_count"] += 1
            self.mem_mgr._release_capacity(victim)
            victim.valid_in_sram = False
            victim.dirty_in_sram = False
            victim.sram_tile = -1
            victim.sram_banks = []
            victim.state = ObjState.EVICTED
            self.mem_mgr.stats["eviction_count"] += 1

        ok = self.mem_mgr.allocate(object_id, tile, banks)
        if not ok:
            self.correctness["capacity_overcommit_events"] += 1
            if self.strict:
                raise RuntimeError(
                    f"Allocation still fails after eviction: {object_id}")
            obj.place_in_sram(tile, banks)
        return extra_lat

    def _blocking_writeback(self, obj: MemoryObject) -> int:
        lat = self.dram.get_write_latency(obj.bytes)
        self.energy.add_dram_write(obj.bytes)
        self.energy.add_noc(obj.bytes)
        n_accesses = ceil_div(obj.bytes, self.config.sram_pim.word_bytes)
        self.energy.add_sram_read(n_accesses)
        obj.writeback_complete()
        self.mem_mgr.stats["spill_count"] += 1
        return lat

    def _blocking_reload(self, obj: MemoryObject) -> int:
        """P0-C: Synchronously reload an evicted object from DRAM."""
        if obj.sram_tile < 0:
            # Simple first-fit placement on tile 0
            tile = 0
            banks = [0, 1]
            ok = self.mem_mgr.allocate(obj.object_id, tile, banks,
                                       make_valid=False)
            if not ok:
                extra = self._allocate_or_spill(obj.object_id, tile, banks)
            else:
                extra = 0
        else:
            extra = 0

        obj.begin_loading()
        lat = self.dram.get_read_latency(obj.bytes)
        self.energy.add_dram_read(obj.bytes)
        self.energy.add_noc(obj.bytes)
        n_accesses = ceil_div(obj.bytes, self.config.sram_pim.word_bytes)
        self.energy.add_sram_write(n_accesses)
        obj.commit_load()
        self.mem_mgr.stats["reload_count"] += 1
        return extra + lat

    # ------------------------------------------------------------------ #
    # PIM data energy accounting (P1-05)
    # ------------------------------------------------------------------ #
    def _account_pim_data_energy(self, cmd: TraceCommand, op_count: int):
        word_bytes = self.config.sram_pim.word_bytes

        # P1-D: Use explicit dataflow bytes from attrs if available
        act_read = cmd.attrs.get("act_read_bytes")
        weight_read = cmd.attrs.get("weight_read_bytes")
        psum_read = cmd.attrs.get("psum_read_bytes")
        out_write = cmd.attrs.get("out_write_bytes")

        if act_read is not None or weight_read is not None:
            # Fine-grained dataflow energy
            if act_read and act_read > 0:
                self.energy.add_sram_read(ceil_div(act_read, word_bytes))
            if weight_read and weight_read > 0:
                self.energy.add_sram_read(ceil_div(weight_read, word_bytes))
            if psum_read and psum_read > 0:
                self.energy.add_sram_read(ceil_div(psum_read, word_bytes))
            if out_write and out_write > 0:
                self.energy.add_sram_write(ceil_div(out_write, word_bytes))
        else:
            # Fallback: estimate from full object sizes
            for iid in parse_src_ids(cmd.src):
                obj = self.mem_mgr.objects.get(iid)
                if obj:
                    self.energy.add_sram_read(ceil_div(obj.bytes, word_bytes))

            out_bytes = cmd.attrs.get("out_bytes", cmd.bytes or 0)
            if out_bytes > 0:
                if cmd.attrs.get("accumulate", False):
                    self.energy.add_sram_read(ceil_div(out_bytes, word_bytes))
                self.energy.add_sram_write(ceil_div(out_bytes, word_bytes))

    # ------------------------------------------------------------------ #
    # P0-B: Final dirty object check
    # ------------------------------------------------------------------ #
    def _is_persistent_object(self, obj: MemoryObject) -> bool:
        persistent = set(self.config.system.persistent_object_types)
        return obj.obj_type.name in persistent

    def _finalize_simulation(self) -> None:
        self.final_dirty_objects = []
        self.final_resident_objects = []

        for oid, obj in self.mem_mgr.objects.items():
            if obj.valid_in_sram:
                self.final_resident_objects.append({
                    "object_id": oid,
                    "type": obj.obj_type.name,
                    "dirty": obj.dirty_in_sram,
                    "tile": obj.sram_tile,
                    "banks": list(obj.sram_banks),
                })

            if obj.dirty_in_sram and self._is_persistent_object(obj):
                self.final_dirty_objects.append({
                    "object_id": oid,
                    "type": obj.obj_type.name,
                    "bytes": obj.bytes,
                })

        if not self.final_dirty_objects:
            return

        policy = self.config.system.final_dirty_policy

        if policy == "error":
            self.correctness["final_dirty_objects"] = len(self.final_dirty_objects)
            if self.strict:
                raise RuntimeError(
                    f"Trace ended with {len(self.final_dirty_objects)} dirty "
                    f"persistent objects: "
                    f"{[d['object_id'] for d in self.final_dirty_objects]}")

        elif policy == "auto_writeback":
            for item in self.final_dirty_objects:
                obj = self.mem_mgr.objects[item["object_id"]]
                lat = self._blocking_writeback(obj)
                self.latency_breakdown["final_writeback_cycles"] += lat
                self.cycle += lat
            self.final_dirty_objects = []  # all written back

        elif policy == "report":
            self.correctness["final_dirty_objects"] = len(self.final_dirty_objects)

        # policy == "ignore": do nothing

    # ------------------------------------------------------------------ #
    # Main simulation loop
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not self.commands:
            return self._make_report()

        pending = {cmd.cmd_id: cmd for cmd in self.commands}
        self.cycle = 0
        max_cycles = 100_000_000

        while (pending or self.event_queue) and self.cycle < max_cycles:
            self.resource.release_at(self.cycle)

            while self.event_queue and self.event_queue[0].finish_cycle <= self.cycle:
                event = heapq.heappop(self.event_queue)
                self._complete_command(event.cmd)
                self.completed.add(event.cmd_id)

            deps_ready = [cid for cid, cmd in pending.items()
                          if self._deps_ready(cmd)]

            resource_stalled = []
            for cid in deps_ready:
                cmd = pending[cid]
                banks = self._get_banks(cmd)
                if self.resource.can_issue(cmd.op, banks, cmd.bytes):
                    pending.pop(cid)
                    latency = self._issue_command(cmd)
                    if latency > 0:
                        self.resource.reserve(cmd.op, banks, cmd.bytes, latency)
                        heapq.heappush(self.event_queue,
                                       Event(self.cycle + latency, cid, cmd))
                    else:
                        self._complete_command(cmd)
                        self.completed.add(cid)
                else:
                    resource_stalled.append(cid)

            if resource_stalled:
                self.latency_breakdown["bank_conflict_stall_cycles"] += 1

            active = self.mem_mgr.count_active_banks()
            self.energy.add_leakage(1, active_banks=active)

            if not self.event_queue and not any(
                self._deps_ready(cmd) for cmd in pending.values()
            ):
                if pending:
                    self.correctness["deadlock_events"] += 1
                    self.correctness["dependency_violations"] += len(pending)
                    unresolved = {
                        cid: [d for d in cmd.deps if d not in self.completed]
                        for cid, cmd in pending.items()
                        if not self._deps_ready(cmd)
                    }
                    if self.strict:
                        raise RuntimeError(
                            f"Deadlock: {len(pending)} commands with "
                            f"unresolved deps: {unresolved}")
                    for cid in list(pending.keys()):
                        self.completed.add(cid)
                    pending.clear()
                break

            self.cycle += 1

        # P0-B: Check for dirty persistent objects at end
        self._finalize_simulation()

        return self._make_report()

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #
    def _make_report(self) -> dict:
        eb = self.energy.get_breakdown()
        freq = self.config.system.frequency_hz
        total_ns = self.cycle * 1e9 / freq
        valid = all(v == 0 for v in self.correctness.values())

        return {
            "latency": {
                "total_cycles": self.cycle,
                "total_ns": total_ns,
                **self.latency_breakdown,
            },
            "energy": eb,
            "traffic": {
                "dram_read_bytes": eb["dram_read_bytes"],
                "dram_write_bytes": eb["dram_write_bytes"],
                "sram_read_accesses": eb["sram_read_count"],
                "sram_write_accesses": eb["sram_write_count"],
                "noc_bytes": eb["noc_bytes"],
            },
            "correctness": self.correctness,
            "valid_simulation": valid,
            "memory_lifecycle": {
                "spill_count": self.mem_mgr.stats["spill_count"],
                "eviction_count": self.mem_mgr.stats["eviction_count"],
                "writeback_count": self.mem_mgr.stats["writeback_count"],
                "reload_count": self.mem_mgr.stats["reload_count"],
            },
            "pim": {
                "mac_count": eb["pim_mac_count"],
                "ew_count": eb["pim_ew_count"],
                "reduce_count": eb["pim_reduce_count"],
                "nl_count": eb["pim_nl_count"],
            },
            "final_state": {
                "final_dirty_policy": self.config.system.final_dirty_policy,
                "final_dirty_objects": self.final_dirty_objects,
                "final_resident_objects": self.final_resident_objects,
            },
            "model_provenance": {
                "correctness_mode": self.config.system.correctness_mode,
                "spill_model": self.config.system.spill_model,
                "dram_model": self.config.dram.model,
                "sram_param_source": self.config.energy.sram.source,
                "workload_trace_semantics": "operator_level_approximation",
            },
        }
