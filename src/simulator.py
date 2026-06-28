import heapq
from src.config import SimConfig
from src.trace_ir import TraceCommand, OpCode
from src.memory_object import MemoryObject, ObjType
from src.memory_manager import MemoryManager
from src.dram_model import DRAMModel
from src.energy_model import EnergyModel
from src.resource_model import ResourceModel


def _parse_sram_loc(loc_str: str):
    """Parse 'SRAM:T0:B0-3' -> (tile=0, banks=[0,1,2,3])"""
    parts = loc_str.split(":")
    tile = int(parts[1][1:])
    bank_str = parts[2][1:]
    if "-" in bank_str:
        lo, hi = bank_str.split("-")
        banks = list(range(int(lo), int(hi) + 1))
    else:
        banks = [int(b) for b in bank_str.split(",")]
    return tile, banks


def _obj_type_from_str(s: str) -> ObjType:
    mapping = {
        "WEIGHT": ObjType.WEIGHT, "ACTIVATION": ObjType.ACTIVATION,
        "STATE": ObjType.STATE, "PSUM": ObjType.PSUM, "LUT": ObjType.LUT,
        "META": ObjType.META, "OUTPUT": ObjType.OUTPUT,
        "COORD": ObjType.COORD, "DIST": ObjType.DIST,
    }
    return mapping.get(s, ObjType.ACTIVATION)


class Simulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.mem_mgr = MemoryManager(config.sram_pim)
        self.dram = DRAMModel(config.dram, config.system.frequency_hz)
        self.energy = EnergyModel(config.energy, config.sram_pim,
                                  config.dram, config.system.frequency_hz)
        self.resource = ResourceModel(config.sram_pim)
        self.cycle = 0
        self.commands = []
        self.completed = set()
        self.event_queue = []  # (finish_cycle, cmd_id)
        self.latency_breakdown = {
            "dram_load_cycles": 0,
            "dram_store_cycles": 0,
            "pim_compute_cycles": 0,
            "pim_reduce_cycles": 0,
            "stall_dependency_cycles": 0,
            "bank_conflict_stall_cycles": 0,
        }
        self.correctness = {
            "illegal_read_unresident_object": 0,
            "illegal_evict_dirty_without_writeback": 0,
            "capacity_overcommit_events": 0,
            "dependency_violations": 0,
        }

    def load_trace(self, commands: list):
        self.commands = list(commands)

    def _deps_ready(self, cmd: TraceCommand) -> bool:
        return all(d in self.completed for d in cmd.deps)

    def _get_banks(self, cmd: TraceCommand) -> list:
        """Extract bank indices from a command for resource tracking."""
        # Explicit banks list in attrs takes priority
        if "banks" in cmd.attrs:
            return list(cmd.attrs["banks"])
        # Parse from dst location if it's an SRAM location
        loc = cmd.dst if cmd.dst and cmd.dst.startswith("SRAM:") else (
            cmd.src if cmd.src and cmd.src.startswith("SRAM:") else None
        )
        if loc:
            try:
                tile, local_banks = _parse_sram_loc(loc)
                banks_per_tile = self.config.sram_pim.banks_per_tile
                return [tile * banks_per_tile + b for b in local_banks]
            except Exception:
                pass
        return []

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
            return 0
        return 0

    def _issue_alloc(self, cmd: TraceCommand) -> int:
        tile, banks = _parse_sram_loc(cmd.dst)
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
        ok = self.mem_mgr.allocate(cmd.object_id, tile, banks)
        if not ok:
            self.correctness["capacity_overcommit_events"] += 1
        if preloaded:
            obj.valid_in_sram = True
        return 0

    def _issue_dma_load(self, cmd: TraceCommand) -> int:
        lat = self.dram.get_read_latency(cmd.bytes)
        self.energy.add_dram_read(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.valid_in_sram = True
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_write(n_accesses)
        self.latency_breakdown["dram_load_cycles"] += lat
        return lat

    def _issue_dma_store(self, cmd: TraceCommand) -> int:
        lat = self.dram.get_write_latency(cmd.bytes)
        self.energy.add_dram_write(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_read(n_accesses)
            obj.writeback_complete()
        self.latency_breakdown["dram_store_cycles"] += lat
        return lat

    def _check_inputs_valid(self, cmd: TraceCommand) -> bool:
        input_ids = [s.strip() for s in cmd.src.split(",") if s.strip() and s.strip() != "-"]
        for iid in input_ids:
            obj = self.mem_mgr.objects.get(iid)
            if obj is None or not obj.valid_in_sram:
                self.correctness["illegal_read_unresident_object"] += 1
                return False
        return True

    def _issue_pim_mac(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        mac_count = cmd.attrs.get("mac_count", 0)
        lat = self.config.sram_pim.pim.mac_latency_cycles
        if mac_count > 0:
            lat = max(lat, mac_count * self.config.sram_pim.pim.mac_issue_interval_cycles)
        self.energy.add_pim_mac(mac_count)
        # Create output object if not exists
        if cmd.object_id not in self.mem_mgr.objects:
            out_obj = MemoryObject(cmd.object_id, ObjType.PSUM, cmd.bytes or 1024, "int32")
            self.mem_mgr.register_object(out_obj)
            if cmd.dst.startswith("SRAM:"):
                tile, banks = _parse_sram_loc(cmd.dst)
                self.mem_mgr.allocate(cmd.object_id, tile, banks)
            out_obj.valid_in_sram = True
            out_obj.mark_dirty()
        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_ew(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = max(1, count // self.config.sram_pim.pim.lanes_per_bank)
        self.energy.add_pim_mac(count)
        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_reduce(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.reduce_latency_cycles
        self.energy.add_pim_reduce(count)
        # Register output object so downstream commands can read it
        if cmd.object_id not in self.mem_mgr.objects:
            out_obj = MemoryObject(cmd.object_id, ObjType.PSUM, cmd.bytes or 1024, "int32")
            self.mem_mgr.register_object(out_obj)
            if cmd.dst and cmd.dst.startswith("SRAM:"):
                tile, banks = _parse_sram_loc(cmd.dst)
                self.mem_mgr.allocate(cmd.object_id, tile, banks)
            out_obj.valid_in_sram = True
        else:
            obj = self.mem_mgr.objects[cmd.object_id]
            obj.valid_in_sram = True
        self.latency_breakdown["pim_reduce_cycles"] += lat
        return lat

    def _issue_pim_nl(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.nonlinear_latency_cycles
        self.energy.add_pim_nl(count)
        # Register output object so downstream commands can read it
        if cmd.object_id not in self.mem_mgr.objects:
            out_obj = MemoryObject(cmd.object_id, ObjType.PSUM, cmd.bytes or 1024, "int32")
            self.mem_mgr.register_object(out_obj)
            if cmd.dst and cmd.dst.startswith("SRAM:"):
                tile, banks = _parse_sram_loc(cmd.dst)
                self.mem_mgr.allocate(cmd.object_id, tile, banks)
            out_obj.valid_in_sram = True
        else:
            obj = self.mem_mgr.objects[cmd.object_id]
            obj.valid_in_sram = True
        return lat

    def _issue_pim_writeback(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.mark_dirty()
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_write(n_accesses)
        return 1

    def _issue_free(self, cmd: TraceCommand) -> int:
        self.mem_mgr.free(cmd.object_id)
        return 0

    def run(self) -> dict:
        if not self.commands:
            return self._make_report()

        pending = {cmd.cmd_id: cmd for cmd in self.commands}
        issued = {}
        self.cycle = 0
        max_cycles = 100_000_000

        while (pending or self.event_queue) and self.cycle < max_cycles:
            # Release resources and complete events at this cycle
            self.resource.release_at(self.cycle)
            while self.event_queue and self.event_queue[0][0] <= self.cycle:
                _, cmd_id = heapq.heappop(self.event_queue)
                self.completed.add(cmd_id)

            # Find commands whose dependencies are met
            deps_ready = [cid for cid, cmd in pending.items() if self._deps_ready(cmd)]

            # Among those, check resource availability and issue if possible
            issued_this_cycle = False
            resource_stalled = []
            for cid in deps_ready:
                cmd = pending[cid]
                banks = self._get_banks(cmd)
                if self.resource.can_issue(cmd.op, banks, cmd.bytes):
                    pending.pop(cid)
                    latency = self._issue_command(cmd)
                    if latency > 0:
                        self.resource.reserve(cmd.op, banks, cmd.bytes, latency)
                        heapq.heappush(self.event_queue, (self.cycle + latency, cid))
                    else:
                        self.completed.add(cid)
                    issued_this_cycle = True
                else:
                    resource_stalled.append(cid)

            # Count stall cycles: any cycle where at least one command is blocked by resources
            if resource_stalled:
                self.latency_breakdown["bank_conflict_stall_cycles"] += 1

            # Add leakage for this cycle
            self.energy.add_leakage(1)

            if not self.event_queue and not any(
                self._deps_ready(cmd) for cmd in pending.values()
            ):
                if pending:
                    # Deadlock or unreachable deps -- force complete remaining
                    for cid in list(pending.keys()):
                        self.completed.add(cid)
                    pending.clear()
                break

            self.cycle += 1

        return self._make_report()

    def _make_report(self) -> dict:
        eb = self.energy.get_breakdown()
        freq = self.config.system.frequency_hz
        total_ns = self.cycle * 1e9 / freq

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
            "pim": {
                "mac_count": eb["pim_mac_count"],
                "reduce_count": eb["pim_reduce_count"],
            },
        }
