from src.memory_object import MemoryObject, ObjType
from src.config import SRAMPIMConfig


class MemoryManager:
    def __init__(self, config: SRAMPIMConfig):
        self.config = config
        self.objects: dict[str, MemoryObject] = {}
        self.tile_usage: dict[int, int] = {}
        self.tile_capacity: dict[int, int] = {}
        for t in range(config.tiles):
            cap = config.banks_per_tile * config.bank_capacity_kb * 1024
            self.tile_capacity[t] = cap
            self.tile_usage[t] = 0
        self.stats = {
            "alloc_bytes": 0,
            "free_bytes": 0,
            "spill_count": 0,
            "writeback_count": 0,
            "eviction_count": 0,
        }

    def register_object(self, obj: MemoryObject) -> None:
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> MemoryObject:
        return self.objects[object_id]

    def allocate(self, object_id: str, sram_tile: int, sram_banks: list) -> bool:
        obj = self.objects[object_id]
        if self.tile_usage[sram_tile] + obj.bytes > self.tile_capacity[sram_tile]:
            return False
        obj.load_to_sram(sram_tile, sram_banks)
        self.tile_usage[sram_tile] += obj.bytes
        self.stats["alloc_bytes"] += obj.bytes
        return True

    def free(self, object_id: str) -> None:
        obj = self.objects[object_id]
        if obj.valid_in_sram and obj.sram_tile >= 0:
            self.tile_usage[obj.sram_tile] -= obj.bytes
            self.stats["free_bytes"] += obj.bytes
        obj.valid_in_sram = False
        obj.sram_tile = -1
        obj.sram_banks = []

    def get_free_bytes(self, tile: int) -> int:
        return self.tile_capacity[tile] - self.tile_usage[tile]

    def find_eviction_candidate(self, tile: int, needed_bytes: int) -> list:
        candidates = []
        freed = 0
        resident = [
            oid for oid, obj in self.objects.items()
            if obj.valid_in_sram and obj.sram_tile == tile and not obj.pinned
        ]
        # Sort: clean activations first, then clean weights, then dirty
        def evict_priority(oid):
            obj = self.objects[oid]
            type_pri = 0 if obj.obj_type == ObjType.ACTIVATION else 1
            dirty_pri = 0 if not obj.dirty_in_sram else 1
            return (dirty_pri, type_pri)

        resident.sort(key=evict_priority)
        for oid in resident:
            if freed >= needed_bytes:
                break
            candidates.append(oid)
            freed += self.objects[oid].bytes
        return candidates

    def check_valid_for_read(self, object_id: str) -> None:
        obj = self.objects[object_id]
        if not obj.valid_in_sram:
            raise RuntimeError(f"Object {object_id} not valid in SRAM, cannot read/PIM")
