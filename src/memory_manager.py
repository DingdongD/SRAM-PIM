from src.memory_object import MemoryObject, ObjType, ObjState
from src.config import SRAMPIMConfig
from src.utils.math_utils import ceil_div


class MemoryManager:
    def __init__(self, config: SRAMPIMConfig):
        self.config = config
        self.objects: dict[str, MemoryObject] = {}
        # Tile-level capacity tracking
        self.tile_usage: dict[int, int] = {}
        self.tile_capacity: dict[int, int] = {}
        # Bank-level capacity tracking (P1-10)
        self.bank_usage: dict[tuple, int] = {}
        self.bank_capacity: dict[tuple, int] = {}
        bank_cap = config.bank_capacity_kb * 1024
        for t in range(config.tiles):
            tile_cap = config.banks_per_tile * bank_cap
            self.tile_capacity[t] = tile_cap
            self.tile_usage[t] = 0
            for b in range(config.banks_per_tile):
                self.bank_usage[(t, b)] = 0
                self.bank_capacity[(t, b)] = bank_cap
        self.stats = {
            "alloc_bytes": 0,
            "free_bytes": 0,
            "spill_count": 0,
            "writeback_count": 0,
            "eviction_count": 0,
            "reload_count": 0,
        }

    def register_object(self, obj: MemoryObject) -> None:
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> MemoryObject:
        return self.objects[object_id]

    def allocate(self, object_id: str, sram_tile: int, sram_banks: list,
                 make_valid: bool = False) -> bool:
        """Reserve SRAM space for an object. Does NOT make data valid unless
        make_valid=True (for preloaded/warm objects)."""
        obj = self.objects[object_id]
        # Check tile capacity
        if self.tile_usage[sram_tile] + obj.bytes > self.tile_capacity[sram_tile]:
            return False
        # Check bank capacity (P1-10) — only for valid bank indices
        valid_banks = [b for b in sram_banks
                       if (sram_tile, b) in self.bank_capacity]
        if valid_banks:
            bytes_per_bank = ceil_div(obj.bytes, len(valid_banks))
            for b in valid_banks:
                if self.bank_usage[(sram_tile, b)] + bytes_per_bank > \
                        self.bank_capacity[(sram_tile, b)]:
                    return False
        # Reserve space
        obj.place_in_sram(sram_tile, sram_banks)
        self.tile_usage[sram_tile] += obj.bytes
        if valid_banks:
            bytes_per_bank = ceil_div(obj.bytes, len(valid_banks))
            for b in valid_banks:
                self.bank_usage[(sram_tile, b)] += bytes_per_bank
        self.stats["alloc_bytes"] += obj.bytes
        if make_valid:
            obj.valid_in_sram = True
            obj.state = ObjState.VALID_CLEAN
        return True

    def free(self, object_id: str) -> None:
        """Free SRAM space. Object must not be dirty (P0-05)."""
        obj = self.objects[object_id]
        if obj.sram_tile >= 0:
            self._release_capacity(obj)
            self.stats["free_bytes"] += obj.bytes
        obj.valid_in_sram = False
        obj.dirty_in_sram = False
        obj.sram_tile = -1
        obj.sram_banks = []
        obj.state = ObjState.EVICTED

    def free_capacity_only(self, object_id: str) -> None:
        """Release capacity tracking without changing object state
        (used after evict() has already updated the object)."""
        obj = self.objects[object_id]
        if obj.sram_tile == -1:
            # Object was already evicted; use stored info if available
            return
        self._release_capacity(obj)

    def _release_capacity(self, obj: MemoryObject) -> None:
        tile = obj.sram_tile
        if tile < 0:
            return
        self.tile_usage[tile] = max(0, self.tile_usage[tile] - obj.bytes)
        valid_banks = [b for b in obj.sram_banks
                       if (tile, b) in self.bank_usage]
        if valid_banks:
            bytes_per_bank = ceil_div(obj.bytes, len(valid_banks))
            for b in valid_banks:
                self.bank_usage[(tile, b)] = max(
                    0, self.bank_usage[(tile, b)] - bytes_per_bank)

    def get_free_bytes(self, tile: int) -> int:
        return self.tile_capacity[tile] - self.tile_usage[tile]

    def find_eviction_candidate(self, tile: int, needed_bytes: int,
                                target_banks: list = None,
                                protected_ids: set = None) -> list:
        if protected_ids is None:
            protected_ids = set()
        candidates = []
        freed = 0
        resident = [
            oid for oid, obj in self.objects.items()
            if obj.valid_in_sram and obj.sram_tile == tile
            and not obj.pinned and oid not in protected_ids
        ]

        target_bank_set = set(target_banks) if target_banks else set()

        def evict_priority(oid):
            obj = self.objects[oid]
            # Prefer objects overlapping target banks (higher overlap = evict first)
            bank_overlap = len(set(obj.sram_banks) & target_bank_set) if target_bank_set else 0
            overlap_pri = -bank_overlap  # negative so higher overlap sorts first
            dirty_pri = 0 if not obj.dirty_in_sram else 1
            type_pri = 0 if obj.obj_type in (ObjType.ACTIVATION, ObjType.PSUM) else 1
            size_pri = -obj.bytes  # prefer larger objects
            return (overlap_pri, dirty_pri, type_pri, size_pri)

        resident.sort(key=evict_priority)
        for oid in resident:
            if freed >= needed_bytes:
                break
            candidates.append(oid)
            freed += self.objects[oid].bytes
        return candidates

    def count_active_banks(self) -> int:
        """Count banks that have at least one valid object."""
        active = set()
        for obj in self.objects.values():
            if obj.valid_in_sram and obj.sram_tile >= 0:
                for b in obj.sram_banks:
                    active.add((obj.sram_tile, b))
        return len(active)
