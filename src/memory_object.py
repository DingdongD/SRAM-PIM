from dataclasses import dataclass, field
from enum import Enum


class ObjType(Enum):
    WEIGHT = "WEIGHT"
    ACTIVATION = "ACTIVATION"
    STATE = "STATE"
    PSUM = "PSUM"
    LUT = "LUT"
    META = "META"
    OUTPUT = "OUTPUT"
    COORD = "COORD"
    DIST = "DIST"


class ObjState(Enum):
    DRAM_ONLY = "DRAM_ONLY"
    SRAM_RESERVED = "SRAM_RESERVED"
    LOADING = "LOADING"
    VALID_CLEAN = "VALID_CLEAN"
    VALID_DIRTY = "VALID_DIRTY"
    PRODUCING = "PRODUCING"
    SPILLING = "SPILLING"
    EVICTED = "EVICTED"
    POWER_GATED = "POWER_GATED"


@dataclass
class MemoryObject:
    object_id: str
    obj_type: ObjType
    bytes: int
    precision: str
    dram_addr: int = 0
    sram_tile: int = -1
    sram_banks: list = field(default_factory=list)
    valid_in_dram: bool = True
    valid_in_sram: bool = False
    dirty_in_sram: bool = False
    pinned: bool = False
    power_state: str = "active"
    state: ObjState = ObjState.DRAM_ONLY
    first_use: str = ""
    last_use: str = ""
    reuse_count: int = 0

    def place_in_sram(self, sram_tile: int, sram_banks: list):
        """Reserve SRAM space. Does NOT make data valid."""
        self.sram_tile = sram_tile
        self.sram_banks = list(sram_banks)
        self.power_state = "active"
        self.state = ObjState.SRAM_RESERVED
        # valid_in_sram remains False until commit_load() or mark_dirty()

    def begin_loading(self):
        """Mark object as being loaded from DRAM (DMA_LOAD issued)."""
        self.state = ObjState.LOADING

    def commit_load(self):
        """Complete DMA_LOAD: data is now valid and clean in SRAM."""
        self.valid_in_sram = True
        self.dirty_in_sram = False
        self.power_state = "active"
        self.state = ObjState.VALID_CLEAN

    def begin_producing(self):
        """Mark output as being computed (PIM issued)."""
        self.state = ObjState.PRODUCING

    def commit_produce(self):
        """Complete PIM operation: output is now valid and dirty in SRAM."""
        self.valid_in_sram = True
        self.dirty_in_sram = True
        self.valid_in_dram = False
        self.state = ObjState.VALID_DIRTY

    def mark_dirty(self):
        """Mark data as modified in SRAM (DRAM copy becomes stale)."""
        self.valid_in_sram = True
        self.dirty_in_sram = True
        self.valid_in_dram = False
        self.state = ObjState.VALID_DIRTY

    def begin_spilling(self):
        """Mark object as being written back to DRAM."""
        self.state = ObjState.SPILLING

    def writeback_complete(self):
        """Complete DMA_STORE: DRAM now has valid copy."""
        self.valid_in_dram = True
        self.dirty_in_sram = False
        self.state = ObjState.VALID_CLEAN

    def evict(self) -> bool:
        """Remove from SRAM. Returns True if writeback needed first."""
        if self.pinned:
            raise RuntimeError(f"Cannot evict pinned object {self.object_id}")
        if self.dirty_in_sram:
            raise RuntimeError(
                f"Cannot evict dirty object {self.object_id} without writeback")
        self.valid_in_sram = False
        self.dirty_in_sram = False
        self.sram_tile = -1
        self.sram_banks = []
        self.state = ObjState.EVICTED
        return False

    def power_gate(self):
        if self.dirty_in_sram:
            raise RuntimeError(
                f"Cannot power-gate dirty object {self.object_id}")
        self.valid_in_sram = False
        self.power_state = "power_gated"
        self.sram_tile = -1
        self.sram_banks = []
        self.state = ObjState.POWER_GATED

    def wakeup(self):
        self.power_state = "active"

    # Legacy compatibility
    def load_to_sram(self, sram_tile: int, sram_banks: list):
        """Legacy: place + make valid. Used only for preloaded/warm objects."""
        self.place_in_sram(sram_tile, sram_banks)
        self.valid_in_sram = True
        self.state = ObjState.VALID_CLEAN
