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
    first_use: str = ""
    last_use: str = ""
    reuse_count: int = 0

    def load_to_sram(self, sram_tile: int, sram_banks: list):
        self.sram_tile = sram_tile
        self.sram_banks = list(sram_banks)
        self.valid_in_sram = True
        self.power_state = "active"

    def mark_dirty(self):
        self.dirty_in_sram = True

    def evict(self) -> bool:
        if self.pinned:
            raise RuntimeError(f"Cannot evict pinned object {self.object_id}")
        needs_writeback = self.dirty_in_sram
        self.valid_in_sram = False
        self.dirty_in_sram = False
        self.sram_tile = -1
        self.sram_banks = []
        return needs_writeback

    def writeback_complete(self):
        self.valid_in_dram = True
        self.dirty_in_sram = False

    def power_gate(self):
        if self.dirty_in_sram:
            raise RuntimeError(f"Cannot power-gate bank with dirty object {self.object_id}")
        self.valid_in_sram = False
        self.power_state = "power_gated"
        self.sram_tile = -1
        self.sram_banks = []

    def wakeup(self):
        self.power_state = "active"
