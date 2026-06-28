import pytest
from src.memory_manager import MemoryManager
from src.memory_object import MemoryObject, ObjType, ObjState


from src.config import SRAMPIMConfig


def make_config():
    return SRAMPIMConfig(tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64)


def test_allocate_object():
    """Allocate reserves space but does NOT make data valid (P0-01)."""
    mgr = MemoryManager(make_config())
    obj = MemoryObject("W0", ObjType.WEIGHT, 8192, "int8")
    mgr.register_object(obj)
    ok = mgr.allocate("W0", sram_tile=0, sram_banks=[0, 1])
    assert ok is True
    assert mgr.get_object("W0").valid_in_sram is False
    assert mgr.get_object("W0").state == ObjState.SRAM_RESERVED


def test_allocate_make_valid():
    """Allocate with make_valid=True for preloaded objects."""
    mgr = MemoryManager(make_config())
    obj = MemoryObject("W0", ObjType.WEIGHT, 8192, "int8")
    mgr.register_object(obj)
    ok = mgr.allocate("W0", sram_tile=0, sram_banks=[0, 1], make_valid=True)
    assert ok is True
    assert mgr.get_object("W0").valid_in_sram is True
    assert mgr.get_object("W0").state == ObjState.VALID_CLEAN


def test_allocate_exceeds_capacity():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("big", ObjType.ACTIVATION, 33 * 1024, "int8")
    mgr.register_object(obj)
    ok = mgr.allocate("big", sram_tile=0, sram_banks=[0, 1, 2, 3])
    assert ok is False


def test_free_object():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", sram_tile=0, sram_banks=[0])
    mgr.free("X0")
    assert mgr.get_object("X0").valid_in_sram is False


def test_free_bytes_tracking():
    cfg = make_config()
    mgr = MemoryManager(cfg)
    total_per_tile = cfg.banks_per_tile * cfg.bank_capacity_kb * 1024
    assert mgr.get_free_bytes(0) == total_per_tile
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", sram_tile=0, sram_banks=[0])
    assert mgr.get_free_bytes(0) == total_per_tile - 4096


def test_eviction_candidate():
    mgr = MemoryManager(make_config())
    obj1 = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    obj2 = MemoryObject("W0", ObjType.WEIGHT, 4096, "int8", pinned=True)
    mgr.register_object(obj1)
    mgr.register_object(obj2)
    # Must make objects valid for them to be eviction candidates
    mgr.allocate("X0", 0, [0], make_valid=True)
    mgr.allocate("W0", 0, [1], make_valid=True)
    candidates = mgr.find_eviction_candidate(tile=0, needed_bytes=4096)
    assert "X0" in candidates
    assert "W0" not in candidates


def test_bank_capacity_overflow():
    """P1-10: Bank-level capacity should be checked."""
    mgr = MemoryManager(make_config())
    # Bank capacity = 8KB. Two 6KB objects on same bank should fail.
    obj1 = MemoryObject("X0", ObjType.ACTIVATION, 6000, "int8")
    obj2 = MemoryObject("X1", ObjType.ACTIVATION, 6000, "int8")
    mgr.register_object(obj1)
    mgr.register_object(obj2)
    assert mgr.allocate("X0", 0, [0]) is True
    assert mgr.allocate("X1", 0, [0]) is False


def test_stats_tracking():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", 0, [0])
    assert mgr.stats["alloc_bytes"] == 4096
