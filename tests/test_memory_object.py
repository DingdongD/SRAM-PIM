from src.memory_object import MemoryObject, ObjType


def test_memory_object_creation():
    obj = MemoryObject(
        object_id="W_l3_tile_07", obj_type=ObjType.WEIGHT,
        bytes=65536, precision="int8"
    )
    assert obj.valid_in_dram is True
    assert obj.valid_in_sram is False
    assert obj.dirty_in_sram is False
    assert obj.pinned is False


def test_memory_object_load_to_sram():
    obj = MemoryObject("W0", ObjType.WEIGHT, 65536, "int8")
    obj.load_to_sram(sram_tile=0, sram_banks=[0, 1, 2, 3])
    assert obj.valid_in_sram is True
    assert obj.sram_tile == 0
    assert obj.sram_banks == [0, 1, 2, 3]


def test_memory_object_mark_dirty():
    obj = MemoryObject("P0", ObjType.PSUM, 1024, "int32")
    obj.load_to_sram(0, [0])
    obj.mark_dirty()
    assert obj.dirty_in_sram is True


def test_memory_object_evict_clean():
    obj = MemoryObject("X0", ObjType.ACTIVATION, 1024, "int8")
    obj.load_to_sram(0, [0])
    needs_writeback = obj.evict()
    assert needs_writeback is False
    assert obj.valid_in_sram is False


def test_memory_object_evict_dirty():
    obj = MemoryObject("P0", ObjType.PSUM, 1024, "int32")
    obj.load_to_sram(0, [0])
    obj.mark_dirty()
    needs_writeback = obj.evict()
    assert needs_writeback is True


def test_memory_object_pinned_cannot_evict():
    obj = MemoryObject("W0", ObjType.WEIGHT, 1024, "int8", pinned=True)
    obj.load_to_sram(0, [0])
    import pytest
    with pytest.raises(RuntimeError):
        obj.evict()


def test_power_gate_clears_valid():
    obj = MemoryObject("W0", ObjType.WEIGHT, 1024, "int8")
    obj.load_to_sram(0, [0])
    obj.power_gate()
    assert obj.valid_in_sram is False
    assert obj.power_state == "power_gated"
