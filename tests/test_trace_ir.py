import tempfile, os
from src.trace_ir import OpCode, TraceCommand, parse_trace, write_trace


def test_opcode_enum():
    assert OpCode.DMA_LOAD.name == "DMA_LOAD"
    assert OpCode.PIM_MAC.name == "PIM_MAC"


def test_trace_command_creation():
    cmd = TraceCommand(
        cmd_id=0, op=OpCode.DMA_LOAD, object_id="W0_tile0",
        src="DRAM:0x1000", dst="SRAM:T0:B0-7",
        bytes=65536, attrs={"stream": "weight"}, deps=[]
    )
    assert cmd.op == OpCode.DMA_LOAD
    assert cmd.bytes == 65536


def test_trace_roundtrip():
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0_tile0", "-", "SRAM:T0:B0-7", 65536, {"pinned": 1}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0_tile0", "DRAM:0x1000", "SRAM:T0:B0-7", 65536, {"stream": "weight"}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0_psum", "W0_tile0,X0_tile0", "SRAM:T2", 0, {"mode": "bank"}, [1]),
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trace', delete=False) as f:
        path = f.name
    try:
        write_trace(cmds, path)
        loaded = parse_trace(path)
        assert len(loaded) == 3
        assert loaded[0].op == OpCode.SRAM_ALLOC
        assert loaded[1].deps == [0]
        assert loaded[2].object_id == "Y0_psum"
    finally:
        os.unlink(path)
