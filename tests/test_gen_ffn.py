from src.tracegen.gen_ffn_trace import gen_ffn_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig


def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)


def test_ffn_two_gemms():
    cmds = gen_ffn_trace(batch=1, seq_len=128, d_model=256, d_ff=1024,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    macs = [c for c in cmds if c.op == OpCode.PIM_MAC]
    # At least 2 GEMM layers (up-project + down-project)
    assert len(macs) >= 2


def test_ffn_has_activation():
    cmds = gen_ffn_trace(batch=1, seq_len=64, d_model=128, d_ff=512,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    nl_ops = [c for c in cmds if c.op == OpCode.PIM_NL]
    assert len(nl_ops) >= 1


def test_ffn_returns_list():
    cmds = gen_ffn_trace(batch=1, seq_len=16, d_model=64, d_ff=256,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    assert isinstance(cmds, list)
    assert len(cmds) > 0


def test_ffn_unique_cmd_ids():
    cmds = gen_ffn_trace(batch=1, seq_len=32, d_model=64, d_ff=256,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    ids = [c.cmd_id for c in cmds]
    assert len(ids) == len(set(ids)), "cmd_ids must be unique"


def test_ffn_cold_start_has_dma_loads():
    cmds = gen_ffn_trace(batch=1, seq_len=16, d_model=64, d_ff=256,
                         sram_config=make_config(), precision="int8", mode="cold_start")
    dma_loads = [c for c in cmds if c.op == OpCode.DMA_LOAD]
    assert len(dma_loads) > 0
