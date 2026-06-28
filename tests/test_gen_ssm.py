from src.tracegen.gen_ssm_trace import gen_ssm_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig


def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)


def test_ssm_has_sequential_dependency():
    cmds = gen_ssm_trace(batch=1, seq_len=8, d_model=64, d_state=16,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    ew_ops = [c for c in cmds if c.op == OpCode.PIM_EW_OP]
    # Each timestep depends on previous
    for i in range(1, len(ew_ops)):
        assert any(d < ew_ops[i].cmd_id for d in ew_ops[i].deps)


def test_ssm_state_persistence():
    cmds = gen_ssm_trace(batch=1, seq_len=4, d_model=32, d_state=8,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    # Should have state objects (h_t)
    state_ops = [c for c in cmds if "h_" in c.object_id]
    assert len(state_ops) > 0


def test_ssm_returns_list():
    cmds = gen_ssm_trace(batch=1, seq_len=2, d_model=16, d_state=4,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    assert isinstance(cmds, list)
    assert len(cmds) > 0


def test_ssm_cold_start_has_dma_loads():
    cmds = gen_ssm_trace(batch=1, seq_len=2, d_model=16, d_state=4,
                         sram_config=make_config(), precision="int8", mode="cold_start")
    dma_loads = [c for c in cmds if c.op == OpCode.DMA_LOAD]
    assert len(dma_loads) > 0


def test_ssm_unique_cmd_ids():
    cmds = gen_ssm_trace(batch=1, seq_len=4, d_model=32, d_state=8,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    ids = [c.cmd_id for c in cmds]
    assert len(ids) == len(set(ids)), "cmd_ids must be unique"
