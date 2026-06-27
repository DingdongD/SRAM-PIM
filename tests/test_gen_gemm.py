from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig


def make_sram_config():
    return SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256)


def test_gemm_trace_has_all_phases():
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=64, Tn=64, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    ops = [c.op for c in cmds]
    assert OpCode.SRAM_ALLOC in ops
    assert OpCode.DMA_LOAD in ops
    assert OpCode.PIM_MAC in ops
    assert OpCode.DMA_STORE in ops


def test_gemm_trace_dependency_chain():
    cmds = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="cold_start")
    cmd_map = {c.cmd_id: c for c in cmds}
    # Every PIM_MAC should depend on DMA_LOADs
    for c in cmds:
        if c.op == OpCode.PIM_MAC:
            assert len(c.deps) > 0
            for d in c.deps:
                assert cmd_map[d].op in (OpCode.DMA_LOAD, OpCode.SRAM_ALLOC, OpCode.PIM_MAC)


def test_gemm_trace_cold_vs_warm():
    cold = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="cold_start")
    warm = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="warm_resident")
    cold_loads = sum(1 for c in cold if c.op == OpCode.DMA_LOAD)
    warm_loads = sum(1 for c in warm if c.op == OpCode.DMA_LOAD)
    # Warm mode skips weight loads
    assert warm_loads < cold_loads


def test_gemm_trace_tiled():
    cmds = gen_gemm_trace(M=256, N=256, K=256, Tm=64, Tn=64, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    mac_cmds = [c for c in cmds if c.op == OpCode.PIM_MAC]
    # Should have (M/Tm) * (N/Tn) * (K/Tk) = 4 * 4 * 2 = 32 MAC commands
    assert len(mac_cmds) == 32


def test_gemm_output_bytes():
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=128, Tn=128, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    stores = [c for c in cmds if c.op == OpCode.DMA_STORE]
    total_store = sum(c.bytes for c in stores)
    # Output Y[128,128] in int8 = 16384 bytes
    assert total_store == 128 * 128 * 1
