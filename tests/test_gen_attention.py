from src.tracegen.gen_attention_trace import gen_attention_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig


def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)


def test_attention_has_score_softmax_context():
    cmds = gen_attention_trace(
        batch=1, seq_len=128, num_heads=8, d_head=64,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    ops = [c.op for c in cmds]
    assert OpCode.PIM_MAC in ops     # score = Q @ K^T
    assert OpCode.PIM_NL in ops      # softmax
    assert OpCode.PIM_REDUCE in ops  # reduction


def test_attention_qkv_load():
    cmds = gen_attention_trace(
        batch=1, seq_len=64, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="cold_start", stage="generation"
    )
    loads = [c for c in cmds if c.op == OpCode.DMA_LOAD]
    # Should load Q, K, V at minimum
    assert len(loads) >= 3


def test_attention_generation_kv_scales_with_seq():
    short = gen_attention_trace(
        batch=1, seq_len=64, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    long = gen_attention_trace(
        batch=1, seq_len=256, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    short_kv_bytes = sum(c.bytes for c in short if c.op == OpCode.DMA_LOAD and "K" in c.object_id or "V" in c.object_id)
    long_kv_bytes = sum(c.bytes for c in long if c.op == OpCode.DMA_LOAD and "K" in c.object_id or "V" in c.object_id)
    assert long_kv_bytes > short_kv_bytes


def test_attention_prefill_loads_q_full_seq():
    cmds = gen_attention_trace(
        batch=1, seq_len=64, num_heads=2, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="cold_start", stage="summarization"
    )
    q_loads = [c for c in cmds if c.op == OpCode.DMA_LOAD and "Q" in c.object_id]
    # In prefill, q_len == seq_len, so Q bytes should be seq_len * d_head
    total_q_bytes = sum(c.bytes for c in q_loads)
    expected_bytes_per_head = 1 * 64 * 32 * 1  # batch * seq_len * d_head * 1byte
    assert total_q_bytes >= expected_bytes_per_head


def test_attention_output_writeback():
    cmds = gen_attention_trace(
        batch=1, seq_len=32, num_heads=2, d_head=16,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    stores = [c for c in cmds if c.op == OpCode.DMA_STORE]
    # Should write back at least one context output per head
    assert len(stores) >= 2


def test_attention_per_head_independence():
    cmds = gen_attention_trace(
        batch=1, seq_len=32, num_heads=4, d_head=16,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    mac_cmds = [c for c in cmds if c.op == OpCode.PIM_MAC]
    # Each head generates at least 2 PIM_MAC ops (QK^T and Score*V)
    assert len(mac_cmds) >= 8


def test_attention_fp16_bytes():
    cmds_int8 = gen_attention_trace(
        batch=1, seq_len=32, num_heads=2, d_head=16,
        sram_config=make_config(), precision="int8",
        mode="cold_start", stage="generation"
    )
    cmds_fp16 = gen_attention_trace(
        batch=1, seq_len=32, num_heads=2, d_head=16,
        sram_config=make_config(), precision="fp16",
        mode="cold_start", stage="generation"
    )
    int8_bytes = sum(c.bytes for c in cmds_int8 if c.op == OpCode.DMA_LOAD)
    fp16_bytes = sum(c.bytes for c in cmds_fp16 if c.op == OpCode.DMA_LOAD)
    assert fp16_bytes == 2 * int8_bytes
