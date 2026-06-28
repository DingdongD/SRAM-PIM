import math
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4, "fp32": 4}.get(precision, 1)


def gen_attention_trace(batch: int, seq_len: int, num_heads: int, d_head: int,
                        sram_config: SRAMPIMConfig, precision: str = "int8",
                        mode: str = "warm_resident", stage: str = "generation") -> list:
    """
    Generate a trace for Transformer multi-head attention.

    Attention = Q*K^T -> Softmax -> Score*V, per head.

    stage="generation"    (decode):  q_len=1, kv_len=seq_len  (GEMV)
    stage="summarization" (prefill): q_len=seq_len, kv_len=seq_len  (GEMM)
    """
    dbyte = _precision_bytes(precision)
    cmds = []
    cmd_id = 0
    dram_addr = 0x2000_0000

    if stage == "generation":
        q_len = 1       # decode: single token query
        kv_len = seq_len
    else:
        q_len = seq_len  # prefill: full sequence
        kv_len = seq_len

    for h in range(num_heads):
        # --- Load Q (batch x q_len x d_head) ---
        q_id = f"Q_h{h}"
        q_bytes = batch * q_len * d_head * dbyte
        q_tile = h % sram_config.tiles

        alloc_q = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, q_id, "-",
                    f"SRAM:T{q_tile}:B0-1", q_bytes, {"type": "ACTIVATION"}, []))
        cmd_id += 1

        load_q = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, q_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{q_tile}:B0-1",
                    q_bytes, {}, [alloc_q]))
        cmd_id += 1
        dram_addr += q_bytes

        # --- Load K (batch x kv_len x d_head) ---
        k_id = f"K_h{h}"
        k_bytes = batch * kv_len * d_head * dbyte
        k_tile = (h + 1) % sram_config.tiles

        alloc_k = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, k_id, "-",
                    f"SRAM:T{k_tile}:B0-3", k_bytes, {"type": "STATE"}, []))
        cmd_id += 1

        load_k = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, k_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{k_tile}:B0-3",
                    k_bytes, {}, [alloc_k]))
        cmd_id += 1
        dram_addr += k_bytes

        # --- Load V (batch x kv_len x d_head) ---
        v_id = f"V_h{h}"
        v_bytes = batch * kv_len * d_head * dbyte
        v_tile = (h + 2) % sram_config.tiles

        alloc_v = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, v_id, "-",
                    f"SRAM:T{v_tile}:B0-3", v_bytes, {"type": "STATE"}, []))
        cmd_id += 1

        load_v = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, v_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{v_tile}:B0-3",
                    v_bytes, {}, [alloc_v]))
        cmd_id += 1
        dram_addr += v_bytes

        # --- Score = Q @ K^T: shape (q_len x kv_len) ---
        score_id = f"Score_h{h}"
        mac_count = batch * q_len * kv_len * d_head
        score_tile = (h + 3) % sram_config.tiles

        mac_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, score_id,
                    f"{q_id},{k_id}", f"SRAM:T{score_tile}:B0-1",
                    0, {"mac_count": mac_count}, [load_q, load_k]))
        cmd_id += 1

        # --- Softmax on score ---
        softmax_id = f"Softmax_h{h}"
        nl_count = batch * q_len * kv_len

        softmax_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_NL, softmax_id,
                    score_id, f"SRAM:T{score_tile}:B0-1",
                    0, {"count": nl_count, "op": "softmax"}, [mac_cmd]))
        cmd_id += 1

        # --- Reduce (numerical stability / partial sum across kv dimension) ---
        reduce_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_REDUCE, softmax_id,
                    softmax_id, f"SRAM:T{score_tile}:B0-1",
                    0, {"count": nl_count}, [softmax_cmd]))
        cmd_id += 1

        # --- Context = Softmax(Score) @ V: shape (q_len x d_head) ---
        context_id = f"Context_h{h}"
        ctx_mac_count = batch * q_len * d_head * kv_len

        ctx_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, context_id,
                    f"{softmax_id},{v_id}", f"SRAM:T{score_tile}:B2-3",
                    0, {"mac_count": ctx_mac_count}, [reduce_cmd, load_v]))
        cmd_id += 1

        # --- Store context output to DRAM ---
        ctx_bytes = batch * q_len * d_head * dbyte
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, context_id,
                    f"SRAM:T{score_tile}:B2-3", f"DRAM:0x{dram_addr:X}",
                    ctx_bytes, {}, [ctx_cmd]))
        cmd_id += 1
        dram_addr += ctx_bytes

    return cmds
