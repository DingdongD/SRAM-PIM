from src.tracegen.gen_gemm_trace import gen_gemm_trace, _precision_bytes
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def gen_ffn_trace(batch: int, seq_len: int, d_model: int, d_ff: int,
                  sram_config: SRAMPIMConfig, precision: str = "int8",
                  mode: str = "warm_resident") -> list:
    """Generate Transformer FFN trace with up-projection, activation, and down-projection.

    Models: H = activation(X @ W1),  O = H @ W2
    For gated FFN: H = (X @ W_gate * sigmoid(X @ W_gate)) * (X @ W_up), O = H @ W_down

    Args:
        batch: batch size
        seq_len: sequence length
        d_model: input/output model dimension
        d_ff: feed-forward hidden dimension
        sram_config: SRAM-PIM hardware configuration
        precision: data precision string
        mode: "warm_resident" or "cold_start"

    Returns:
        List of TraceCommand objects representing the FFN computation.
    """
    M = batch * seq_len
    Tm = min(64, M)
    Tn_up = min(64, d_ff)
    Tn_down = min(64, d_model)
    Tk_up = min(128, d_model)
    Tk_down = min(128, d_ff)

    # --- Up-projection: X[M, d_model] @ W1[d_model, d_ff] -> H[M, d_ff] ---
    up_cmds = gen_gemm_trace(M=M, N=d_ff, K=d_model,
                             Tm=Tm, Tn=Tn_up, Tk=Tk_up,
                             sram_config=sram_config, precision=precision, mode=mode)

    # Determine the next available cmd_id after up_cmds
    max_id = max(c.cmd_id for c in up_cmds) + 1

    # Find the last PIM_MAC in the up-projection to depend the activation on
    last_up_mac = max(c.cmd_id for c in up_cmds if c.op == OpCode.PIM_MAC)

    # --- Activation (GeLU/SiLU) applied element-wise to H ---
    nl_count = M * d_ff
    nl_cmd = TraceCommand(max_id, OpCode.PIM_NL, "FFN_act",
                          "Y_m0_n0", "SRAM:T0:B0-1",
                          0, {"count": nl_count, "op": "gelu"},
                          [last_up_mac])
    max_id += 1

    # --- Down-projection: H[M, d_ff] @ W2[d_ff, d_model] -> O[M, d_model] ---
    down_cmds_raw = gen_gemm_trace(M=M, N=d_model, K=d_ff,
                                   Tm=Tm, Tn=Tn_down, Tk=Tk_down,
                                   sram_config=sram_config, precision=precision, mode=mode)

    # Remap cmd_ids to avoid collision with up_cmds, and fix up internal deps
    id_offset = max_id
    old_to_new = {}
    for c in down_cmds_raw:
        old_to_new[c.cmd_id] = c.cmd_id + id_offset

    for c in down_cmds_raw:
        c.cmd_id = old_to_new[c.cmd_id]
        c.deps = [old_to_new.get(d, d + id_offset) for d in c.deps]

    # Rename object IDs to avoid collision with up_cmds objects
    for c in down_cmds_raw:
        if c.object_id.startswith("W_"):
            c.object_id = "FFN_down_" + c.object_id
        elif c.object_id.startswith("X_"):
            c.object_id = "FFN_down_" + c.object_id
        elif c.object_id.startswith("Y_"):
            c.object_id = "FFN_out_" + c.object_id

    # First command of down-projection depends on the activation completing
    if down_cmds_raw:
        down_cmds_raw[0].deps.append(nl_cmd.cmd_id)

    return up_cmds + [nl_cmd] + down_cmds_raw
