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

    # Build a rename map for down-projection object IDs, then apply to both
    # object_id fields and src references so the simulator can track residency.
    down_obj_ids: set[str] = set()
    for c in down_cmds_raw:
        down_obj_ids.add(c.object_id)
        for tok in c.src.split(","):
            tok = tok.strip()
            if tok and tok != "-" and not tok.startswith("DRAM:") and not tok.startswith("SRAM:"):
                down_obj_ids.add(tok)

    def _down_rename(name: str) -> str:
        if name.startswith("W_"):
            return "FFN_down_" + name
        if name.startswith("X_"):
            return "FFN_down_" + name
        if name.startswith("Y_"):
            return "FFN_out_" + name
        return name

    down_rename_map = {oid: _down_rename(oid) for oid in down_obj_ids}

    # Rename object IDs to avoid collision with up_cmds objects
    for c in down_cmds_raw:
        c.object_id = down_rename_map.get(c.object_id, c.object_id)
        src_parts = [s.strip() for s in c.src.split(",")]
        c.src = ",".join(down_rename_map.get(s, s) for s in src_parts)

    # First command of down-projection depends on the activation completing
    if down_cmds_raw:
        down_cmds_raw[0].deps.append(nl_cmd.cmd_id)

    return up_cmds + [nl_cmd] + down_cmds_raw
