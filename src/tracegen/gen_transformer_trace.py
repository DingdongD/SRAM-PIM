from src.tracegen.gen_attention_trace import gen_attention_trace
from src.tracegen.gen_ffn_trace import gen_ffn_trace
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def gen_transformer_trace(model_config: dict, sram_config: SRAMPIMConfig,
                          precision: str = "int8", mode: str = "warm_resident",
                          stage: str = "generation") -> list:
    """Generate a full Transformer model trace spanning multiple decoder layers.

    Each layer is composed of:
      - Multi-head attention (gen_attention_trace)
      - Feed-forward network (gen_ffn_trace)

    cmd_ids are globally unique across all layers. Inter-layer and intra-layer
    dependencies are correctly wired: FFN depends on attention output; layer N+1
    depends on layer N's last command.

    Args:
        model_config: dict with keys:
            name, ndec, hdim, num_heads, d_head, ff_scale, batch, seq_len, gen_len
        sram_config: SRAMPIMConfig hardware description
        precision: data precision string (e.g. "int8", "fp16")
        mode: "warm_resident" or "cold_start"
        stage: "generation" (decode, q_len=1) or "summarization" (prefill)

    Returns:
        List of TraceCommand objects with unique cmd_ids.
    """
    ndec = model_config["ndec"]
    hdim = model_config["hdim"]
    num_heads = model_config["num_heads"]
    d_head = model_config["d_head"]
    ff_scale = model_config["ff_scale"]
    batch = model_config["batch"]
    seq_len = model_config["seq_len"]
    d_ff = int(hdim * ff_scale)
    q_len = 1 if stage == "generation" else seq_len

    all_cmds: list[TraceCommand] = []
    global_id = 0
    # Track last cmd_id of the previous layer (for inter-layer dependency)
    prev_layer_last_id: int | None = None

    for layer in range(ndec):
        # ------------------------------------------------------------------ #
        # Attention sub-trace
        # ------------------------------------------------------------------ #
        attn_cmds = gen_attention_trace(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            d_head=d_head,
            sram_config=sram_config,
            precision=precision,
            mode=mode,
            stage=stage,
        )

        # Remap cmd_ids and internal deps, prefix object_ids with layer tag
        attn_id_map: dict[int, int] = {}
        # Collect all object_ids so we can rename src references too
        attn_obj_ids = {c.object_id for c in attn_cmds}
        for c in attn_cmds:
            for tok in c.src.split(","):
                tok = tok.strip()
                if (tok and tok != "-" and not tok.startswith("SRAM:")
                        and not tok.startswith("DRAM:")):
                    attn_obj_ids.add(tok)
        attn_obj_rename = {oid: f"L{layer}_{oid}" for oid in attn_obj_ids}

        for c in attn_cmds:
            old_id = c.cmd_id
            attn_id_map[old_id] = global_id
            c.cmd_id = global_id
            global_id += 1

        for c in attn_cmds:
            c.deps = [attn_id_map[d] for d in c.deps if d in attn_id_map]
            c.object_id = attn_obj_rename.get(c.object_id, c.object_id)
            # Rename object references in src field (comma-separated)
            src_parts = [s.strip() for s in c.src.split(",")]
            c.src = ",".join(attn_obj_rename.get(s, s) for s in src_parts)

        # First attention command depends on the last command of the previous layer
        if prev_layer_last_id is not None and attn_cmds:
            if prev_layer_last_id not in attn_cmds[0].deps:
                attn_cmds[0].deps.append(prev_layer_last_id)

        all_cmds.extend(attn_cmds)
        last_attn_id = attn_cmds[-1].cmd_id if attn_cmds else prev_layer_last_id

        # ------------------------------------------------------------------ #
        # FFN sub-trace
        # ------------------------------------------------------------------ #
        ffn_cmds = gen_ffn_trace(
            batch=batch,
            seq_len=q_len,
            d_model=hdim,
            d_ff=d_ff,
            sram_config=sram_config,
            precision=precision,
            mode=mode,
        )

        # Remap cmd_ids and internal deps, prefix object_ids with layer+FFN tag
        ffn_id_map: dict[int, int] = {}
        ffn_prefix = f"L{layer}_FFN_"
        # Collect all object names from both object_id fields AND src references
        # Filter out SRAM/DRAM location strings which are not object IDs
        ffn_obj_ids = {c.object_id for c in ffn_cmds}
        for c in ffn_cmds:
            for tok in c.src.split(","):
                tok = tok.strip()
                if (tok and tok != "-" and not tok.startswith("SRAM:")
                        and not tok.startswith("DRAM:")):
                    ffn_obj_ids.add(tok)
        ffn_obj_rename = {oid: f"{ffn_prefix}{oid}" for oid in ffn_obj_ids}

        for c in ffn_cmds:
            old_id = c.cmd_id
            ffn_id_map[old_id] = global_id
            c.cmd_id = global_id
            global_id += 1

        for c in ffn_cmds:
            c.deps = [ffn_id_map[d] for d in c.deps if d in ffn_id_map]
            c.object_id = ffn_obj_rename.get(c.object_id, c.object_id)
            # Rename object references in src field (comma-separated)
            src_parts = [s.strip() for s in c.src.split(",")]
            c.src = ",".join(ffn_obj_rename.get(s, s) for s in src_parts)

        # First FFN command depends on the last attention command of this layer
        if last_attn_id is not None and ffn_cmds:
            if last_attn_id not in ffn_cmds[0].deps:
                ffn_cmds[0].deps.append(last_attn_id)

        all_cmds.extend(ffn_cmds)
        prev_layer_last_id = ffn_cmds[-1].cmd_id if ffn_cmds else last_attn_id

    return all_cmds
