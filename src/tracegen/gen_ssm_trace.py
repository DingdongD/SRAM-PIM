from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4, "fp32": 4}.get(precision, 1)


def gen_ssm_trace(batch: int, seq_len: int, d_model: int, d_state: int,
                  sram_config: SRAMPIMConfig, precision: str = "int8",
                  mode: str = "warm_resident") -> list:
    """Generate SSM/Mamba selective scan trace.

    Models the recurrence: h_t = A * h_{t-1} + B * u_t, y_t = C * h_t + D * u_t

    Args:
        batch: batch size
        seq_len: sequence length (number of timesteps)
        d_model: model dimension
        d_state: SSM state dimension
        sram_config: SRAM-PIM hardware configuration
        precision: data precision string
        mode: "warm_resident" (weights pre-loaded) or "cold_start" (DMA load weights)

    Returns:
        List of TraceCommand objects representing the SSM scan.
    """
    dbyte = _precision_bytes(precision)
    cmds = []
    cmd_id = 0
    dram_addr = 0x5000_0000

    # Allocate (and optionally DMA-load) SSM parameters A, B, C, D
    param_ids = {}
    for name in ["A", "B", "C", "D"]:
        p_id = f"SSM_{name}"
        p_bytes = d_model * d_state * dbyte if name != "D" else d_model * dbyte
        tile = cmd_id % sram_config.tiles
        preloaded = (mode == "warm_resident")

        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, p_id, "-",
                    f"SRAM:T{tile}:B0-1", p_bytes,
                    {"pinned": True, "type": "WEIGHT", "preloaded": preloaded}, []))
        alloc_id = cmd_id
        cmd_id += 1

        if mode == "cold_start":
            cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, p_id,
                        f"DRAM:0x{dram_addr:X}", f"SRAM:T{tile}:B0-1",
                        p_bytes, {}, [alloc_id]))
            param_ids[name] = cmd_id
            cmd_id += 1
            dram_addr += p_bytes
        else:
            param_ids[name] = alloc_id

    # Initialize h_0 state (zeros)
    h_bytes = batch * d_model * d_state * dbyte
    h_prev_id = "h_0"
    tile = cmd_id % sram_config.tiles
    cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, h_prev_id, "-",
                f"SRAM:T{tile}:B0-3", h_bytes, {"type": "STATE"}, []))
    h_alloc = cmd_id
    cmd_id += 1

    prev_dep = h_alloc

    # Sequential scan over timesteps
    for t in range(seq_len):
        u_id = f"u_{t}"
        u_bytes = batch * d_model * dbyte
        u_tile = cmd_id % sram_config.tiles

        # Allocate and load input u_t
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, u_id, "-",
                    f"SRAM:T{u_tile}:B0-0", u_bytes, {"type": "ACTIVATION"}, []))
        u_alloc = cmd_id
        cmd_id += 1

        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, u_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{u_tile}:B0-0",
                    u_bytes, {}, [u_alloc]))
        u_load = cmd_id
        cmd_id += 1
        dram_addr += u_bytes

        count = batch * d_model * d_state

        # A * h_{t-1} element-wise (depends on previous timestep state)
        ah_id = f"Ah_{t}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, ah_id,
                    f"SSM_A,{h_prev_id}", f"SRAM:T{u_tile}:B1-1",
                    0, {"count": count, "op": "MUL"},
                    [param_ids["A"], prev_dep]))
        ah_cmd = cmd_id
        cmd_id += 1

        # B * u_t element-wise
        bu_id = f"Bu_{t}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, bu_id,
                    f"SSM_B,{u_id}", f"SRAM:T{u_tile}:B2-2",
                    0, {"count": count, "op": "MUL"},
                    [param_ids["B"], u_load]))
        bu_cmd = cmd_id
        cmd_id += 1

        # h_t = Ah + Bu
        h_id = f"h_{t + 1}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, h_id,
                    f"{ah_id},{bu_id}", f"SRAM:T{u_tile}:B3-3",
                    0, {"count": count, "op": "ADD"},
                    [ah_cmd, bu_cmd]))
        prev_dep = cmd_id
        h_prev_id = h_id
        cmd_id += 1

        # y_t = C * h_t + D * u_t (output projection)
        y_id = f"y_{t}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, y_id,
                    f"SSM_C,{h_id}", f"SRAM:T{u_tile}:B4-4",
                    0, {"count": batch * d_model, "op": "MUL_ADD"},
                    [param_ids["C"], prev_dep, param_ids["D"], u_load]))
        y_cmd = cmd_id
        cmd_id += 1

        # Store y_t back to DRAM
        y_bytes = batch * d_model * dbyte
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, y_id,
                    f"SRAM:T{u_tile}:B4-4", f"DRAM:0x{dram_addr:X}",
                    y_bytes, {}, [y_cmd]))
        cmd_id += 1
        dram_addr += y_bytes

    return cmds
