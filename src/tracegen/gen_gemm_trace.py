import math
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4, "fp32": 4}.get(precision, 1)


def gen_gemm_trace(M: int, N: int, K: int, Tm: int, Tn: int, Tk: int,
                   sram_config: SRAMPIMConfig, precision: str = "int8",
                   mode: str = "cold_start") -> list:
    dbyte = _precision_bytes(precision)
    cmds = []
    cmd_id = 0
    dram_addr = 0x1000_0000
    tile_idx = 0
    banks_per_tile = sram_config.banks_per_tile

    m_tiles = math.ceil(M / Tm)
    n_tiles = math.ceil(N / Tn)
    k_tiles = math.ceil(K / Tk)

    # Pre-allocate and load weight tiles in cold_start
    weight_load_ids = {}
    if mode == "cold_start":
        for ni in range(n_tiles):
            for ki in range(k_tiles):
                w_id = f"W_n{ni}_k{ki}"
                w_bytes = Tn * Tk * dbyte
                w_tile = tile_idx % sram_config.tiles
                w_banks_lo = (tile_idx * 4) % banks_per_tile
                w_banks = list(range(w_banks_lo, min(w_banks_lo + 4, banks_per_tile)))

                alloc_id = cmd_id
                cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, w_id, "-",
                            f"SRAM:T{w_tile}:B{w_banks[0]}-{w_banks[-1]}",
                            w_bytes, {"pinned": True, "type": "WEIGHT"}, []))
                cmd_id += 1

                load_id = cmd_id
                cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, w_id,
                            f"DRAM:0x{dram_addr:X}",
                            f"SRAM:T{w_tile}:B{w_banks[0]}-{w_banks[-1]}",
                            w_bytes, {}, [alloc_id]))
                cmd_id += 1
                dram_addr += w_bytes

                weight_load_ids[(ni, ki)] = load_id
                tile_idx += 1
    else:
        # Warm mode: weights pre-loaded, just record alloc (no DMA_LOAD for weights)
        for ni in range(n_tiles):
            for ki in range(k_tiles):
                w_id = f"W_n{ni}_k{ki}"
                w_bytes = Tn * Tk * dbyte
                w_tile = tile_idx % sram_config.tiles
                w_banks_lo = (tile_idx * 4) % banks_per_tile
                w_banks = list(range(w_banks_lo, min(w_banks_lo + 4, banks_per_tile)))

                alloc_id = cmd_id
                cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, w_id, "-",
                            f"SRAM:T{w_tile}:B{w_banks[0]}-{w_banks[-1]}",
                            w_bytes, {"pinned": True, "type": "WEIGHT", "preloaded": True}, []))
                cmd_id += 1
                weight_load_ids[(ni, ki)] = alloc_id
                tile_idx += 1

    # Compute tiles
    for mi in range(m_tiles):
        actual_m = min(Tm, M - mi * Tm)

        for ki in range(k_tiles):
            actual_k = min(Tk, K - ki * Tk)

            # Load activation tile X[m, k]
            x_id = f"X_m{mi}_k{ki}"
            x_bytes = actual_m * actual_k * dbyte
            x_tile = tile_idx % sram_config.tiles

            alloc_x_id = cmd_id
            cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, x_id, "-",
                        f"SRAM:T{x_tile}:B0-1",
                        x_bytes, {"type": "ACTIVATION"}, []))
            cmd_id += 1

            load_x_id = cmd_id
            cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, x_id,
                        f"DRAM:0x{dram_addr:X}",
                        f"SRAM:T{x_tile}:B0-1",
                        x_bytes, {}, [alloc_x_id]))
            cmd_id += 1
            dram_addr += x_bytes

            for ni in range(n_tiles):
                actual_n = min(Tn, N - ni * Tn)
                mac_count = actual_m * actual_n * actual_k

                y_id = f"Y_m{mi}_n{ni}"
                y_tile = (tile_idx + 1) % sram_config.tiles

                deps = [load_x_id, weight_load_ids[(ni, ki)]]

                cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, y_id,
                            f"{x_id},{f'W_n{ni}_k{ki}'}",
                            f"SRAM:T{y_tile}:B0-1",
                            0, {"mac_count": mac_count}, deps))
                cmd_id += 1

            # Free activation tile
            cmds.append(TraceCommand(cmd_id, OpCode.SRAM_FREE, x_id,
                        f"SRAM:T{x_tile}", "-", x_bytes, {}, [cmd_id - 1]))
            cmd_id += 1

    # Store output tiles
    for mi in range(m_tiles):
        actual_m = min(Tm, M - mi * Tm)
        for ni in range(n_tiles):
            actual_n = min(Tn, N - ni * Tn)
            y_id = f"Y_m{mi}_n{ni}"
            y_bytes = actual_m * actual_n * dbyte

            # Find the last PIM_MAC that wrote this Y tile
            last_mac = max(c.cmd_id for c in cmds if c.op == OpCode.PIM_MAC and c.object_id == y_id)

            cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, y_id,
                        f"SRAM:T0:B0-1", f"DRAM:0x{dram_addr:X}",
                        y_bytes, {}, [last_mac]))
            cmd_id += 1
            dram_addr += y_bytes

    return cmds
