import math
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig
from src.utils.math_utils import ceil_div


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4, "fp32": 4}.get(precision, 1)


def gen_gemm_trace(M: int, N: int, K: int, Tm: int, Tn: int, Tk: int,
                   sram_config: SRAMPIMConfig, precision: str = "int8",
                   mode: str = "cold_start") -> list:
    dbyte = _precision_bytes(precision)
    psum_dbyte = 4  # int32 for partial sums
    cmds = []
    cmd_id = 0
    dram_addr = 0x1000_0000
    tile_idx = 0
    banks_per_tile = sram_config.banks_per_tile

    m_tiles = math.ceil(M / Tm)
    n_tiles = math.ceil(N / Tn)
    k_tiles = math.ceil(K / Tk)

    # Pre-allocate and load weight tiles
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

    # P0-07: Track last writer to each Y tile for accumulation dependency
    last_y_writer: dict[str, int] = {}

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

            mac_ids_this_k = []
            for ni in range(n_tiles):
                actual_n = min(Tn, N - ni * Tn)
                mac_count = actual_m * actual_n * actual_k
                y_id = f"Y_m{mi}_n{ni}"
                y_tile = (tile_idx + 1) % sram_config.tiles
                y_bytes_psum = actual_m * actual_n * psum_dbyte

                deps = [load_x_id, weight_load_ids[(ni, ki)]]

                # P0-07: Chain accumulation — each MAC on same Y depends on prior
                accumulate = y_id in last_y_writer
                if accumulate:
                    deps.append(last_y_writer[y_id])

                cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, y_id,
                            f"{x_id},W_n{ni}_k{ki}",
                            f"SRAM:T{y_tile}:B0-1",
                            y_bytes_psum,
                            {"mac_count": mac_count,
                             "accumulate": accumulate,
                             "out_bytes": y_bytes_psum,
                             "dtype_out": "int32"},
                            deps))
                last_y_writer[y_id] = cmd_id
                mac_ids_this_k.append(cmd_id)
                cmd_id += 1

            # Free activation tile after ALL MACs for this (m,k) are done
            cmds.append(TraceCommand(cmd_id, OpCode.SRAM_FREE, x_id,
                        f"SRAM:T{x_tile}", "-", x_bytes, {}, mac_ids_this_k))
            cmd_id += 1

    # Store output tiles
    for mi in range(m_tiles):
        actual_m = min(Tm, M - mi * Tm)
        for ni in range(n_tiles):
            actual_n = min(Tn, N - ni * Tn)
            y_id = f"Y_m{mi}_n{ni}"
            y_bytes = actual_m * actual_n * dbyte

            # DMA_STORE depends on the LAST PIM_MAC that wrote this Y
            store_dep = last_y_writer[y_id]

            cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, y_id,
                        f"SRAM:T0:B0-1", f"DRAM:0x{dram_addr:X}",
                        y_bytes, {}, [store_dep]))
            cmd_id += 1
            dram_addr += y_bytes

    return cmds
