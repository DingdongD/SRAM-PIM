# SRAM-PIM Simulation Framework Design Spec

## Goal

Build a trace-driven SRAM-PIM simulation framework that models DRAM as backing store, SRAM residency with valid/dirty tracking, DMA traffic, bank conflicts, PIM compute pipelines, reduction, NoC contention, and energy breakdown — targeting LLM inference workloads (Transformer attention, SSM/Mamba, KAN, point-cloud operators).

## Architecture

```
PyTorch/ONNX Model → Layer/Tile Mapper → Unified Trace IR (DMA + SRAM + PIM)
                                              ↓
                         ┌──────────────────────┴──────────────────────┐
                    DRAM Timing Model              SRAM-PIM Event Model
                  (analytical / DRAMsim3)        (bank/PIM/NoC/reduce)
                         └──────────────────────┬──────────────────────┘
                                              ↓
                                     Energy Integrator
                                    (DESTINY + analytical)
                                              ↓
                                      Report/Breakdown
```

## Core Principles

1. Data not in SRAM cannot execute SRAM-PIM
2. Dirty data must be written back before eviction/power-gating
3. Capacity overflow triggers tiling, eviction, or spill
4. DMA can overlap PIM compute under resource constraints
5. DESTINY provides SRAM macro params only, not PIM scheduling
6. Cold-start, warm-run, amortized modes reported separately

## Key Components

### 1. Memory Object Model
Every data tile tracked as `MemoryObject` with: object_id, type (WEIGHT/ACTIVATION/STATE/PSUM/LUT/META/OUTPUT), bytes, precision, location (dram_addr, sram_tile, sram_banks), state (valid_in_dram, valid_in_sram, dirty_in_sram, pinned, power_state), lifetime (first_use, last_use, reuse_count).

### 2. Two-Level Trace ISA
- High-level: SRAM_ALLOC, SRAM_FREE, DMA_LOAD, DMA_STORE, PIM_MAC, PIM_EW_OP, PIM_REDUCE, PIM_NL, PIM_WRITEBACK, BARRIER, POWER_SET
- Lowered: DRAM READ/WRITE bursts for DRAMsim3 integration

### 3. Event-Driven Simulator
Global event queue + dependency DAG. Each cycle: complete events → update object states → find ready commands → schedule with resource constraints → issue → collect stall stats → add leakage.

### 4. Resource Model
DRAM channels, DMA engines, NoC bandwidth, SRAM tiles/banks, bank ports (read/write/pim exclusivity), PIM lanes, reduction trees, global output bus.

### 5. Three Simulation Modes
- Cold-start: all SRAM invalid, full weight/LUT preload
- Warm-resident: weights already loaded, only activations stream
- Amortized: preload cost / N queries + per-query cost

### 6. Energy Model
E_total = E_dram_read + E_dram_write + E_offchip_io + E_dma + E_sram_read + E_sram_write + E_sram_leakage + E_pim_compute + E_reduce + E_nonlinear + E_noc + E_control

SRAM params from DESTINY/CACTI; PIM compute from analytical config; DRAM from analytical or DRAMsim3.

### 7. Allocation Policies
STATIC_PINNED (weights pinned, activations stream), DYNAMIC_LRU, LIFETIME_AWARE.

### 8. YAML Configuration
system.yaml, dram.yaml, sram_pim.yaml, energy.yaml — all defined in the original spec §12.

## Phased Implementation

- Phase 1: Object table + capacity manager + analytical DMA/PIM + GEMM trace → end-to-end
- Phase 2: Bank conflicts, PIM pipeline, NoC/DMA bandwidth, dependency DAG, double buffering
- Phase 3: DRAMsim3 integration
- Phase 4: DESTINY/CACTI adapter, RTL PIM energy
- Phase 5: Attention, SSM/Mamba, KAN, point-cloud operator traces

## Reference Projects
- ATTACC simulator: trace-driven DRAM-PIM pattern, Ramulator2 integration
- DRAMsim3: C++ DRAM timing via subprocess/shared library
- DESTINY: SRAM macro parameters (latency, energy, area, leakage)
- CACTI: Cross-validation of SRAM parameters
