# SRAM-PIM + DRAM 严格仿真框架 Spec

> 目标：借鉴 ATTACC simulator 的 trace-driven DRAM-PIM 建模思想，构建一个可扩展到 SRAM-PIM / SRAM-CIM / near-SRAM digital PIM 的严格仿真框架。该框架显式考虑 DRAM 作为 backing store、SRAM 的易失性、上电初始化、数据驻留、容量溢出、dirty writeback、DMA/NoC/Bank 资源冲突、PIM 计算、能耗和端到端 latency。

---

## 0. 设计结论

SRAM-PIM 仿真不能只建模 `SRAM_PIM_MAC`。严格系统中至少有两个存储层级：

```text
Off-chip DRAM / HBM / LPDDR / DDR
        |
        | DMA / Memory Controller / NoC
        v
On-chip SRAM-PIM scratchpad / cache / local buffer
        |
        | bank/subarray/tile-level PIM command
        v
PIM compute + reduction + writeback
```

因此必须同时建模：

1. **DRAM 侧指令/事务**：权重、激活、KV/state、坐标、临时结果从 DRAM 读入 SRAM；最终输出、溢出的 partial sum、evicted dirty tile 写回 DRAM。
2. **SRAM 驻留状态**：SRAM 是易失性存储，上电后无有效模型/数据；持续供电期间可以保留 resident 数据；power-gating 后数据失效。
3. **SRAM-PIM 指令**：只有数据已经 valid/resident 于 SRAM 后，才能执行 SRAM-PIM 计算。
4. **容量/生命周期管理**：SRAM 容量不足时，必须 tile、prefetch、evict、spill、writeback。
5. **资源冲突与重叠**：DRAM DMA、SRAM bank access、PIM MAC、reduction、NoC、global buffer 都可能并行或冲突。
6. **能耗统计**：DRAM access/IO + SRAM read/write/leakage + PIM compute + reduction + NoC + leakage/power-state。

一句话：

```text
SRAM-PIM simulator = workload mapper + DRAM backing-store model + SRAM residency manager + PIM command scheduler + energy model
```

---

## 1. 与 ATTACC simulator 的关系

ATTACC simulator 的核心思想是：

```text
Transformer attention workload
    -> PIM command trace
    -> modified Ramulator2 HBM3-PIM timing model
    -> command count / memory_system_cycles
    -> Python energy/performance post-processing
```

SRAM-PIM 可以继承这个思想，但需要替换 DRAM-PIM 的行激活模型：

| ATTACC DRAM-PIM | SRAM-PIM 框架中的对应物 |
|---|---|
| HBM3 channel / pseudo-channel | SRAM tile / cluster / scratchpad partition |
| bank group / bank | SRAM bank / subarray |
| ACT/PRE/RD/WR/REF | SRAM read/write/allocate/free，无 ACT/PRE/REF |
| MACAB/MACSB/MACPB | tile-level / bank-level / subarray-level PIM MAC |
| WRGB/MVSB/MVGB/SFM | load local buffer / move psum / reduce / nonlinear |
| nCCDAB/nCCDSB power throttling | nMAC_TILE/nMAC_BANK/max_parallel_bank power throttling |
| Ramulator timing constraints | SRAM bank-port/NoC/reduction/buffer availability constraints |
| Python energy model | DESTINY + RTL/analytical PIM energy model |

关键区别：

- DRAM-PIM 的底层时序由 ACT、PRE、row-buffer、refresh 和 JEDEC timing 决定。
- SRAM-PIM 的底层时序主要由 bank port、subarray access、local buffer、PIM compute pipeline、reduction tree、NoC、DMA 带宽和功耗约束决定。
- SRAM 是易失性的；因此需要建模上电装载、数据驻留、power state 和失效行为。

---

## 2. 仿真边界与模式

### 2.1 仿真目标

框架输出以下指标：

```text
total_cycles
total_latency_ns
total_energy_pJ
energy_breakdown
latency_breakdown
DRAM_read_bytes
DRAM_write_bytes
SRAM_read_count
SRAM_write_count
PIM_MAC_count
PIM_reduction_count
NoC_bytes
DMA_stall_cycles
SRAM_bank_conflict_cycles
PIM_compute_stall_cycles
capacity_spill_count
writeback_count
warm_residency_hit_rate
```

### 2.2 三种运行模式

SRAM 易失性导致必须区分 cold start 与 warm resident。否则评估会不公平。

#### Mode A: Cold-start inference

用于评估完整开机后第一次推理。

```text
power_on
  -> SRAM valid bits = false
  -> load model weights / constants / LUT / state from DRAM
  -> run SRAM-PIM compute
  -> write output to DRAM or host-visible buffer
```

特点：

- 模型权重加载能耗和延迟必须计入。
- LUT、PWL 系数、常量表、position encoding 等也必须计入。
- 对小 batch 或单次推理，cold-start 代价可能非常大。

#### Mode B: Warm-resident inference

用于评估服务持续运行时的稳态推理。

```text
weights / constants already resident in SRAM
new activations loaded from DRAM or host
run SRAM-PIM compute
write final output or next-layer activation
```

特点：

- 权重预加载可以不计入单次 inference，但必须报告为 amortized cost。
- 中间 activation 是否 resident 取决于 layer pipeline 和 SRAM 容量。
- 如果 bank 被 power-gated，resident 数据失效，需要重新加载。

#### Mode C: Amortized multi-query inference

用于评估服务端连续请求。

```text
initial preload cost / number_of_queries
+ per-query activation load
+ per-query compute
+ per-query writeback
+ leakage over service time
```

推荐论文中同时报告：

```text
cold-start latency / energy
warm-run latency / energy
amortized cost over N queries
```

---

## 3. 系统架构抽象

### 3.1 顶层模块

```text
+--------------------------------------------------+
| Workload Frontend                                |
|  - ONNX / PyTorch layer parser                   |
|  - operator shape extractor                      |
|  - tile and dataflow planner                     |
+-------------------------+------------------------+
                          |
                          v
+--------------------------------------------------+
| Trace Generator                                  |
|  - DRAM_DMA_LOAD / DRAM_DMA_STORE                |
|  - SRAM_ALLOC / SRAM_FREE                        |
|  - SRAM_PIM_MAC / REDUCE / NL / WRITEBACK        |
|  - BARRIER / WAIT / PREFETCH                     |
+-------------------------+------------------------+
                          |
                          v
+--------------------------------------------------+
| System-level Event Simulator                     |
|  - global event queue                            |
|  - dependency graph                              |
|  - resource table                                |
|  - memory object table                           |
+-------------+--------------------+---------------+
              |                    |
              v                    v
+------------------------+  +-----------------------+
| DRAM Model             |  | SRAM-PIM Model         |
| - Ramulator/DRAMSim3   |  | - bank/subarray ports  |
| - analytical BW model  |  | - PIM pipeline         |
| - DRAM energy          |  | - reduction/NoC        |
+------------------------+  +-----------------------+
              |                    |
              +----------+---------+
                         v
+--------------------------------------------------+
| Energy / Area / Report                           |
| - DESTINY SRAM array params                      |
| - RTL/analytical PIM params                      |
| - DRAM energy params                             |
| - breakdown reports                              |
+--------------------------------------------------+
```

### 3.2 建议目录结构

```text
sram_pim_sim/
  README.md
  configs/
    system.yaml
    dram.yaml
    sram_pim.yaml
    destiny_sram_params.yaml
    energy.yaml
  traces/
    example_gemm.trace
    example_pointmamba.trace
    example_pointkan.trace
  src/
    frontend/
      layer_parser.py
      shape_infer.py
    mapper/
      tile_mapper.py
      dataflow_planner.py
      lifetime_analyzer.py
    tracegen/
      trace_ir.py
      gen_gemm_trace.py
      gen_pointcloud_trace.py
      lower_to_dram_trace.py
    sim/
      event.py
      simulator.py
      dram_model.py
      sram_pim_model.py
      memory_manager.py
      scheduler.py
      dependency.py
    energy/
      destiny_adapter.py
      pim_energy.py
      dram_energy.py
      noc_energy.py
      report.py
    tests/
      test_capacity.py
      test_dma_overlap.py
      test_bank_conflict.py
      test_writeback.py
```

---

## 4. Memory Object Model

所有数据都用 object 管理，而不是只用裸地址。这样可以严格处理 valid/dirty/resident/eviction。

### 4.1 Object 类型

```text
WEIGHT        model weight tile
ACTIVATION    input/output activation tile
STATE         SSM hidden state, KV cache, recurrent state
COORD         point cloud coordinate tile
DIST          distance / MDT / KNN temporary table
PSUM          partial sum buffer
LUT           activation LUT / PWL coefficient / GRF coefficient
META          top-k index / mask / pointer / segment metadata
OUTPUT        final output visible to host or next stage
```

### 4.2 Object table 字段

```yaml
object_id: W_l3_tile_07
type: WEIGHT
bytes: 65536
precision: int8
shape: [128, 512]
layout: tile_m_major
location:
  dram_addr: 0x0000_8000_0000
  sram_tile: 3
  sram_bank_range: [0, 7]
  sram_offset: 0x2000
state:
  valid_in_dram: true
  valid_in_sram: true
  dirty_in_sram: false
  pinned: true
  power_state: active
lifetime:
  first_use: layer3.op2
  last_use: layer3.op9
  reuse_count: 16
policy:
  evictable: false
  reloadable: true
```

### 4.3 Valid / Dirty 规则

| 状态 | 含义 | 行为 |
|---|---|---|
| `valid_in_dram=true` | DRAM 中有最新或可恢复版本 | 可从 DRAM load 到 SRAM |
| `valid_in_sram=true` | SRAM 中有有效副本 | 可执行 SRAM-PIM |
| `dirty_in_sram=true` | SRAM 副本比 DRAM 新 | eviction 前必须 writeback |
| `pinned=true` | 不允许 eviction | 常用于权重/LUT/常量 |
| `power_state=off` | SRAM bank power-gated | valid_in_sram 必须清零 |

严格规则：

```text
若 command 读取 SRAM object，则必须 valid_in_sram = true
若 command 修改 SRAM object，则 dirty_in_sram = true
若 dirty object 被 evict，则必须先 DMA_STORE 到 DRAM
若 SRAM bank power-gated，则其中 object valid_in_sram = false
若 object 只存在 SRAM 且 dirty，但被覆盖/释放前未写回，则 simulator 报错
```

---

## 5. SRAM 易失性与上电初始化

### 5.1 Power state

SRAM bank/tile 至少有三种电源状态：

```text
ACTIVE       正常读写和 PIM，数据保持，付 active leakage
CLOCK_GATED 不能发新操作，数据保持，付 reduced leakage
POWER_GATED 数据不保持，valid bits 清零，付 near-zero leakage，唤醒有 latency/energy
```

### 5.2 上电行为

```text
on_power_on():
    for all SRAM banks:
        state = ACTIVE or CLOCK_GATED
        valid_in_sram = false
        dirty_in_sram = false
    DRAM remains valid
```

因此 cold-start trace 必须包含：

```text
DMA_LOAD WEIGHT tiles
DMA_LOAD LUT/constant tiles
DMA_LOAD input activation tiles
SRAM_PIM compute
DMA_STORE final output or dirty state
```

### 5.3 不同 SRAM 保存策略

| 策略 | 数据是否保留 | 是否计入 reload | 适用场景 |
|---|---:|---:|---|
| 持续供电 | 保留 | 否 | server steady-state |
| clock gating | 保留 | 否 | 短空闲 |
| power gating | 丢失 | 是 | 端侧低功耗休眠 |
| retention mode | 可选 | 取决于工艺模型 | 需要额外 SRAM macro 参数 |

---

## 6. 指令/Trace ISA

Trace 采用两级 ISA：

1. **High-level system ISA**：描述 DMA、allocation、PIM op、dependency。
2. **Lowered memory ISA**：把 DMA load/store 降成 DRAM read/write burst；把 SRAM-PIM op 降成 bank/subarray command。

### 6.1 High-level trace 格式

推荐文本格式：

```text
cycle_hint op object_id src dst bytes attrs dep_id
```

示例：

```text
0   SRAM_ALLOC W0_tile0       -          SRAM:T0:B0-7   65536  pinned=1             -
1   DMA_LOAD   W0_tile0       DRAM:0x1000 SRAM:T0:B0-7   65536  stream=weight         alloc_W0
2   SRAM_ALLOC X0_tile0       -          SRAM:T1:B0-3   16384  pinned=0             -
3   DMA_LOAD   X0_tile0       DRAM:0x9000 SRAM:T1:B0-3   16384  stream=activation     alloc_X0
4   BARRIER    ready_l0       W0_tile0,X0_tile0 -       0      type=all_done         load_W0,load_X0
5   PIM_MAC    Y0_psum0       W0_tile0,X0_tile0 SRAM:T2  0      mode=bank;acc=psum    ready_l0
6   PIM_REDUCE Y0_tile0       Y0_psum0   SRAM:T2        0      tree=local            mac0
7   DMA_STORE  Y0_tile0       SRAM:T2    DRAM:0xA000    16384  stream=output         reduce0
8   SRAM_FREE  X0_tile0       SRAM:T1    -              16384  -                     store_Y0
```

### 6.2 System ISA 定义

| 指令 | 作用 | 必须检查 | 产生统计 |
|---|---|---|---|
| `SRAM_ALLOC` | 分配 SRAM 空间 | capacity、bank availability | alloc bytes |
| `SRAM_FREE` | 释放 SRAM object | object 不再使用或已写回 | free bytes |
| `DMA_LOAD` | DRAM -> SRAM | DRAM valid，SRAM space，DMA/NoC bandwidth | DRAM read bytes, SRAM write |
| `DMA_STORE` | SRAM -> DRAM | SRAM valid，DMA/NoC bandwidth | SRAM read, DRAM write bytes |
| `DMA_PREFETCH` | 提前加载未来 tile | capacity，prefetch distance | prefetch hit/miss |
| `SRAM_RD` | 普通 SRAM 读 | valid、bank port | SRAM read |
| `SRAM_WR` | 普通 SRAM 写 | bank port | SRAM write, dirty |
| `PIM_MAC` | SRAM-PIM MAC | input/weight valid，PIM lane free | MAC count, energy |
| `PIM_EW_OP` | element-wise mul/add/sub/max | input valid | op count |
| `PIM_REDUCE` | partial-sum 或 max/min reduction | psum valid，reduction tree free | reduce count |
| `PIM_NL` | nonlinear/LUT/PWL/softmax | LUT/coeff valid | nonlinear count |
| `PIM_WRITEBACK` | PIM result 写 SRAM object | output space，bank port | SRAM write, dirty |
| `BARRIER` | 同步多个依赖 | dependency done | stall cycles |
| `POWER_SET` | bank/tile power state 切换 | dirty object safety | wakeup/sleep energy |

### 6.3 Lowered DRAM ISA

如果接 Ramulator/DRAMSim3，`DMA_LOAD` / `DMA_STORE` 需要降成 DRAM trace：

```text
READ  dram_addr
READ  dram_addr + burst_size
...
WRITE dram_addr
WRITE dram_addr + burst_size
...
```

若使用 Ramulator2，可由 Ramulator 内部继续建模：

```text
READ/WRITE request
  -> ACT/RD/WR/PRE/REF timing
  -> memory_system_cycles
```

如果暂时不接 Ramulator，可以使用 analytical DRAM model：

```text
T_dram_load = T_cmd_overhead + ceil(bytes / effective_bandwidth_per_cycle)
E_dram_load = bytes * E_dram_read_per_byte + bytes * E_io_per_byte
```

但论文级结果建议至少提供：

```text
Mode 1: analytical DRAM bandwidth model
Mode 2: Ramulator/DRAMSim3 validated timing model
```

### 6.4 Lowered SRAM-PIM ISA

`PIM_MAC` 可以进一步降成：

```text
SRAM_ACT_BANK      # 可选；SRAM 不像 DRAM 必需 ACT，但可表示 bank/subarray enable
SRAM_RD_ROW        # digital near-SRAM PIM 需要读出 operand
PIM_LATCH_ACT      # activation/input latch
PIM_MAC_SUBARRAY   # subarray-level MAC
PIM_MAC_BANK       # bank-level MAC
PIM_ACCUM          # local accumulation
PIM_REDUCE_TILE    # tile-level reduction
SRAM_WR_ROW        # write result / psum
```

是否需要 `SRAM_ACT_BANK` 取决于建模粒度。对于简单模型可省略；对于 bitline CIM，建议保留 `WL_ENABLE/BL_COMPUTE/SENSE/ADC` 类命令。

---

## 7. DRAM Backing Store 建模

### 7.1 为什么必须有 DRAM 指令

SRAM 是片上易失性存储，不是永久模型存储。实际系统中：

- 权重长期存放在 DRAM / Flash / host memory。
- 上电或模型切换时，需要把权重、LUT、常量加载到 SRAM。
- 每一轮 inference 的输入 activation 需要从上一级/DRAM/host 进入 SRAM。
- SRAM-PIM 的输出如果是最终输出，需要写回 DRAM/host-visible memory。
- 如果中间结果超过 SRAM 容量，需要 spill 到 DRAM。
- 如果 dirty object 被淘汰，必须 writeback 到 DRAM。

因此仿真中必须存在 DRAM load/store 事务。否则会低估 latency 和 energy。

### 7.2 DRAM load/store 触发条件

#### Load 触发

```text
1. object not valid in SRAM and current op needs it
2. cold-start preload of weights/LUT/constants
3. prefetch future tile
4. power-gated bank wakeup后 resident 数据失效，需要 reload
5. cache/scratchpad miss
```

#### Store/writeback 触发

```text
1. final output must be visible to host/next accelerator
2. dirty object eviction
3. psum/output/MDT/dist buffer exceeds SRAM capacity
4. checkpoint/recurrent state/KV cache must persist across calls
5. power-gating before data loss
```

### 7.3 Dirty writeback 协议

```python
def evict(obj):
    assert obj.valid_in_sram
    if obj.dirty_in_sram:
        emit(DMA_STORE, src=obj.sram_addr, dst=obj.dram_addr, bytes=obj.bytes)
        obj.valid_in_dram = True
        obj.dirty_in_sram = False
    obj.valid_in_sram = False
    free_sram_region(obj)
```

### 7.4 Capacity overflow / spill

若 SRAM 容量不足，有两种策略：

#### Strategy A: Tile recomputation / reload

```text
不写回中间结果；需要时重新从 DRAM 加载输入并重算。
```

适合：计算便宜、写回贵、中间结果可重算。

#### Strategy B: Spill to DRAM

```text
将中间 psum/dist/activation 写回 DRAM，后续再 DMA_LOAD。
```

适合：计算贵、重算代价高、结果必须跨 layer 保存。

仿真需要显式记录：

```text
spill_count
spill_bytes
spill_write_energy
spill_reload_energy
spill_stall_cycles
```

---

## 8. SRAM-PIM Timing Model

### 8.1 资源表

每个周期 scheduler 需要检查以下资源：

```yaml
resources:
  dram_channels:
    count: 1
    bandwidth_bytes_per_cycle: 64
  dma_engines:
    count: 2
    max_outstanding: 16
  noc:
    bandwidth_bytes_per_cycle: 128
    latency_cycles_per_hop: 1
  sram_tiles:
    count: 16
  sram_banks_per_tile: 32
  bank_ports:
    read_ports: 1
    write_ports: 1
    pim_ports: 1
  pim_lanes_per_bank: 128
  reduction_trees_per_tile: 1
  global_output_bus_bytes_per_cycle: 64
```

### 8.2 Bank conflict

同一个 bank 在同一周期能做的事情受端口约束。例如：

```text
single-port SRAM bank:
  RD / WR / PIM_MAC 三者互斥

1R1W SRAM bank:
  RD 与 WR 可并行，但 PIM_MAC 独占 bitline/wordline

dual-bank interleaving:
  不同 bank 可并行
```

需要配置：

```yaml
bank_access_policy:
  read_write_same_cycle: false
  pim_blocks_read: true
  pim_blocks_write: true
  pim_blocks_dma: true
```

### 8.3 PIM compute latency

对于 digital near-SRAM PIM：

```text
PIM_MAC latency = operand SRAM read + local MAC pipeline + psum write
```

对于 bitline SRAM-CIM：

```text
PIM_MAC latency = WL drive + bitline compute + sense/ADC + shift-add + output latch
```

通用公式：

```text
T_pim_mac_tile = T_input_latch + T_compute + T_accumulate + T_writeback
```

如果 MAC pipeline 可以接受新命令：

```text
issue_interval = max(1, nMAC_power, nMAC_resource)
latency = pipeline_depth
```

### 8.4 功耗约束

借鉴 ATTACC 中通过增加连续 MAC command 间隔来表示 power constraint 的思想，SRAM-PIM 可定义：

```text
nMAC_TILE = ceil(E_mac_tile / (P_tile_budget * Tclk))
nMAC_CHIP = ceil(E_mac_parallel / (P_chip_budget * Tclk))
```

调度规则：

```python
if current_cycle - last_mac_issue_cycle < nMAC_TILE:
    stall_reason = "power_throttle"
```

也可以使用硬限制：

```text
parallel_active_banks <= max_parallel_banks_under_power_budget
```

推荐同时统计：

```text
power_throttle_cycles
max_parallel_active_banks
average_active_banks
```

### 8.5 DMA 与 PIM overlap

真实系统中 DMA load 可以和 PIM compute 重叠，但受以下约束：

```text
1. DMA 写入的 SRAM bank 与 PIM 正在访问的 bank 是否冲突
2. NoC 是否被 DMA 和 PIM result movement 共享
3. global buffer 是否有双缓冲
4. 当前 PIM op 是否依赖尚未加载完成的数据
```

双缓冲示例：

```text
Tile k compute on buffer A
Tile k+1 DMA_LOAD into buffer B
After barrier, swap A/B
```

仿真中需要用 dependency DAG 表示：

```text
PIM_MAC(tile_k) depends on DMA_LOAD(W_k), DMA_LOAD(X_k)
DMA_LOAD(tile_k+1) can overlap with PIM_MAC(tile_k) if bank/resource non-conflict
```

---

## 9. Memory Manager 与调度策略

### 9.1 Allocation policy

推荐实现三种策略：

```text
STATIC_PINNED
  权重/LUT 常驻 SRAM，activation/psum streaming

DYNAMIC_LRU
  SRAM 类似 software-managed cache，按 LRU evict

LIFETIME_AWARE
  根据 layer graph 的 first_use/last_use 做 optimal-ish eviction
```

论文/研究建议使用 `LIFETIME_AWARE`，并把 `DYNAMIC_LRU` 作为保守 baseline。

### 9.2 Eviction policy

优先级：

```text
1. clean activation whose last_use passed
2. clean weight tile with low future reuse
3. dirty activation after DMA_STORE
4. dirty psum after spill
5. pinned object forbidden
```

### 9.3 Prefetch policy

```text
prefetch_distance = K tiles
prefetch only if free_sram_bytes >= threshold
prefetch cannot evict object needed before prefetch target
```

需要统计：

```text
prefetch_issued
prefetch_useful
prefetch_useless
prefetch_eviction_harm
```

---

## 10. 能耗模型

### 10.1 总能耗

```text
E_total =
    E_dram_read
  + E_dram_write
  + E_offchip_io
  + E_dma
  + E_sram_read
  + E_sram_write
  + E_sram_leakage
  + E_pim_compute
  + E_reduce
  + E_nonlinear
  + E_noc
  + E_control
```

### 10.2 DRAM energy

```text
E_dram_read  = DRAM_read_bytes  * E_dram_read_per_byte
E_dram_write = DRAM_write_bytes * E_dram_write_per_byte
E_io         = (DRAM_read_bytes + DRAM_write_bytes) * E_io_per_byte
```

如果接 DRAMPower/Ramulator/DRAMSim3，可使用 command-level energy。

### 10.3 SRAM energy from DESTINY

DESTINY 适合提供：

```text
SRAM read latency
SRAM write latency
SRAM read dynamic energy
SRAM write dynamic energy
SRAM leakage power
SRAM area
```

在仿真中转换为：

```text
E_sram_read  = N_sram_read  * E_read_per_access
E_sram_write = N_sram_write * E_write_per_access
E_leakage    = P_leakage_active * T_active
             + P_leakage_gated  * T_clock_gated
```

### 10.4 PIM compute energy

DESTINY 不会自动建模 PIM MAC/reduction/ADC/DAC。需要单独来源：

```text
Digital near-SRAM PIM:
  E_mac = RTL synthesis / standard-cell analytical model
  E_reduce = adder tree RTL / analytical model

Bitline SRAM-CIM:
  E_compute_bitline
  E_wordline_driver
  E_sense_amp / ADC
  E_DAC/input driver
  E_shift_add
  E_output_latch
```

### 10.5 Traffic-aware energy

对于 SRAM-PIM，很多能耗来自搬运而不是 MAC：

```text
E_move = bytes_tile_to_bank * E_local_bus
       + bytes_bank_to_tile * E_bank_bus
       + bytes_tile_to_global * E_noc
       + bytes_global_to_dram * E_offchip_io
```

必须单独统计：

```text
local_sram_bytes
bank_crossbar_bytes
tile_noc_bytes
dram_bytes
```

---

## 11. DESTINY 使用边界

### 11.1 可以用 DESTINY 做什么

```text
1. SRAM bank/buffer 的 area
2. SRAM read/write latency
3. SRAM read/write dynamic energy
4. SRAM leakage
5. 不同容量、bank 数、associativity、technology node 的 design-space exploration
```

### 11.2 不能只靠 DESTINY 做什么

```text
1. PIM instruction scheduling
2. bank/subarray conflict
3. DMA overlap
4. DRAM load/store timing
5. capacity spill/writeback
6. PIM MAC/reduction/nonlinear energy
7. analog SRAM-CIM ADC/DAC energy
8. end-to-end neural network layer mapping
```

### 11.3 推荐集成方式

```text
DESTINY output yaml
    -> destiny_adapter.py
    -> SRAM macro parameter table
    -> sram_pim_model.py uses latency/energy/area
```

示例：

```yaml
sram_macro:
  source: destiny
  tech_node_nm: 28
  capacity_kb: 256
  banks: 32
  read_latency_cycles: 1
  write_latency_cycles: 1
  read_energy_pj: 3.2
  write_energy_pj: 3.8
  leakage_mw: 12.5
  area_mm2: 0.42
```

---

## 12. 配置文件模板

### 12.1 `system.yaml`

```yaml
system:
  frequency_hz: 1000000000
  mode: warm_resident        # cold_start | warm_resident | amortized
  num_queries_for_amortization: 1000
  count_initial_preload: false
  count_final_writeback: true

workload:
  model: PointMamba_or_PointKAN
  batch_size: 1
  precision:
    activation: int8
    weight: int8
    psum: int32
```

### 12.2 `dram.yaml`

```yaml
dram:
  model: ramulator            # ramulator | dramsim3 | analytical
  type: LPDDR4                # DDR4 | LPDDR4 | HBM2 | HBM3
  channels: 1
  bus_width_bits: 64
  burst_bytes: 64
  effective_bandwidth_GBps: 25.6
  fixed_latency_ns: 80
  energy:
    read_pj_per_byte: 15.0
    write_pj_per_byte: 18.0
    io_pj_per_byte: 8.0
```

### 12.3 `sram_pim.yaml`

```yaml
sram_pim:
  tiles: 16
  banks_per_tile: 32
  bank_capacity_kb: 8
  total_capacity_kb: 4096
  word_bytes: 16

  port_model:
    read_ports_per_bank: 1
    write_ports_per_bank: 1
    pim_ports_per_bank: 1
    pim_exclusive_with_read: true
    pim_exclusive_with_write: true

  pim:
    mode: digital_near_sram       # digital_near_sram | bitline_cim
    lanes_per_bank: 128
    mac_latency_cycles: 2
    mac_issue_interval_cycles: 1
    reduce_latency_cycles: 4
    nonlinear_latency_cycles: 4
    max_parallel_banks: 16
    power_throttle: true
    tile_power_budget_mw: 100

  buffers:
    input_buffer_kb_per_tile: 16
    psum_buffer_kb_per_tile: 32
    output_buffer_kb_per_tile: 16
    double_buffering: true

  power_state:
    allow_clock_gating: true
    allow_power_gating: false
    wakeup_latency_cycles: 20
    wakeup_energy_pj: 500
```

### 12.4 `energy.yaml`

```yaml
energy:
  sram:
    source: destiny
    read_pj_per_access: 3.2
    write_pj_per_access: 3.8
    leakage_mw_per_bank: 0.39

  pim:
    mac_pj_per_op: 0.08
    reduce_pj_per_op: 0.04
    nonlinear_pj_per_elem: 0.12
    control_pj_per_command: 1.0

  noc:
    pj_per_byte_per_hop: 0.2
    average_hops: 2

  dma:
    pj_per_byte: 0.5
```

---

## 13. 核心仿真流程

### 13.1 主循环伪代码

```python
cycle = 0
while not all_commands_done():
    # 1. 完成当前周期结束的事件
    complete_events(cycle)

    # 2. 更新 object valid/dirty/resident 状态
    memory_manager.update_states(cycle)

    # 3. 找到 dependency ready 的 command
    ready_cmds = dependency_graph.get_ready_commands(cycle)

    # 4. 根据资源约束选择可发射 command
    issue_list = scheduler.select(
        ready_cmds,
        dram_model.resource_state,
        sram_pim_model.resource_state,
        noc_state,
        dma_state,
        power_state,
    )

    # 5. 发射 command，并注册完成事件
    for cmd in issue_list:
        latency = issue_command(cmd)
        event_queue.push(cycle + latency, cmd)

    # 6. 统计 stall 原因
    collect_stall_breakdown(ready_cmds, issue_list)

    # 7. leakage energy
    energy_model.add_leakage(cycle)

    cycle += 1
```

### 13.2 `issue_command` 行为

```python
def issue_command(cmd):
    if cmd.op == "DMA_LOAD":
        assert obj.valid_in_dram
        assert sram.has_space_or_can_evict(obj.bytes)
        dram_model.issue_read(cmd.dram_addr, cmd.bytes)
        dma_model.reserve(cmd.bytes)
        noc_model.reserve(cmd.bytes)
        sram_model.reserve_write_ports(cmd.dst_banks)
        energy.add_dram_read(cmd.bytes)
        energy.add_sram_write(cmd.bytes)
        return max(dram_latency, dma_latency, noc_latency, sram_write_latency)

    if cmd.op == "DMA_STORE":
        assert obj.valid_in_sram
        sram_model.reserve_read_ports(cmd.src_banks)
        noc_model.reserve(cmd.bytes)
        dram_model.issue_write(cmd.dram_addr, cmd.bytes)
        energy.add_sram_read(cmd.bytes)
        energy.add_dram_write(cmd.bytes)
        return max(sram_read_latency, noc_latency, dma_latency, dram_latency)

    if cmd.op == "PIM_MAC":
        assert all(inputs.valid_in_sram)
        assert banks_available(cmd.banks)
        assert pim_lanes_available(cmd.banks)
        assert power_budget_available(cmd)
        sram_model.reserve_pim_ports(cmd.banks)
        energy.add_pim_mac(cmd.mac_count)
        return pim_mac_latency
```

---

## 14. 算子到 Trace 的映射

### 14.1 GEMM / Linear

矩阵乘：

```text
Y[M, N] = X[M, K] * W[K, N]
```

Tiling：

```text
M tile = Tm
N tile = Tn
K tile = Tk
```

Trace：

```text
for m in M_tiles:
  DMA_LOAD X[m, k]
  for n in N_tiles:
    if W[k, n] not resident:
      DMA_LOAD W[k, n]
    PIM_MAC X[m,k], W[k,n] -> PSUM[m,n]
  PIM_REDUCE PSUM[m,n]
  if Y tile consumed by next layer in SRAM:
      keep resident
  else:
      DMA_STORE Y[m,n]
```

容量约束：

```text
bytes(X_tile) + bytes(W_resident_tiles) + bytes(PSUM_tile) + bytes(Y_tile) <= SRAM capacity
```

若不满足：

```text
reduce Tm/Tn/Tk
or spill PSUM
or evict low-reuse W tile
```

### 14.2 Point-cloud distance / KNN / FPS

距离：

```text
d(i,j) = ||p_i - p_j||^2
```

Trace：

```text
DMA_LOAD coordinate tile P_i
DMA_LOAD coordinate tile P_j
PIM_EW_OP SUB
PIM_EW_OP SQUARE
PIM_REDUCE SUM_DIM
PIM_REDUCE MIN/MAX/TOPK
DMA_STORE distance/topk metadata if cannot stay in SRAM
```

对 MDT/FPS：

```text
MDT[i] = min(MDT[i], d(i, selected_point))
```

若 MDT 超过 SRAM：

```text
stream MDT tile from DRAM
update MDT tile in SRAM-PIM
write dirty MDT tile back to DRAM
```

这里必须建模 MDT spill，否则大点云下会严重低估 DRAM traffic。

### 14.3 PointMamba / SSM scan

典型状态更新：

```text
h_t = A_t * h_{t-1} + B_t * u_t
y_t = C_t * h_t + D * u_t
```

Trace：

```text
DMA_LOAD u_t tile
DMA_LOAD A/B/C/D tile or keep resident
DMA_LOAD h_{t-1} if state not resident
PIM_EW_OP MUL A_t, h_{t-1}
PIM_EW_OP MUL B_t, u_t
PIM_EW_OP ADD -> h_t
PIM_MAC / EW_OP C_t, h_t -> y_t
PIM_EW_OP ADD D*u_t
DMA_STORE h_t if state must persist or SRAM capacity insufficient
```

由于 scan 有时序依赖：

```text
h_t depends on h_{t-1}
```

仿真必须保留 dependency chain，不能把所有 token 完全并行化。

### 14.4 PointKAN / GRF / PWL activation

如果 GRF/PWL 系数常驻 SRAM：

```text
cold-start:
  DMA_LOAD PWL/GRF coefficient table
warm-run:
  PIM_NL / LUT_READ / AFFINE
```

Trace：

```text
DMA_LOAD activation tile
PIM_NL index/segment selection
SRAM_RD coefficient/LUT
PIM_EW_OP affine/PWL
PIM_WRITEBACK output tile
```

若 LUT/coeff 不常驻：

```text
DMA_LOAD coeff tile before PIM_NL
```

---

## 15. 输出溢出与写回策略

### 15.1 输出是否必须写回 DRAM

不一定。取决于下一个消费者在哪里。

| 输出消费者 | 行为 |
|---|---|
| 下一个 SRAM-PIM layer | 保持在 SRAM，避免 DRAM write/read |
| GPU/CPU/host | DMA_STORE 到 DRAM 或 host-visible buffer |
| 另一个 accelerator | 通过 NoC/DMA 或 DRAM handoff |
| SRAM 容量不足 | spill/writeback 到 DRAM |
| power-gating 前 | dirty data 必须 writeback，否则丢失 |

### 15.2 Strict writeback rule

```text
Only write back when:
  1. output is final or external-visible
  2. dirty object must be evicted
  3. intermediate object lifetime crosses a memory boundary
  4. SRAM capacity requires spill
  5. power state transition would destroy data
```

这避免过度保守地“每层都写 DRAM”，也避免过度乐观地“永远不写 DRAM”。

---

## 16. 统计与报告格式

### 16.1 Latency breakdown

```yaml
latency_cycles:
  total: 1234567
  dram_load: 220000
  dram_store: 50000
  dma: 180000
  sram_read_write: 90000
  pim_compute: 500000
  reduction: 80000
  nonlinear: 30000
  stalls:
    dependency: 100000
    bank_conflict: 40000
    noc_contention: 20000
    power_throttle: 30000
    capacity_spill: 20000
```

### 16.2 Energy breakdown

```yaml
energy_pj:
  total: 987654321
  dram_read: 200000000
  dram_write: 80000000
  offchip_io: 120000000
  dma: 20000000
  sram_read: 90000000
  sram_write: 70000000
  sram_leakage: 60000000
  pim_mac: 180000000
  reduce: 50000000
  nonlinear: 30000000
  noc: 70000000
  control: 17654321
```

### 16.3 Correctness/debug report

```yaml
correctness:
  illegal_read_unresident_object: 0
  illegal_evict_dirty_without_writeback: 0
  capacity_overcommit_events: 0
  dependency_violations: 0
  power_gated_data_loss_events: 0

traffic:
  dram_read_bytes: 123456789
  dram_write_bytes: 9876543
  sram_read_accesses: 123456
  sram_write_accesses: 654321
  noc_bytes: 22222222

residency:
  weight_hit_rate: 0.92
  activation_hit_rate: 0.31
  psum_spill_count: 12
  dirty_writeback_count: 18
```

---

## 17. 最小可实现版本 MVP

如果从零实现，建议按以下顺序：

### Phase 1: Analytical baseline

```text
1. 实现 object table
2. 实现 SRAM capacity manager
3. 实现 DMA_LOAD/DMA_STORE analytical latency/energy
4. 实现 PIM_MAC analytical latency/energy
5. 实现 GEMM trace generator
```

先得到可跑通的：

```text
GEMM layer -> trace -> cycles/energy/traffic
```

### Phase 2: Resource conflict

```text
1. 加 bank port conflict
2. 加 PIM lane/tile pipeline
3. 加 NoC/DMA bandwidth
4. 加 dependency DAG
5. 加 double buffering overlap
```

### Phase 3: DRAM simulator 接入

```text
1. 将 DMA_LOAD/STORE 降成 DRAM read/write trace
2. 接 Ramulator2 或 DRAMSim3
3. 用 memory_system_cycles 替换 analytical DRAM latency
4. 保留 analytical 模式用于快速设计空间探索
```

### Phase 4: DESTINY/RTL energy 接入

```text
1. 用 DESTINY 生成 SRAM macro 参数
2. 用 RTL synthesis 或 analytical model 生成 PIM MAC/reduce 能耗
3. 合并 energy table
4. 输出完整 breakdown
```

### Phase 5: Point-cloud / Mamba / KAN 算子

```text
1. distance/KNN/FPS tracegen
2. MDT capacity/spill model
3. SSM scan dependency model
4. GRF/PWL/LUT activation model
```

---

## 18. 验证计划

### 18.1 Unit tests

```text
test_01_unresident_read:
  PIM_MAC before DMA_LOAD should fail

test_02_dirty_evict:
  dirty object eviction should auto emit DMA_STORE or fail

test_03_power_gate_loss:
  POWER_GATED bank should clear valid_in_sram

test_04_capacity_overflow:
  allocation beyond capacity should trigger eviction/spill

test_05_bank_conflict:
  two commands to same single-port bank cannot issue same cycle

test_06_dma_overlap:
  DMA to buffer B can overlap PIM on buffer A

test_07_dependency:
  PIM_MAC cannot issue before dependent DMA_LOAD completes
```

### 18.2 Analytical sanity checks

```text
1. DRAM bytes should equal sum of all cold-start loads + spill + final stores.
2. If SRAM capacity is infinite, spill_count should be zero.
3. If warm_resident=true and weights pinned, weight DRAM traffic should be zero except initial preload.
4. If bank count doubles and no other bottleneck, bank-conflict stalls should decrease.
5. If DRAM bandwidth halves, DMA stall should approximately increase.
```

### 18.3 Cross-validation

```text
DRAM timing:
  compare analytical model vs Ramulator/DRAMSim3 for same DMA trace

SRAM macro:
  compare DESTINY read/write latency/energy vs CACTI or foundry SRAM compiler when available

PIM compute:
  compare analytical MAC/reduce energy vs RTL synthesis

End-to-end:
  compare small GEMM latency against hand calculation
```

---

## 19. 论文写法建议

在论文或报告中不要写：

```text
We use DESTINY to simulate SRAM-PIM.
```

这不严谨，因为 DESTINY 不负责 PIM 调度和计算。

更严谨的写法是：

```text
We build a trace-driven SRAM-PIM simulator. The simulator models DRAM as the backing store and explicitly tracks SRAM residency, dirty writeback, capacity spills, DMA traffic, SRAM bank conflicts, PIM compute pipelines, reduction, and NoC contention. SRAM macro latency, dynamic energy, leakage, and area are obtained from DESTINY, while PIM compute and reduction energy are obtained from RTL synthesis or analytical circuit models. DRAM timing is modeled using Ramulator/DRAMSim3 or an analytical bandwidth model.
```

中文表述：

```text
我们构建了一个 trace-driven SRAM-PIM 仿真框架。该框架将 DRAM 建模为 SRAM 的 backing store，显式维护 SRAM 中数据的 valid/dirty/resident 状态，并对 cold-start 装载、warm-resident 复用、DMA load/store、容量溢出、dirty writeback、bank conflict、PIM 计算流水、reduction、NoC contention 与功耗 throttling 进行周期级或准周期级建模。SRAM macro 的读写延迟、动态能耗、泄漏与面积由 DESTINY 提供；PIM MAC、reduction、nonlinear 等计算单元由 RTL 综合或解析模型提供；DRAM 侧由 Ramulator/DRAMSim3 或带宽模型提供。
```

---

## 20. 最终推荐框架

推荐最终系统：

```text
                    +-------------------+
                    | PyTorch/ONNX Model|
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Layer/Tile Mapper |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Unified Trace IR  |
                    | DMA + SRAM + PIM  |
                    +---------+---------+
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
+-------------------+                  +----------------------+
| DRAM Timing Model |                  | SRAM-PIM Event Model |
| Ramulator/DRAMSim3|                  | bank/PIM/NoC/reduce  |
+---------+---------+                  +----------+-----------+
          |                                       |
          +-------------------+-------------------+
                              v
                    +-------------------+
                    | Energy Integrator |
                    | DESTINY + RTL     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Report/Breakdown  |
                    +-------------------+
```

最终原则：

```text
1. 数据不在 SRAM，就不能执行 SRAM-PIM。
2. SRAM 脏数据被覆盖、淘汰、掉电前必须写回。
3. SRAM 容量不足必须触发 tiling、eviction 或 spill。
4. DRAM load/store 可以与 PIM compute overlap，但必须受 DMA/NoC/bank 资源限制。
5. DESTINY 只提供 SRAM macro 参数，不能替代 PIM simulator。
6. cold-start、warm-run、amortized 三种模式必须分开报告。
```

---

## 21. 参考依据

- ATTACC simulator GitHub repository: Python simulator + modified Ramulator2 HBM3-PIM model, bank/BG/buffer PIM organizations, trace generation and power-constraint modeling.
- Ramulator2: modular, extensible, cycle-level DRAM memory-system simulator.
- DESTINY: microarchitecture-level tool for SRAM/eDRAM/ReRAM/STT-RAM/PCM 2D/3D cache/memory modeling, suitable for latency/area/energy/leakage estimation rather than complete PIM scheduling.

