# SRAM-PIM 仿真器问题清单与修改指南

> 面向仓库：`https://github.com/DingdongD/SRAM-PIM`  
> 目标：将当前可运行原型升级为更严格的 **DRAM backing store + SRAM resident state + SRAM-PIM command-level timing/energy simulator**。  
> 核心原则：先保证数据生命周期和容量/写回正确，再逐步细化 timing/energy。一个 SRAM-PIM simulator 的可信度首先取决于它能否严格回答：数据在哪里、是否 valid、是否 dirty、是否需要从 DRAM reload、是否需要 writeback、当前 cycle 是否允许发射该 PIM command。

---

## 0. 总体结论

当前仓库已经搭好了正确的骨架：

- 有 command-level `TraceCommand` / `OpCode`；
- 有 `DMA_LOAD` / `DMA_STORE`；
- 有 `MemoryObject` 的 `valid_in_sram` / `dirty_in_sram` / `valid_in_dram`；
- 有 `MemoryManager` 管理 SRAM 容量；
- 有 `ResourceModel` 处理 bank busy / PIM active bank；
- 有 analytical DRAM / energy model / DESTINY adapter；
- 有 GEMM、Attention、SSM trace generator。

但当前实现还不能算严格 architectural simulator。主要问题是：

1. `SRAM_ALLOC` 与 `valid_in_sram` 语义混淆；
2. 非 resident 输入仍可能继续执行 PIM；
3. DMA/PIM 的状态更新发生在 issue 时，而不是完成时；
4. SRAM 容量不足、dirty eviction、writeback 没有闭环；
5. dirty object 被 `SRAM_FREE` 时没有强制写回或报错；
6. GEMM partial-sum 累加依赖不严格；
7. PIM timing 过粗，未按 lanes / active banks / issue interval 计算；
8. SRAM / PIM / DMA / DRAM energy 统计未与真实数据流绑定；
9. DESTINY adapter 有接口，但尚未真正驱动配置参数；
10. power-gating / leakage / warm-resident amortization 没有闭环。

---

## 1. 严格 SRAM-PIM 仿真器必须满足的 10 条 invariant

后续所有修改都应围绕这些 invariant 设计。

### I1. Allocate 不等于 valid

`SRAM_ALLOC` 只能表示 SRAM 空间被预留，不代表数据已经可读。

```text
SRAM_ALLOC:     reserve SRAM space only
DMA_LOAD done:  valid_in_sram = true
PIM output done: valid_in_sram = true, dirty_in_sram = true
```

### I2. 数据不在 SRAM，PIM 不能执行

任何 `PIM_MAC`、`PIM_EW_OP`、`PIM_REDUCE`、`PIM_NL` 的输入必须满足：

```text
object exists
object.valid_in_sram == true
object.power_state == active
object is not loading / producing / spilling
```

否则 strict mode 必须报错，不能只计数后继续执行。

### I3. 状态更新必须发生在 command 完成时

`DMA_LOAD` 在 issue 时不能立刻把 object 设为 valid；`PIM_MAC` 在 issue 时不能立刻把 output 设为 valid。严格行为应该是：

```text
issue DMA_LOAD  -> object.state = LOADING
complete DMA_LOAD -> object.state = VALID_CLEAN

issue PIM_MAC -> output.state = PRODUCING
complete PIM_MAC -> output.state = VALID_DIRTY
```

### I4. SRAM 容量不足必须触发 spill/evict/writeback

如果 tile/bank 空间不足，不能只记录 `capacity_overcommit_events`。必须执行：

```text
find victim
if victim.dirty: DMA_STORE victim
free victim
retry allocate
```

### I5. Dirty SRAM 数据不能被直接丢弃

任何 dirty object 在以下事件前必须被 writeback，或者 strict mode 报错：

```text
SRAM_FREE
EVICT
POWER_GATE
simulation end, if final output or persistent state is dirty
```

### I6. DRAM 中的数据版本必须可区分 stale / valid

当 SRAM 中的 object 被修改后，DRAM copy 应该变成 stale：

```text
mark_dirty(): dirty_in_sram = true; valid_in_dram = false
writeback_complete(): dirty_in_sram = false; valid_in_dram = true
```

### I7. 同一个输出 tile 的多次累加必须串行依赖

例如 GEMM 的 K tile 累加：

```text
Y += X_k0 @ W_k0
Y += X_k1 @ W_k1
Y += X_k2 @ W_k2
```

这些写同一个 `Y` / `PSUM` 的 `PIM_MAC` 不能并行乱序，必须有 RAW/WAW dependency。

### I8. SRAM bank/resource 必须基于 object 实际位置

PIM command 的资源占用不能只看 `cmd.dst`，还应看输入 object 所在的 banks。DMA_STORE 也应从 object 的真实 SRAM location 读出，而不是依赖 trace 里手写的 `SRAM:T0:B0-1`。

### I9. Energy counter 必须与数据流绑定

例如 `PIM_MAC` 不仅有 MAC energy，还应统计：

```text
activation SRAM read
weight SRAM read
psum read/write
local reduction
output SRAM write
```

### I10. 报告中 correctness counters 必须为 0

任何用于论文/架构结论的 run，必须满足：

```text
illegal_read_unresident_object == 0
capacity_overcommit_events == 0
illegal_evict_dirty_without_writeback == 0
dependency_violations == 0
deadlock_events == 0
```

---

## 2. 优先级总表

| ID | 严重级别 | 模块 | 问题摘要 | 影响 |
|---|---:|---|---|---|
| P0-01 | 致命 | `MemoryManager.allocate` | allocate 直接让 object valid | 可能跳过 DRAM load，PIM 读到不存在的数据 |
| P0-02 | 致命 | `Simulator._check_inputs_valid` | 非 resident 输入只计数，不阻止 PIM | correctness counters 非 0 但结果仍被使用 |
| P0-03 | 致命 | event model | DMA/PIM side effect 在 issue 时生效 | 数据可在完成前被错误消费 |
| P0-04 | 致命 | capacity | allocation failure 只计数，不 spill | 容量约束没有真正生效 |
| P0-05 | 致命 | dirty lifecycle | `SRAM_FREE` 可丢弃 dirty object | 输出/psum 可能丢失且不计 DRAM writeback |
| P0-06 | 致命 | output object | 新 PIM output 默认 `valid_in_dram=True` | DRAM 版本语义错误 |
| P0-07 | 致命 | GEMM trace | 多个 K tile 写同一 Y 没有累加依赖 | GEMM 结果和 latency 都不严格 |
| P0-08 | 致命 | resource | 资源 banks 只看 dst/trace 字符串 | 忽略输入 bank conflict 和真实 object 位置 |
| P0-09 | 致命 | simulator loop | deadlock 时强制 complete remaining | 掩盖 trace dependency 错误 |
| P1-01 | 高 | timing | PIM MAC latency 未除以 lanes/banks | 延迟可能严重失真 |
| P1-02 | 高 | timing | 多处 floor division | 小 tensor / 非整除 tile 延迟偏小 |
| P1-03 | 高 | resource | DMA_LOAD/STORE 读写方向分类错误 | bank port conflict 统计错误 |
| P1-04 | 高 | PIM ops | `PIM_NL` 未纳入 PIM resource | nonlinear 与 PIM/port 冲突未建模 |
| P1-05 | 高 | energy | PIM MAC 未绑定 SRAM read/write | 能耗拆分不可信 |
| P1-06 | 高 | energy | EW_OP 复用 MAC energy counter | element-wise 能耗语义混乱 |
| P1-07 | 高 | power | `POWER_SET` no-op，leakage 全 bank 常开 | power gating/clock gating 无效 |
| P1-08 | 高 | DESTINY | adapter 未覆盖 SimConfig | SRAM macro 参数不是 DESTINY-backed |
| P1-09 | 高 | DRAM | analytical DRAM 无队列/bank/refresh | DRAM traffic 多时不够严格 |
| P1-10 | 高 | memory | 只统计 tile capacity，无 bank capacity | bank-level overflow 不可见 |
| P2-01 | 中 | trace IR | `src/dst` 字符串过载 | 难做验证、重定位、reload |
| P2-02 | 中 | validation | 缺 trace validator | 难发现 WAW/RAW/未加载输入 |
| P2-03 | 中 | tests | 缺 negative tests | 错误状态不会被 CI 捕获 |
| P2-04 | 中 | report | 缺 utilization/stall breakdown | 不能定位性能瓶颈 |
| P2-05 | 中 | mode | cold/warm/amortized 未统一 | initial preload 与 final writeback 统计不一致 |

---

# 3. P0 级问题与修改指南

---

## P0-01：`SRAM_ALLOC` 让 object 变成 valid

### 当前现象

当前 `MemoryManager.allocate()` 会调用：

```python
obj.load_to_sram(sram_tile, sram_banks)
```

而 `MemoryObject.load_to_sram()` 会设置：

```python
self.valid_in_sram = True
```

这导致 `SRAM_ALLOC` 之后 object 就可以被 PIM 读取。严格来说，这是错误的。

### 为什么错误

SRAM 是易失性存储。`SRAM_ALLOC` 只表示给 object 分配 SRAM 空间，不代表 DRAM 中的数据已经搬入 SRAM。正确流程应该是：

```text
SRAM_ALLOC -> reserved, invalid
DMA_LOAD issue -> loading
DMA_LOAD complete -> valid clean
```

### 修改方案

在 `MemoryObject` 中拆分位置绑定和数据有效性。

```python
class ObjState(Enum):
    DRAM_ONLY = "DRAM_ONLY"
    SRAM_RESERVED = "SRAM_RESERVED"
    LOADING = "LOADING"
    VALID_CLEAN = "VALID_CLEAN"
    VALID_DIRTY = "VALID_DIRTY"
    PRODUCING = "PRODUCING"
    SPILLING = "SPILLING"
    EVICTED = "EVICTED"
    POWER_GATED = "POWER_GATED"
```

推荐将原来的 `load_to_sram()` 拆成：

```python
def place_in_sram(self, sram_tile: int, sram_banks: list) -> None:
    self.sram_tile = sram_tile
    self.sram_banks = list(sram_banks)
    self.power_state = "active"
    # 注意：这里不设置 valid_in_sram


def commit_dma_load(self) -> None:
    self.valid_in_sram = True
    self.dirty_in_sram = False
    self.power_state = "active"
```

`MemoryManager.allocate()` 改成：

```python
def allocate(self, object_id: str, sram_tile: int, sram_banks: list, make_valid: bool = False) -> bool:
    obj = self.objects[object_id]
    if self.tile_usage[sram_tile] + obj.bytes > self.tile_capacity[sram_tile]:
        return False
    obj.place_in_sram(sram_tile, sram_banks)
    if make_valid:
        obj.valid_in_sram = True
    self.tile_usage[sram_tile] += obj.bytes
    self.stats["alloc_bytes"] += obj.bytes
    return True
```

`_issue_alloc()` 中：

```python
ok = self.mem_mgr.allocate(cmd.object_id, tile, banks, make_valid=preloaded)
```

### 验收测试

构造 trace：

```text
0 SRAM_ALLOC X - SRAM:T0:B0-1 1024 {type:ACTIVATION} -
1 PIM_MAC Y X SRAM:T0:B2-3 1024 {mac_count:128} 0
```

strict mode 应该报错：

```text
Object X is allocated but not valid in SRAM
```

---

## P0-02：非 resident 输入不能继续 PIM

### 当前现象

`_check_inputs_valid()` 返回 `False` 后，`_issue_pim_mac()`、`_issue_pim_ew()`、`_issue_pim_reduce()`、`_issue_pim_nl()` 仍然继续执行。

### 为什么错误

这违反核心 invariant：数据不在 SRAM，不能执行 SRAM-PIM。否则模拟器会输出一个 latency/energy 报告，但这个报告其实对应非法执行。

### 修改方案

增加 correctness mode。

```python
class CorrectnessMode(Enum):
    STRICT = "strict"
    WARN = "warn"
    AUTO_RELOAD = "auto_reload"
```

在 `SystemConfig` 中加入：

```python
correctness_mode: str = "strict"
```

将 `_check_inputs_valid()` 改为：

```python
def _require_inputs_valid(self, cmd: TraceCommand) -> None:
    input_ids = [s.strip() for s in cmd.src.split(",") if s.strip() and s.strip() != "-"]
    for iid in input_ids:
        obj = self.mem_mgr.objects.get(iid)
        if obj is None:
            self.correctness["missing_input_object"] += 1
            raise RuntimeError(f"Missing input object {iid} for {cmd.op.value} cmd={cmd.cmd_id}")
        if not obj.valid_in_sram:
            self.correctness["illegal_read_unresident_object"] += 1
            raise RuntimeError(f"Input {iid} is not valid in SRAM for {cmd.op.value} cmd={cmd.cmd_id}")
        if obj.power_state != "active":
            self.correctness["illegal_read_power_gated_object"] += 1
            raise RuntimeError(f"Input {iid} is power-gated for {cmd.op.value} cmd={cmd.cmd_id}")
```

然后所有 PIM issue 函数改为：

```python
def _issue_pim_mac(self, cmd):
    self._require_inputs_valid(cmd)
    ...
```

### 验收测试

任何 correctness counter 非 0 的 run，默认不应生成“有效结果”。report 中加入：

```python
"valid_simulation": all(v == 0 for v in self.correctness.values())
```

---

## P0-03：DMA/PIM 状态更新必须在完成时生效

### 当前现象

当前 `_issue_dma_load()` 在 issue 时立即执行：

```python
obj.valid_in_sram = True
```

`_issue_dma_store()` 在 issue 时立即执行：

```python
obj.writeback_complete()
```

`_issue_pim_mac()` 在 issue 时立即把输出设为 valid/dirty。

### 为什么错误

在 cycle-level simulator 中，一个 command issue 之后需要若干 cycles 才完成。状态如果在 issue 时就更新，后续没有显式 dependency 的 command 可能提前读到未完成数据。

### 修改方案

事件队列中不要只存 `(finish_cycle, cmd_id)`，而是存 completion callback 或 command 本身。

```python
@dataclass(order=True)
class Event:
    finish_cycle: int
    cmd_id: int
    cmd: TraceCommand = field(compare=False)
```

issue 阶段只设置 pending 状态：

```python
def _issue_dma_load(self, cmd):
    obj = self.mem_mgr.objects[cmd.object_id]
    obj.state = ObjState.LOADING
    lat = self.dram.get_read_latency(cmd.bytes)
    self.energy.add_dram_read(cmd.bytes)
    self.energy.add_noc(cmd.bytes)
    return lat
```

completion 阶段提交状态：

```python
def _complete_command(self, cmd):
    if cmd.op == OpCode.DMA_LOAD:
        obj = self.mem_mgr.objects[cmd.object_id]
        obj.commit_dma_load()
        n_accesses = ceil_div(cmd.bytes, self.config.sram_pim.word_bytes)
        self.energy.add_sram_write(n_accesses)

    elif cmd.op == OpCode.DMA_STORE:
        obj = self.mem_mgr.objects[cmd.object_id]
        obj.writeback_complete()

    elif cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE, OpCode.PIM_NL}:
        obj = self.mem_mgr.objects[cmd.object_id]
        obj.valid_in_sram = True
        obj.mark_dirty()
```

主循环中：

```python
while self.event_queue and self.event_queue[0].finish_cycle <= self.cycle:
    event = heapq.heappop(self.event_queue)
    self._complete_command(event.cmd)
    self.completed.add(event.cmd_id)
```

### 验收测试

构造两个没有 dependency 的命令：

```text
DMA_LOAD X latency=100
PIM_MAC Y src=X deps=[]
```

strict mode 应报错或要求 trace validator 发现 dependency 缺失。

---

## P0-04：容量不足必须 spill/writeback/evict

### 当前现象

`_issue_alloc()` 如果 `allocate()` 返回 False，只增加：

```python
capacity_overcommit_events += 1
```

然后继续执行。

### 为什么错误

这等价于 SRAM 容量无限大，只是记录了一个 warning。大模型场景下，这会低估 DRAM traffic、NoC traffic、latency 和 energy。

### 修改方案

实现 `allocate_or_spill()`。

```python
def allocate_or_spill(self, object_id: str, tile: int, banks: list) -> int:
    """Return extra latency caused by spilling/writeback."""
    if self.mem_mgr.allocate(object_id, tile, banks):
        return 0

    obj = self.mem_mgr.objects[object_id]
    needed = obj.bytes - self.mem_mgr.get_free_bytes(tile)
    victims = self.mem_mgr.find_eviction_candidate(tile, needed)

    if not victims:
        raise RuntimeError(f"No eviction candidate for allocating {object_id}, need {needed} bytes")

    extra_lat = 0
    for vid in victims:
        victim = self.mem_mgr.objects[vid]
        if victim.dirty_in_sram:
            extra_lat += self._blocking_writeback(victim)
            self.mem_mgr.stats["writeback_count"] += 1
        needs_wb = victim.evict()
        assert not needs_wb, "Dirty victim must be written back before evict"
        self.mem_mgr.free_capacity_only(vid)
        self.mem_mgr.stats["eviction_count"] += 1

    if not self.mem_mgr.allocate(object_id, tile, banks):
        raise RuntimeError(f"Allocation still fails after eviction: {object_id}")

    return extra_lat
```

需要补充 `free_capacity_only()`，避免 `evict()` 后 `valid_in_sram=False` 导致 `free()` 不减少 usage。

### 注意

如果采用 non-blocking eviction，需要将 writeback 插入 event queue，并让当前 allocation 等待 writeback 完成。第一阶段建议先做 blocking，保证正确性。

### 验收测试

设置 tile capacity = 1 KB，连续 allocate 三个 800B object，其中前两个 dirty。期望：

```text
capacity_overcommit_events == 0
spill_count > 0
writeback_count > 0
dram_write_bytes 增加
```

---

## P0-05：`SRAM_FREE` 不能丢弃 dirty object

### 当前现象

`MemoryManager.free()` 直接将 object invalid：

```python
obj.valid_in_sram = False
obj.sram_tile = -1
obj.sram_banks = []
```

没有检查 `dirty_in_sram`。

### 为什么错误

dirty object 表示 SRAM 中的数据比 DRAM 新。直接 free 会丢失结果。

### 修改方案

给 `SRAM_FREE` 增加策略：

```python
free_policy: str = "strict"  # strict | auto_writeback | discard_clean_only
```

严格模式：

```python
def _issue_free(self, cmd):
    obj = self.mem_mgr.objects[cmd.object_id]
    if obj.dirty_in_sram:
        self.correctness["illegal_evict_dirty_without_writeback"] += 1
        raise RuntimeError(f"Cannot free dirty object {obj.object_id} without DMA_STORE")
    self.mem_mgr.free(cmd.object_id)
    return 0
```

自动写回模式：

```python
if obj.dirty_in_sram:
    lat = self._blocking_writeback(obj)
    self.mem_mgr.free(cmd.object_id)
    return lat
```

### 验收测试

```text
PIM_MAC Y ...
SRAM_FREE Y
```

strict mode 必须报错。

---

## P0-06：PIM output 的 DRAM 版本不能默认 valid

### 当前现象

`MemoryObject` 默认：

```python
valid_in_dram = True
```

新建 PIM output 时没有覆盖这一值。

### 为什么错误

`Y` 是 PIM 计算产生的新 object，在 `DMA_STORE` 前 DRAM 中并没有有效版本。默认 `valid_in_dram=True` 会让 reload/spill 语义错误。

### 修改方案

创建新 output/psum 时显式设置：

```python
out_obj = MemoryObject(
    object_id=cmd.object_id,
    obj_type=ObjType.PSUM,
    bytes=out_bytes,
    precision="int32",
    valid_in_dram=False,
    valid_in_sram=False,
)
```

并修改 `mark_dirty()`：

```python
def mark_dirty(self):
    self.dirty_in_sram = True
    self.valid_in_dram = False
```

`writeback_complete()`：

```python
def writeback_complete(self):
    self.valid_in_dram = True
    self.dirty_in_sram = False
```

### 验收测试

PIM 产生 `Y` 后，不执行 `DMA_STORE`，然后强制 evict。期望：

```text
must writeback before evict
DRAM version initially invalid/stale
```

---

## P0-07：GEMM partial-sum 累加依赖不严格

### 当前现象

GEMM trace 中每个 `K tile` 都对同一个 `Y_m{mi}_n{ni}` 发 `PIM_MAC`，但这些 `PIM_MAC` 之间没有依赖关系。`DMA_STORE` 只依赖最后一个 `PIM_MAC`。

### 为什么错误

GEMM 的 tile 累加是：

```text
Y_mn = sum_k X_mk @ W_nk
```

同一个 `Y_mn` 的不同 K tile 需要累加到同一个 psum。不能让它们并行写同一 object。

### 修改方案

在 `gen_gemm_trace.py` 中维护 `last_y_writer`。

```python
last_y_writer = {}

for mi in range(m_tiles):
    for ki in range(k_tiles):
        ...
        for ni in range(n_tiles):
            y_id = f"Y_m{mi}_n{ni}"
            deps = [load_x_id, weight_load_ids[(ni, ki)]]
            if y_id in last_y_writer:
                deps.append(last_y_writer[y_id])
                accumulate = True
            else:
                accumulate = False

            cmds.append(TraceCommand(
                cmd_id, OpCode.PIM_MAC, y_id,
                f"{x_id},W_n{ni}_k{ki}" ,
                f"SRAM:T{y_tile}:B0-1",
                y_bytes_psum,
                {
                    "mac_count": mac_count,
                    "accumulate": accumulate,
                    "out_bytes": actual_m * actual_n * psum_dbyte,
                    "dtype_out": "int32",
                },
                deps
            ))
            last_y_writer[y_id] = cmd_id
            cmd_id += 1
```

`DMA_STORE` 应依赖：

```python
deps=[last_y_writer[y_id]]
```

并且 `Y` 的 SRAM object 应按 psum precision 分配：

```python
y_bytes_psum = actual_m * actual_n * precision_bytes("int32")
```

最终写回如果是 int8 output，应额外建模 requant/cast：

```text
PIM_REDUCE / REQUANT Y_psum -> Y_out_int8
DMA_STORE Y_out_int8
```

### 验收测试

对 `K=2*Tk` 的 GEMM，检查同一个 `Y` 至少有两条 `PIM_MAC`，第二条必须依赖第一条。

---

## P0-08：资源 banks 必须来自 object 实际位置

### 当前现象

`_get_banks()` 优先从 `cmd.dst` 解析 SRAM 位置。如果 PIM command 的 `src` 是 object id，例如 `X,W`，它不会查 `X` 和 `W` 的实际 SRAM banks。

### 为什么错误

PIM 的输入读取会占用 activation/weight 所在 SRAM banks。只看 dst 会低估 bank conflict，也会让 trace 中错误的 `src` location 不被发现。

### 修改方案

实现：

```python
def _banks_of_object(self, oid: str) -> list[int]:
    obj = self.mem_mgr.objects[oid]
    return [obj.sram_tile * self.config.sram_pim.banks_per_tile + b for b in obj.sram_banks]


def _get_banks(self, cmd: TraceCommand) -> list[int]:
    banks = set()

    if cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE, OpCode.PIM_NL}:
        for iid in parse_src_ids(cmd.src):
            if iid in self.mem_mgr.objects:
                banks.update(self._banks_of_object(iid))
        if cmd.object_id in self.mem_mgr.objects:
            banks.update(self._banks_of_object(cmd.object_id))
        elif cmd.dst.startswith("SRAM:"):
            banks.update(parse_loc_to_global_banks(cmd.dst))
        return sorted(banks)

    if cmd.op == OpCode.DMA_LOAD:
        return parse_loc_to_global_banks(cmd.dst)

    if cmd.op == OpCode.DMA_STORE:
        if cmd.object_id in self.mem_mgr.objects:
            return self._banks_of_object(cmd.object_id)
        return parse_loc_to_global_banks(cmd.src)

    ...
```

### 验收测试

构造两个 PIM command，它们不同 dst，但共享同一个 input bank。应出现 bank conflict 或被端口模型限制。

---

## P0-09：deadlock 不能强制 complete

### 当前现象

主循环中如果 pending command 的 deps 永远无法满足，会强制：

```python
for cid in pending:
    self.completed.add(cid)
pending.clear()
```

### 为什么错误

这会掩盖 trace generator 的 dependency bug。严格 simulator 不能把错误 trace 修成“成功完成”。

### 修改方案

改为报错：

```python
if pending and not self.event_queue and not any(self._deps_ready(cmd) for cmd in pending.values()):
    self.correctness["dependency_violations"] += 1
    unresolved = {cid: cmd.deps for cid, cmd in pending.items() if not self._deps_ready(cmd)}
    raise RuntimeError(f"Deadlock/unresolved dependencies: {unresolved}")
```

### 验收测试

构造一条依赖不存在 command id 的 trace。strict mode 必须报错。

---

# 4. P1 级问题与修改指南

---

## P1-01：PIM MAC latency 应按 effective lanes 计算

### 当前现象

当前 `PIM_MAC` latency 类似：

```python
lat = max(mac_latency_cycles, mac_count * mac_issue_interval_cycles)
```

这没有考虑 `lanes_per_bank` 和 active bank 数。

### 推荐公式

对于 digital near-SRAM PIM：

```python
active_banks = max(1, len(output_banks_or_compute_banks))
effective_lanes = active_banks * lanes_per_bank
compute_steps = ceil_div(mac_count, effective_lanes)
lat = setup_cycles + compute_steps * mac_issue_interval_cycles + output_write_cycles
lat = max(lat, mac_latency_cycles)
```

更严格：

```python
input_read_cycles = max(
    ceil_div(input_a_bytes, word_bytes * read_ports * active_banks),
    ceil_div(input_b_bytes, word_bytes * read_ports * active_banks),
)
compute_cycles = ceil_div(mac_count, active_banks * lanes_per_bank) * mac_issue_interval_cycles
psum_cycles = ceil_div(out_bytes, word_bytes * write_ports * active_banks)
lat = input_read_cycles + compute_cycles + psum_cycles
```

对于 SRAM-CIM bit-serial：

```text
lat = input_bit_cycles * weight_bit_cycles * adc_cycles + shift_add + reduce + writeback
```

### 验收测试

固定 `mac_count=1024`：

- banks=1, lanes=128 -> compute steps = 8；
- banks=4, lanes=128 -> compute steps = 2；
- latency 应随 active banks 增加而下降，直到受带宽/issue limit 限制。

---

## P1-02：所有除法都要用 ceil

### 当前问题位置

当前多处使用：

```python
cmd.bytes // word_bytes
count // lanes_per_bank
nbytes // burst_bytes
```

这会在非整除时低估访问次数或 cycles。

### 修改方案

统一定义：

```python
def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return (a + b - 1) // b
```

替换：

```python
n_accesses = max(1, ceil_div(cmd.bytes, self.config.sram_pim.word_bytes))
lat = max(1, ceil_div(count, effective_lanes))
n_bursts = max(1, ceil_div(nbytes, burst_bytes))
```

### 验收测试

`bytes=17, word_bytes=16` 应得到 2 accesses，不是 1。

---

## P1-03：DMA_LOAD / DMA_STORE 的 SRAM 端口方向应修正

### 当前现象

`ResourceModel` 中：

```python
_READ_OPS = {SRAM_RD, DMA_LOAD, DMA_PREFETCH}
_WRITE_OPS = {SRAM_WR, DMA_STORE}
```

但从 SRAM 角度：

```text
DMA_LOAD:  DRAM -> SRAM，是 SRAM write
DMA_STORE: SRAM -> DRAM，是 SRAM read
```

### 修改方案

```python
_SRAM_READ_OPS = {OpCode.SRAM_RD, OpCode.DMA_STORE}
_SRAM_WRITE_OPS = {OpCode.SRAM_WR, OpCode.DMA_LOAD, OpCode.DMA_PREFETCH}
_PIM_OPS = {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE, OpCode.PIM_NL}
```

### 验收测试

同一 bank 上 `DMA_LOAD` 与 `PIM_MAC` 若 `pim_exclusive_with_write=True`，应冲突。

---

## P1-04：`PIM_NL` 应纳入 PIM resource

### 当前现象

`PIM_NL` 不在 `_PIM_OPS` 中，可能不占用 bank/PIM resource。

### 修改方案

```python
_PIM_OPS = {
    OpCode.PIM_MAC,
    OpCode.PIM_EW_OP,
    OpCode.PIM_REDUCE,
    OpCode.PIM_NL,
}
```

并为 nonlinear unit 增加可配置资源：

```python
@dataclass
class PIMHardwareConfig:
    nonlinear_units_per_tile: int = 1
    nonlinear_issue_interval_cycles: int = 1
```

### 验收测试

两个 softmax/nonlinear command 若映射到同 tile 且只有 1 个 nonlinear unit，应串行。

---

## P1-05：PIM energy 要绑定 SRAM array read/write

### 当前现象

`PIM_MAC` 只统计：

```python
self.energy.add_pim_mac(mac_count)
```

这低估/混淆了 SRAM-PIM 的核心能耗来源。

### 修改方案

在 `_issue_pim_mac()` 或 completion 阶段增加数据流能耗：

```python
def _account_pim_mac_energy(self, cmd):
    input_ids = parse_src_ids(cmd.src)
    for iid in input_ids:
        obj = self.mem_mgr.objects[iid]
        self.energy.add_sram_read(ceil_div(obj.bytes, self.config.sram_pim.word_bytes))

    out_bytes = get_out_bytes(cmd)
    if cmd.attrs.get("accumulate", False):
        # read old psum
        self.energy.add_sram_read(ceil_div(out_bytes, self.config.sram_pim.word_bytes))

    # write new psum/output
    self.energy.add_sram_write(ceil_div(out_bytes, self.config.sram_pim.word_bytes))
    self.energy.add_pim_mac(cmd.attrs["mac_count"])
```

对 near-SRAM digital PIM，推荐：

```text
E_total_PIM_MAC = E_sram_read_act + E_sram_read_weight + E_mac_array + E_psum_read/write + E_reduce
```

对 SRAM-CIM，推荐将 SRAM read/write 进一步拆成：

```text
E_wordline + E_bitline + E_input_driver + E_ADC/SenseAmp + E_shift_add + E_accumulator
```

### 验收测试

同样 `mac_count`，如果输入/输出 bytes 增大，SRAM read/write energy 应增加，而不是只有 MAC energy 不变。

---

## P1-06：EW_OP 不应复用 MAC counter

### 当前现象

`_issue_pim_ew()` 调用：

```python
self.energy.add_pim_mac(count)
```

这会把 element-wise op 统计为 MAC。

### 修改方案

增加：

```python
"pim_ew_count": 0
"pim_ew_pj": ...
```

配置：

```python
@dataclass
class PIMEnergyConfig:
    mac_pj_per_op: float = 0.08
    ew_pj_per_op: float = 0.03
    reduce_pj_per_op: float = 0.04
    nonlinear_pj_per_elem: float = 0.12
```

使用：

```python
self.energy.add_pim_ew(count)
```

### 验收测试

SSM trace 运行后 report 应区分 `pim_ew_count` 和 `pim_mac_count`。

---

## P1-07：`POWER_SET` 不能是 no-op

### 当前现象

`_issue_command()` 中：

```python
elif cmd.op == OpCode.POWER_SET:
    return 0
```

### 修改方案

支持至少三种 power state：

```text
active
clock_gated
power_gated
```

`POWER_SET` 的语义：

```python
def _issue_power_set(self, cmd):
    target = cmd.attrs.get("state")
    object_ids = cmd.attrs.get("objects", [])
    bank_ids = cmd.attrs.get("banks", [])

    if target == "power_gated":
        for obj in affected_objects:
            if obj.dirty_in_sram:
                if self.config.system.correctness_mode == "strict":
                    raise RuntimeError("Cannot power-gate dirty object without writeback")
                else:
                    self._blocking_writeback(obj)
            obj.power_gate()
        self.energy.add_wakeup_or_power_energy(...)
        return self.config.sram_pim.power_state.wakeup_latency_cycles
```

Leakage 计算应区分 active/gated bank：

```python
active_banks = self.mem_mgr.count_active_banks()
clock_gated_banks = ...
power_gated_banks = ...
self.energy.add_leakage_by_state(active_banks, clock_gated_banks, power_gated_banks, cycles=1)
```

### 验收测试

将一半 banks power-gated 后，leakage 应下降；dirty object power-gate 必须先 writeback 或报错。

---

## P1-08：DESTINY adapter 要真正驱动配置

### 当前现象

有 `DestinyAdapter`，但主 `SimConfig` 并没有在启动时自动用 DESTINY 结果覆盖 SRAM latency/energy/leakage/area。

### 修改方案

在 config loader 后增加：

```python
def apply_destiny_params(config: SimConfig) -> None:
    if config.energy.sram.source != "destiny":
        return
    adapter = DestinyAdapter(config.energy.sram.destiny_dir)
    params = adapter.run(
        tech_nm=config.energy.sram.tech_nm,
        capacity_kb=config.sram_pim.bank_capacity_kb * config.sram_pim.banks_per_tile,
        banks=config.sram_pim.banks_per_tile,
        word_bits=config.sram_pim.word_bytes * 8,
    )
    config.energy.sram.read_pj_per_access = params.read_energy_pj
    config.energy.sram.write_pj_per_access = params.write_energy_pj
    config.energy.sram.leakage_mw_per_bank = params.leakage_mw / config.sram_pim.banks_per_tile
    config.sram_pim.sram_read_latency_cycles = ns_to_cycles(params.read_latency_ns, config.system.frequency_hz)
    config.sram_pim.sram_write_latency_cycles = ns_to_cycles(params.write_latency_ns, config.system.frequency_hz)
```

Report 中必须记录：

```json
"sram_params_source": "DESTINY" or "analytical_fallback"
```

### 验收测试

设置 `energy.sram.source=destiny` 后，report 中的 SRAM read/write energy 应来自 adapter，而不是默认 3.2/3.8 pJ。

---

## P1-09：DRAM model 建议分层

当前 analytical DRAM 可保留作为 fast mode，但严格模式需要至少提供：

```text
fast mode: fixed latency + bandwidth
trace mode: generate DRAMSim3/Ramulator trace
strict mode: DRAMSim3/Ramulator command-level timing
```

### 修改方案

`DRAMConfig.model`：

```yaml
dram:
  model: analytical   # analytical | dramsim3 | ramulator
```

`DRAMModel` 抽象：

```python
class BaseDRAMModel:
    def issue_read(self, addr: int, nbytes: int, cycle: int) -> int: ...
    def issue_write(self, addr: int, nbytes: int, cycle: int) -> int: ...
```

Analytical version 继续使用：

```python
fixed_latency + ceil(nbytes / bytes_per_cycle)
```

DRAMSim3/Ramulator version 则输出 trace 并解析完成时间。

### 验收测试

大量连续 DMA 与随机 DMA 在 strict DRAM mode 下应出现不同 latency/row-buffer 行为，而 analytical mode 不要求。

---

## P1-10：tile capacity 不够，建议增加 bank capacity

### 当前现象

`MemoryManager` 只维护 `tile_usage`，但 object 可能只映射到 `B0-1`。如果许多 object 都塞到同一 bank range，只要 tile 总容量没爆，就不会报错。

### 修改方案

增加：

```python
self.bank_usage[(tile, bank)] = 0
self.bank_capacity[(tile, bank)] = config.bank_capacity_kb * 1024
```

allocate 时检查每个 bank：

```python
bytes_per_bank = ceil_div(obj.bytes, len(sram_banks))
for b in sram_banks:
    if self.bank_usage[(tile, b)] + bytes_per_bank > self.bank_capacity[(tile, b)]:
        return False
```

### 验收测试

将多个 object 全部分配到 `B0-0`，即使 tile 总容量够，bank capacity 爆时也必须 spill/报错。

---

# 5. P2 级工程化增强

---

## P2-01：重构 Trace IR，减少字符串过载

当前 `TraceCommand` 使用 `src` / `dst` 字符串表示 object id、location、DRAM addr。建议引入结构化字段：

```python
@dataclass
class TraceCommand:
    cmd_id: int
    op: OpCode
    dst_obj: str = ""
    src_objs: list[str] = field(default_factory=list)
    sram_loc: Optional[SRAMLoc] = None
    dram_addr: Optional[int] = None
    nbytes: int = 0
    attrs: dict = field(default_factory=dict)
    deps: list[int] = field(default_factory=list)
```

保留旧格式 parser，但内部统一转成结构化 IR。

---

## P2-02：增加 Trace Validator

在仿真前静态检查：

```text
每个 src object 是否已有 producer / DMA_LOAD / preloaded alloc
每个 PIM command 是否依赖其输入 producer
同一个 object 的多个 writer 是否有 WAW dependency
DMA_STORE 是否依赖最后 producer
SRAM_FREE dirty object 是否之前有 DMA_STORE
是否存在未知 dependency id
是否存在循环 dependency
```

示例：

```python
class TraceValidator:
    def validate(self, commands: list[TraceCommand]) -> list[TraceError]:
        ...
```

CLI 中默认：

```bash
python main.py --validate-trace --strict
```

---

## P2-03：补充 CI tests

建议最少添加以下测试文件：

```text
tests/test_memory_lifecycle.py
tests/test_capacity_spill.py
tests/test_dirty_writeback.py
tests/test_pim_dependency.py
tests/test_timing_ceil.py
tests/test_resource_conflict.py
tests/test_energy_accounting.py
tests/test_trace_validator.py
```

关键 negative tests：

1. alloc but not load -> PIM，应报错；
2. dirty free without store，应报错；
3. invalid dependency id，应报错；
4. insufficient capacity，应触发 spill/writeback；
5. bytes 非整除 word bytes，应 ceil；
6. 两个 PIM 使用同 bank，应冲突或串行；
7. power-gate dirty object，应报错。

---

## P2-04：增强 report

建议 report 增加：

```json
"utilization": {
  "bank_active_cycles": ..., 
  "pim_active_cycles": ...,
  "dram_active_cycles": ...,
  "noc_active_cycles": ...
},
"stalls": {
  "dependency_stall_cycles": ...,
  "bank_conflict_stall_cycles": ...,
  "capacity_spill_cycles": ...,
  "dram_queue_stall_cycles": ...,
  "power_throttle_stall_cycles": ...
},
"memory_lifecycle": {
  "spill_count": ...,
  "eviction_count": ...,
  "writeback_count": ...,
  "reload_count": ...
},
"valid_simulation": true
```

---

## P2-05：统一 cold/warm/amortized 模式

建议定义三种模式：

### cold_start

统计所有权重/输入从 DRAM 加载：

```text
initial weight DMA_LOAD counted
intermediate spills counted
final output DMA_STORE counted
```

### warm_resident

权重/LUT/常量假设已经 resident：

```text
initial preload not counted
but SRAM capacity still occupied
input DMA_LOAD counted
final output DMA_STORE counted if required
```

### amortized

初始 preload 成本按查询次数摊销：

```python
amortized_preload_energy = preload_energy / num_queries
amortized_preload_latency = preload_latency / num_queries
```

当前 `SystemConfig` 中已有 `num_queries_for_amortization`、`count_initial_preload`、`count_final_writeback` 的雏形，建议真正接入 trace generator 和 simulator report。

---

# 6. 推荐修改顺序

## Phase 1：正确性闭环，优先级最高

目标：任何非法 trace 都不能被误判为合法结果。

1. 拆分 `allocate` 与 `valid`；
2. PIM 输入 invalid 时 strict 报错；
3. issue/complete 分离，状态在 complete 时生效；
4. dirty free/evict/power-gate 必须 writeback 或报错；
5. deadlock 不再强制 complete；
6. GEMM partial-sum dependency 修正。

完成标准：

```text
所有 strict tests 通过
correctness counters 全为 0
非法 trace 必须失败
```

## Phase 2：容量与 DRAM/SRAM 数据生命周期闭环

1. 实现 `allocate_or_spill()`；
2. 实现 dirty victim writeback；
3. 增加 bank-level capacity；
4. 增加 reload 机制；
5. report 加入 spill/writeback/reload。

完成标准：

```text
小 SRAM 配置下能看到 spill/writeback traffic
没有 capacity_overcommit warning
```

## Phase 3：timing/resource 精细化

1. PIM MAC latency 按 effective lanes；
2. 所有 floor division 改 ceil；
3. 修正 DMA_LOAD/STORE 端口方向；
4. PIM_NL 纳入 PIM resource；
5. `_get_banks()` 使用 object 实际位置；
6. 增加 issue width / NoC bandwidth / DMA engine 数量。

完成标准：

```text
active banks 越多，PIM latency 合理下降
bank conflict 能改变 total cycles
DMA/PIM overlap 受端口和 NoC 限制
```

## Phase 4：energy/DESTINY 接入

1. DESTINY 参数覆盖 SimConfig；
2. PIM MAC 绑定 SRAM read/write；
3. EW/MAC/REDUCE/NL energy 分开；
4. leakage 按 power state；
5. report 标明 analytical / DESTINY / RTL 参数来源。

完成标准：

```text
SRAM capacity/word width/tech 改变会影响 SRAM energy/latency/area
PIM energy breakdown 与数据流一致
```

## Phase 5：DRAM 严格模式与 trace validator

1. 接入 DRAMSim3/Ramulator trace mode；
2. 增加 trace validator；
3. 增加 workload-level golden tests；
4. 输出论文可用报告。

---

# 7. 建议新增代码结构

推荐目录：

```text
src/
  config.py
  trace_ir.py
  trace_validator.py
  simulator.py
  event.py
  memory_object.py
  memory_manager.py
  dram/
    base.py
    analytical.py
    dramsim3.py
    ramulator.py
  energy/
    energy_model.py
    destiny_adapter.py
    rtl_adapter.py
  resource/
    bank_resource.py
    noc_resource.py
    dma_resource.py
    power_resource.py
  tracegen/
    gen_gemm_trace.py
    gen_attention_trace.py
    gen_ssm_trace.py
  utils/
    math_utils.py
    location.py
```

其中 `utils/math_utils.py`：

```python
def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return (a + b - 1) // b
```

`utils/location.py`：

```python
@dataclass(frozen=True)
class SRAMLoc:
    tile: int
    banks: tuple[int, ...]


def parse_sram_loc(s: str) -> SRAMLoc:
    ...


def global_bank_ids(loc: SRAMLoc, banks_per_tile: int) -> list[int]:
    return [loc.tile * banks_per_tile + b for b in loc.banks]
```

---

# 8. 最小 patch 骨架

下面是建议最先落地的最小 patch 方向。

## 8.1 `MemoryObject`

```python
@dataclass
class MemoryObject:
    object_id: str
    obj_type: ObjType
    bytes: int
    precision: str
    dram_addr: int = 0
    sram_tile: int = -1
    sram_banks: list[int] = field(default_factory=list)
    valid_in_dram: bool = True
    valid_in_sram: bool = False
    dirty_in_sram: bool = False
    pinned: bool = False
    power_state: str = "active"
    state: str = "DRAM_ONLY"

    def place_in_sram(self, tile: int, banks: list[int]):
        self.sram_tile = tile
        self.sram_banks = list(banks)
        self.state = "SRAM_RESERVED"

    def commit_load(self):
        self.valid_in_sram = True
        self.dirty_in_sram = False
        self.state = "VALID_CLEAN"

    def mark_dirty(self):
        self.valid_in_sram = True
        self.dirty_in_sram = True
        self.valid_in_dram = False
        self.state = "VALID_DIRTY"

    def writeback_complete(self):
        self.valid_in_dram = True
        self.dirty_in_sram = False
        self.state = "VALID_CLEAN"
```

## 8.2 `Simulator._require_inputs_valid`

```python
def _require_inputs_valid(self, cmd: TraceCommand) -> None:
    for iid in parse_src_ids(cmd.src):
        obj = self.mem_mgr.objects.get(iid)
        if obj is None:
            raise RuntimeError(f"cmd {cmd.cmd_id}: missing input {iid}")
        if not obj.valid_in_sram:
            self.correctness["illegal_read_unresident_object"] += 1
            raise RuntimeError(f"cmd {cmd.cmd_id}: input {iid} is not resident in SRAM")
        if obj.power_state != "active":
            raise RuntimeError(f"cmd {cmd.cmd_id}: input {iid} not active")
```

## 8.3 `Simulator._complete_command`

```python
def _complete_command(self, cmd: TraceCommand) -> None:
    if cmd.op == OpCode.DMA_LOAD:
        obj = self.mem_mgr.objects[cmd.object_id]
        obj.commit_load()
        self.energy.add_sram_write(ceil_div(cmd.bytes, self.config.sram_pim.word_bytes))

    elif cmd.op == OpCode.DMA_STORE:
        obj = self.mem_mgr.objects[cmd.object_id]
        self.energy.add_sram_read(ceil_div(cmd.bytes, self.config.sram_pim.word_bytes))
        obj.writeback_complete()

    elif cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE, OpCode.PIM_NL}:
        obj = self.mem_mgr.objects[cmd.object_id]
        obj.mark_dirty()
```

---

# 9. 论文/实验使用前的验收标准

在用于任何论文实验或定量比较前，建议要求每个 workload 报告满足：

```text
valid_simulation = true
correctness counters all zero
capacity_overcommit_events = 0
deadlock_events = 0
all final outputs either valid_in_dram or explicitly marked SRAM-resident output
all dirty non-temporary objects are written back or intentionally retained
```

并在论文或报告中明确说明：

```text
DRAM model: analytical / DRAMSim3 / Ramulator
SRAM macro params: analytical / DESTINY / CACTI / RTL
PIM compute energy: analytical / RTL / circuit estimate
Simulation mode: cold_start / warm_resident / amortized
Capacity policy: spill/writeback/eviction policy
Power policy: clock-gating / power-gating considered or not
```

---

# 10. 一句话修改策略

不要优先扩展更多算子。当前最应该先修的是：

```text
SRAM_ALLOC != valid
invalid input must stop
side effects at completion
capacity spill/writeback closed loop
dirty object cannot be silently freed
GEMM psum dependency strict
```

这 6 件事完成后，当前 SRAM-PIM 仿真器才真正从“可以跑的原型”升级为“有严格数据生命周期约束的架构级 simulator”。
