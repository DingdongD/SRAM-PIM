# SRAM-PIM `fix/strict-simulator` 最新复查与下一轮修复指南

> 目标：基于当前 `fix/strict-simulator` 分支的最新修改，复核其是否已经满足严格 SRAM-PIM architectural simulator 的要求，并给出下一轮需要继续修复的边界问题。
>
> 当前结论：这一版已经把上一轮的主要 P0 问题基本落地，可以称为 **严格 SRAM-PIM architectural simulator prototype**。但若要进一步接近论文级、可解释、可复现实验框架，仍建议补齐 protected spill、bank-aware spill、latency breakdown 去重、event-level spill 命名或实现等问题。

---

## 1. 总体结论

当前版本已经从上一轮的：

```text
严格语义设计完整，但主实现未闭环
```

提升为：

```text
严格 SRAM-PIM architectural simulator prototype
```

我对当前成熟度的判断是：

```text
8.0 ~ 8.3 / 10
```

当前已经比较接近论文级 architectural simulator 原型，但仍不能声称是：

```text
- cycle-accurate DRAM-backed SRAM-PIM simulator
- circuit-level SRAM-CIM simulator
- full operator-accurate transformer / SSM simulator
```

当前可以较有底气地表述为：

```text
The simulator models a DRAM-backed digital near-SRAM PIM architecture at the architectural command level. It explicitly tracks SRAM residency, valid/dirty state, DMA load/store, capacity spill, dirty writeback, auto-reload, final dirty policy, and provenance of timing/energy parameters.
```

---

## 2. 当前已经修复的关键点

---

### 2.1 PIM output allocation spill 已经补上

#### 之前的问题

上一版中，`_ensure_output_object()` 在 PIM output object 未分配 SRAM 时，只调用：

```python
self.mem_mgr.allocate(cmd.object_id, loc.tile, list(loc.banks))
```

但没有检查返回值，也没有在分配失败时调用 `_allocate_or_spill()`。

这会导致：

```text
PIM output allocation failure 被忽略
output object 仍可能进入 PRODUCING / VALID_DIRTY 状态
SRAM 容量约束被绕过
spill/writeback latency 和 energy 被低估
```

#### 当前状态

当前 `_ensure_output_object()` 已经会：

```text
1. 创建 output object；
2. 解析 dst SRAM 位置；
3. 如果 object 尚未放入 SRAM，则尝试 allocate；
4. 如果 allocate 失败，则调用 _allocate_or_spill()；
5. 返回 output allocation / spill 带来的 extra_lat；
6. PIM_MAC / PIM_EW_OP / PIM_REDUCE / PIM_NL 将 alloc_lat 加入 command latency。
```

因此现在 PIM output 不再能绕过 SRAM 容量约束。

#### 当前评价

```text
状态：基本合格
严格性：architectural-level 合格
```

#### 推荐保留测试

```bash
pytest tests/test_output_allocation_spill.py -q
```

建议测试覆盖：

```text
- clean victim eviction
- dirty victim writeback
- all-pinned no victim raise
- output allocation 成功后 object 才能进入 VALID_DIRTY
```

---

### 2.2 `auto_reload` 已经进入主逻辑

#### 之前的问题

上一版中，`SystemConfig` 已声明：

```python
correctness_mode: str = "strict"  # strict | warn | auto_reload
```

但 simulator 主逻辑只把 strict mode 转成布尔：

```python
self.strict = config.system.correctness_mode == "strict"
```

`auto_reload` 没有真正实现。

#### 当前状态

当前 simulator 已经新增输入 ready 检查逻辑，例如：

```python
_ensure_inputs_ready()
_blocking_reload()
```

其语义大致为：

```text
strict:
  input 不在 SRAM -> raise

warn:
  input 不在 SRAM -> 计数但继续

auto_reload:
  input 不在 SRAM 且 valid_in_dram=True -> blocking reload
  input 不在 SRAM 且 valid_in_dram=False -> raise
```

这已经把 auto-reload 从配置/测试层推进到了主实现层。

#### 当前评价

```text
状态：基本合格
严格性：architectural-level 合格
```

#### 推荐保留测试

```bash
pytest tests/test_auto_reload.py -q
```

测试应覆盖：

```text
- strict 模式下 evicted input 被使用时 raise
- auto_reload 模式下 DRAM copy 有效时成功 reload
- auto_reload 模式下 DRAM copy 无效时 raise
- reload_count / DRAM read bytes / SRAM write energy 增加
```

---

### 2.3 final dirty policy 已经实现

#### 之前的问题

上一版中，配置已有：

```python
final_dirty_policy: str = "report"  # error | auto_writeback | report | ignore
```

但 simulator 结束时没有扫描 dirty persistent object。

这会导致：

```text
output/state/meta 仍 dirty in SRAM
trace 没有 DMA_STORE
仿真仍然输出 valid result
```

#### 当前状态

当前 simulator 已经新增：

```python
_finalize_simulation()
```

并在 run 结束后根据 `final_dirty_policy` 做：

```text
error:
  若存在 dirty persistent object，则 raise

auto_writeback:
  自动写回 dirty persistent object

report:
  不报错，但 report 中列出 dirty object，并标记 valid_simulation=False 或 final_dirty_warning

ignore:
  不处理，但应在 report 中标记 policy
```

#### 当前评价

```text
状态：合格
严格性：architectural-level 合格
```

#### 推荐保留测试

```bash
pytest tests/test_final_dirty_policy.py -q
```

测试应覆盖：

```text
- final_dirty_policy=error 时 raise
- final_dirty_policy=auto_writeback 时 writeback_count 增加
- final_dirty_policy=report 时 final_dirty_objects 被记录
- valid_simulation 语义与 policy 一致
```

---

### 2.4 report provenance 已经补上

当前 report 中已经加入了类似如下 provenance 信息：

```yaml
correctness_mode: strict
final_dirty_policy: report
spill_model: blocking
timing_fidelity: architectural_blocking_spill
dram_model: analytical
sram_param_source: analytical/destiny
pim_mode: digital_near_sram
pim_energy_source: analytical
workload_trace_semantics: operator_level_approximation
```

这对论文写作很重要，因为它能避免 simulator 能力被过度声明。

#### 当前评价

```text
状态：合格
严格性：论文表述友好
```

建议所有实验结果都输出 provenance，否则 reviewer 很容易质疑：

```text
- SRAM 参数到底来自 DESTINY 还是手填？
- DRAM 是 analytical 还是 cycle accurate？
- PIM MAC energy 是 RTL、SPICE、还是 analytical？
- spill/writeback 是 blocking 还是 event-level？
```

---

## 3. 当前仍然存在的关键问题

---

## P1-A. `valid_simulation` 对 `auto_reload` 可能过于严格

### 问题描述

当前 `_ensure_inputs_ready()` 在发现 input 不在 SRAM 时，可能会增加：

```text
illegal_read_unresident_object
```

即使后续 `auto_reload` 成功完成，这个 counter 也可能大于 0。

如果 `valid_simulation` 的计算方式是：

```python
valid_simulation = all(correctness_counter == 0)
```

那么会出现：

```text
auto_reload 成功，本质上是合法 cache miss / reload 行为；
但 report 仍将 simulation 标记为 invalid。
```

### 为什么这是问题？

在 `auto_reload` 模式下，unresident input 不一定是错误。它应该更像 cache miss：

```text
unresident input event -> reload from DRAM -> command becomes legal
```

因此它应该被记录为 lifecycle event，而不是 correctness error。

### 建议修改

将 counter 拆成两类：

```yaml
correctness_errors:
  missing_input_object: 0
  invalid_dram_copy: 0
  illegal_read_power_gated_object: 0
  final_dirty_error: 0

lifecycle_events:
  unresident_input_events: 3
  reload_count: 3
  eviction_count: 2
  spill_count: 1
```

或者最小修改为：

```python
def _compute_valid_simulation(self) -> bool:
    mode = self.config.system.correctness_mode

    counters = dict(self.correctness)

    if mode == "auto_reload":
        # unresident input is legal if reload succeeds
        counters.pop("illegal_read_unresident_object", None)

    return all(v == 0 for v in counters.values()) and self.finalization_valid
```

### 建议新增测试

```python
def test_auto_reload_success_keeps_valid_simulation_true():
    """
    auto_reload 成功后，simulation 不应因为 unresident event 被标记为 invalid。
    """
```

验收标准：

```text
- auto_reload 成功时 reload_count >= 1
- valid_simulation == True
- unresident_input_events 可以大于 0
- correctness_errors 应为 0
```

---

## P1-B. spill victim selection 需要保护当前 command 的 input objects

### 问题描述

当前 `_allocate_or_spill()` 会从目标 tile 上选择 victim。若它不知道当前 PIM command 的 input objects，就可能在极端情况下选中当前 command 正在使用的 input。

可能出现：

```text
PIM command 需要 input A、B
output allocation 触发 spill
victim selection 选择 A 或 B
A/B 被 evict
PIM command 继续执行
```

这会破坏核心不变量：

```text
PIM command 执行时，所有 input 必须 valid in SRAM。
```

### 建议修改

给 `_allocate_or_spill()` 增加 `protected_ids` 参数：

```python
def _allocate_or_spill(
    self,
    object_id: str,
    tile: int,
    banks: list[int],
    protected_ids: set[str] | None = None,
) -> int:
    ...
```

PIM command 调用时：

```python
input_ids = set(self._parse_input_object_ids(cmd))
protected_ids = input_ids | {cmd.object_id}

alloc_lat = self._ensure_output_object(
    cmd,
    ObjType.PSUM,
    "int32",
    protected_ids=protected_ids,
)
```

`MemoryManager.find_eviction_candidate()` 也应支持：

```python
def find_eviction_candidate(
    self,
    tile: int,
    needed_bytes: int,
    target_banks: list[int] | None = None,
    protected_ids: set[str] | None = None,
) -> MemoryObject | None:
    ...
```

### victim selection 规则

建议优先级：

```text
1. 不允许 evict protected_ids；
2. 不允许 evict pinned objects；
3. 优先 evict clean activation / temporary / psum；
4. 如果必须 evict dirty object，则先 writeback；
5. 如果无 victim，则 strict raise。
```

### 建议新增测试

```python
def test_output_spill_does_not_evict_current_inputs():
    """
    PIM output allocation 触发 spill 时，
    victim selection 不能 evict 当前 command 的 src input。
    """
```

验收标准：

```text
- output allocation spill 发生后，当前 command input 仍 valid_in_sram=True
- protected input 不出现在 evicted object 列表中
- 若只有 protected objects 可 evict，则 strict mode raise
```

---

## P1-C. spill victim selection 需要 bank-aware

### 问题描述

当前 `MemoryManager.allocate()` 可能同时检查：

```text
tile capacity
bank capacity
```

但 `_allocate_or_spill()` 的 victim selection 若主要基于 tile-level needed bytes，则可能在 bank overflow 时 evict 错对象。

例如：

```text
tile 总容量仍足够
但 bank 0 已满
new object 必须放入 bank 0
victim selection 却 evict bank 3 上的 object
allocation retry 仍然失败
```

这说明 spill 应该区分：

```text
tile_capacity_spill
bank_capacity_spill
```

### 建议修改

`find_eviction_candidate()` 应接受目标 banks：

```python
def find_eviction_candidate(
    self,
    tile: int,
    needed_bytes: int,
    target_banks: list[int] | None = None,
    protected_ids: set[str] | None = None,
) -> MemoryObject | None:
    ...
```

选择 victim 时：

```text
如果 target_banks 非空：
  优先选择占用 target_banks 的 object
否则：
  按 tile-level capacity pressure 选择 victim
```

### 建议 bank-aware victim score

```python
def victim_score(obj, target_banks):
    score = 0

    # Prefer objects that overlap target banks.
    overlap = len(set(obj.sram_banks) & set(target_banks))
    score += 100 * overlap

    # Prefer clean objects.
    if not obj.dirty_in_sram:
        score += 50

    # Prefer temporary objects.
    if obj.obj_type in (ObjType.TEMP, ObjType.PSUM):
        score += 20

    # Prefer larger objects if freeing more capacity helps.
    score += obj.bytes / 1024

    return score
```

### 建议新增测试

```python
def test_bank_overflow_spill_evicts_target_bank_object():
    """
    tile 总容量足够，但目标 bank 满；
    spill 应该 evict 占用目标 bank 的对象，而不是其他 bank 的对象。
    """
```

验收标准：

```text
- bank overflow 时 evicted victim 与 target_banks 有交集
- allocation retry 成功
- 不应因为 evict 错 bank 导致反复失败
```

---

## P1-D. `spill_model="event_level"` 当前仍更像 event-level stub

### 问题描述

当前代码已经具备类似：

```python
_writeback_victim()
_blocking_writeback()
_event_level_writeback()
```

的接口，这说明你已经开始为 event-level spill 做准备。

但如果 `_event_level_writeback()` 实际仍是同步计算 latency/energy，然后直接：

```text
writeback_complete()
evict()
retry allocation
```

那么它本质上仍然是 blocking model，只是接口叫 event-level。

### 为什么这是问题？

真正 event-level spill 应该进入统一事件队列，并占用：

```text
- DMA engine
- DRAM channel
- NoC bandwidth
- SRAM read port
- SRAM bank resource
```

否则它无法模拟 spill/writeback 与正常 DMA/PIM command 的资源竞争。

### 短期建议

短期如果不真正实现 event queue，建议将 provenance 改得更保守：

```yaml
spill_model: blocking
timing_fidelity: architectural_blocking_spill
```

或者：

```yaml
spill_model: event_level_stub
timing_fidelity: blocking_writeback_with_event_level_interface
```

不建议写成：

```yaml
timing_fidelity: architectural_event_level_spill
```

除非已经真正通过 event queue 调度 internal DMA_STORE。

### 中期建议

实现真正 event-level writeback：

```text
ALLOC fails
  -> select victim
  -> if victim dirty:
       create INTERNAL_DMA_STORE command
       reserve SRAM read / DMA / DRAM / NoC resource
       enqueue event
       wait until event complete
  -> writeback_complete
  -> evict victim
  -> retry allocation
```

### internal DMA_STORE 示例

```python
internal_cmd = TraceCommand(
    cmd_id=self._new_internal_cmd_id(),
    op=OpCode.DMA_STORE,
    object_id=victim.object_id,
    src=f"SRAM:T{victim.sram_tile}:B{victim.sram_banks}",
    dst=f"DRAM:{victim.object_id}",
    bytes=victim.bytes,
    attrs={
        "internal": True,
        "reason": "spill_writeback",
    },
)
```

### 建议新增测试

```python
def test_event_level_spill_uses_dma_resource():
    """
    event_level spill 应通过统一 DMA path，
    并占用 DMA/NoC/DRAM/SRAM read resource。
    """
```

---

## P1-E. latency breakdown 存在重叠统计风险

### 问题描述

如果 PIM command 的 latency 写法类似：

```python
lat = compute_lat + alloc_lat + reload_lat
self.latency_breakdown["pim_compute_cycles"] += lat
```

同时其他地方又统计：

```python
self.latency_breakdown["auto_reload_cycles"] += reload_lat
self.latency_breakdown["spill_writeback_cycles"] += alloc_lat
```

那么 breakdown 会出现语义重叠：

```text
pim_compute_cycles 包含 compute + reload + spill
auto_reload_cycles 又单独包含 reload
spill_writeback_cycles 又单独包含 spill
```

### 影响

这会导致报告解释困难：

```text
pim_compute_cycles 到底是不是纯计算？
总 latency breakdown 各项是否可加？
reload/spill 是否被重复解释？
```

### 建议修改

将 PIM command 分成：

```python
compute_lat = self._compute_pim_mac_latency(cmd)
reload_lat = self._ensure_inputs_ready(cmd)
alloc_lat = self._ensure_output_object(cmd, ...)

total_lat = reload_lat + alloc_lat + compute_lat

self.latency_breakdown["pim_compute_cycles"] += compute_lat
self.latency_breakdown["auto_reload_cycles"] += reload_lat
self.latency_breakdown["output_alloc_spill_cycles"] += alloc_lat

return total_lat
```

### report 建议区分

```yaml
latency_breakdown:
  pim_compute_cycles: ...
  pim_ew_cycles: ...
  pim_reduce_cycles: ...
  pim_nl_cycles: ...
  dma_load_cycles: ...
  dma_store_cycles: ...
  auto_reload_cycles: ...
  output_alloc_spill_cycles: ...
  final_writeback_cycles: ...
```

### 建议新增测试

```python
def test_latency_breakdown_no_double_count_reload():
    """
    auto_reload command 的 total latency 可以包含 reload，
    但 pim_compute_cycles 不应包含 reload cycles。
    """
```

---

## P1-F. final auto-writeback 不应混入 capacity spill 统计

### 问题描述

`_finalize_simulation()` 在 `auto_writeback` 下会调用 `_blocking_writeback()`。

如果 `_blocking_writeback()` 内部会增加：

```text
spill_count
```

那么 final writeback 也会被统计为 spill。

但二者语义不同：

```text
spill:
  因 SRAM capacity pressure 触发的写回/淘汰

final_writeback:
  因仿真结束 policy 触发的 dirty flush
```

### 建议修改

将 writeback reason 显式传入：

```python
def _blocking_writeback(
    self,
    obj: MemoryObject,
    reason: str = "spill",
) -> int:
    ...
```

统计时：

```python
if reason == "spill":
    self.mem_mgr.stats["spill_count"] += 1
elif reason == "final":
    self.mem_mgr.stats["final_writeback_count"] += 1
elif reason == "auto_reload_eviction":
    self.mem_mgr.stats["reload_eviction_writeback_count"] += 1
```

### 同时补 leakage

如果 final auto-writeback 会增加：

```python
self.cycle += total_lat
```

那么应补充这段时间的 leakage energy：

```python
self.energy.add_leakage(total_lat)
```

或者在统一的 cycle advance 函数中处理：

```python
def _advance_cycles(self, cycles: int, reason: str):
    self.cycle += cycles
    self.energy.add_leakage(cycles)
```

### 建议新增测试

```python
def test_final_auto_writeback_not_counted_as_spill():
    """
    final_dirty_policy=auto_writeback 触发的写回，
    应计入 final_writeback_count，而不是 spill_count。
    """
```

---

## 4. 当前严格性矩阵

| 项目 | 当前状态 | 评价 |
|---|---:|---:|
| `SRAM_ALLOC != valid` | 已实现 | 合格 |
| PIM input strict validation | 已实现 | 合格 |
| PIM output allocation spill | 已实现 | 基本合格 |
| dirty victim writeback | 已实现 | 基本合格 |
| auto_reload | 已实现 | 基本合格 |
| final dirty policy | 已实现 | 合格 |
| report provenance | 已实现 | 合格 |
| current-input spill protection | 未见完整保护 | 需要补 |
| bank-aware spill | 仍较粗 | 需要补 |
| event-level spill | 接口有，实质仍 blocking/stub | 需要标注或实现 |
| latency breakdown | 有重叠统计风险 | 需要修 |
| final writeback vs spill stats | 可能混淆 | 需要修 |
| DRAM timing | analytical | 可接受，但需声明 |
| SRAM-CIM energy | digital near-SRAM analytical | 可接受，但需声明 |
| tests | 明显增强 | 建议继续加边界测试 |

---

## 5. 建议下一步最优先修的 4 件事

### 5.1 修 `valid_simulation` 在 `auto_reload` 下的语义

成功 reload 不应默认使 simulation invalid。

```text
unresident input event != correctness error
```

建议拆分：

```text
correctness_errors
lifecycle_events
```

---

### 5.2 给 `_allocate_or_spill()` 增加 `protected_ids`

防止 spill 当前 PIM command 的 input/output。

```python
protected_ids = set(input_ids) | {cmd.object_id}
```

---

### 5.3 修 latency breakdown 去重

`pim_compute_cycles` 应只统计纯 PIM compute，不应包含 reload/spill/writeback。

---

### 5.4 重新命名或真正实现 `event_level` spill

当前若没有 internal DMA event queue，则更准确表述为：

```text
blocking writeback with event-level interface
```

---

## 6. 建议 CI 测试集合

当前建议至少保留：

```bash
pytest tests/test_memory_lifecycle.py -q
pytest tests/test_output_allocation_spill.py -q
pytest tests/test_auto_reload.py -q
pytest tests/test_final_dirty_policy.py -q
pytest tests/test_timing_ceil.py -q
pytest tests/test_resource_model.py -q
pytest tests/test_energy_accounting.py -q
pytest tests/test_gemm_trace.py -q
```

建议新增：

```bash
pytest tests/test_protected_spill.py -q
pytest tests/test_bank_aware_spill.py -q
pytest tests/test_latency_breakdown.py -q
pytest tests/test_final_writeback_stats.py -q
```

新增测试应覆盖：

```text
- protected input 不会被当前 command 的 output spill evict
- bank overflow 时 evict target bank 上的 object
- auto_reload 成功后 valid_simulation=True
- final auto-writeback 不计入 capacity spill
- pim_compute_cycles 不重复统计 reload/spill
```

---

## 7. 论文中推荐表述

### 推荐表述

```text
We build a DRAM-backed architectural simulator for a digital near-SRAM PIM architecture. The simulator uses command-level traces to model DMA load/store, SRAM residency, valid/dirty object state, PIM execution, capacity spill, dirty writeback, and final dirty-state handling. SRAM macro parameters can be supplied analytically or through DESTINY, while DRAM is modeled as an analytical backing store unless otherwise specified.
```

### 不推荐表述

除非继续接入 DRAMSim3/Ramulator/RTL/SPICE，否则不建议写：

```text
cycle-accurate DRAM timing
circuit-level SRAM-CIM energy
ADC/DAC-aware CIM model
bitline-level SRAM compute model
full transformer operator-accurate simulation
```

### 推荐 provenance disclosure

```text
All reported results use the architectural timing model with analytical DRAM latency and bandwidth, digital near-SRAM PIM energy, and explicit SRAM capacity spill/writeback modeling. We report the simulator configuration, memory provenance, and timing/energy parameter sources with each experiment.
```

---

## 8. 最终判断

这次修改后，框架已经比较扎实。

当前版本可以作为：

```text
- 论文中的 architectural simulator 原型；
- SRAM-PIM 数据生命周期验证框架；
- capacity spill / auto_reload / final_dirty 策略研究框架；
- digital near-SRAM PIM 的趋势分析工具。
```

但仍需明确：

```text
- DRAM 是 analytical；
- PIM 是 digital near-SRAM；
- spill 默认仍偏 blocking；
- energy 是 architectural-level estimation；
- attention / SSM trace 是 operator-level approximation。
```

如果继续补齐以下问题：

```text
1. protected spill
2. bank-aware spill
3. latency breakdown 去重
4. final writeback 与 capacity spill 统计分离
5. event-level spill 真实进入 event queue
```

则 simulator 可以进一步从当前约：

```text
8.0 ~ 8.3 / 10
```

提升到：

```text
8.7 / 10 左右
```

届时作为论文架构评估框架会更加稳健。
