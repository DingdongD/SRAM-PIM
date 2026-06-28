# SRAM-PIM `fix/strict-simulator` 最新审查与下一步修复指南

> 目标：将当前 `fix/strict-simulator` 分支从“严格仿真器测试/配置雏形”推进到“实现与测试语义一致的严格 SRAM-PIM architectural simulator 原型”。
>
> 本文基于当前可见的 `fix/strict-simulator` 分支代码审查结论整理，重点检查上一轮提出的四个核心问题是否闭环：
>
> 1. PIM output allocation spill；
> 2. final dirty object 检查；
> 3. `auto_reload` 语义；
> 4. spill/writeback timing 是否进入更严格路径。

---

## 1. 总体结论

当前分支已经新增了正确方向的配置项和测试文件，但核心 simulator 实现仍未完全闭环。

更准确地说：

```text
测试设计：已经接近严格 simulator 要求
配置接口：已经接近严格 simulator 要求
主 simulator 实现：仍缺 PIM output spill、auto_reload、final dirty finalize 三个关键闭环
```

因此当前版本还不能认为已经严格完善。

推荐当前成熟度判断：

```text
数据生命周期：8 / 10
SRAM_ALLOC != valid：已正确
strict invalid input：已正确
PIM output allocation spill：仍未闭环
auto_reload：测试有，主逻辑缺
final dirty：配置有，主逻辑缺
spill/writeback：blocking 闭环，event-level 未闭环
DRAM timing：仍是 analytical
energy：架构级可用，电路级不足

总体成熟度：约 7.2 / 10
```

当前版本可以继续作为：

```text
- SRAM-PIM 数据生命周期验证
- command-level SRAM-PIM 原型
- cold/warm residency 对比
- capacity spill 行为测试框架
```

但暂时不建议直接用于：

```text
- 强声明 cycle-accurate DRAM timing
- 强声明 circuit-level SRAM-CIM energy
- 未经 RTL/DESTINY/DRAMSim3 校准的论文级强定量 speedup/energy 结论
```

---

## 2. 已经补得比较好的部分

### 2.1 配置层已经加入严格策略参数

当前 `SystemConfig` 已经具备如下策略接口：

```python
correctness_mode: str = "strict"      # strict | warn | auto_reload
final_dirty_policy: str = "report"    # error | auto_writeback | report | ignore
spill_model: str = "blocking"         # blocking | event_level
```

这说明框架已经开始支持以下严格语义：

```text
strict:
  input 不在 SRAM 时直接报错

warn:
  input 不在 SRAM 时只记录 correctness warning

auto_reload:
  input 不在 SRAM 但 DRAM copy 有效时自动 reload

final_dirty_policy:
  控制仿真结束时 dirty persistent object 的处理方式

spill_model:
  控制 capacity spill / dirty writeback 是 blocking 还是 event-level
```

配置方向是正确的。

---

### 2.2 测试侧已经开始覆盖 P0 correctness 场景

新增测试已经覆盖关键严格语义，例如：

```text
test_output_allocation_spill.py:
  - PIM output 分配时 SRAM 满，需要触发 victim eviction
  - dirty victim 需要 writeback
  - writeback_count / eviction_count / DRAM write bytes 需要增加

test_auto_reload.py:
  - strict 模式下使用 evicted object 应该报错
  - auto_reload 模式下，如果 DRAM copy 有效，应自动 reload
  - reload_count 应增加
```

这些测试设计方向是正确的。

但目前主要问题是：**测试表达了期望语义，主 simulator 代码还没有完全实现这些语义。**

---

## 3. 当前仍未闭环的关键问题

---

## P0-A. `_ensure_output_object()` 没有处理 output allocation failure

### 问题描述

当前 `PIM_MAC` 会调用 `_ensure_output_object()` 创建输出 object，然后继续 `begin_producing()`。

但是 `_ensure_output_object()` 内部如果调用：

```python
self.mem_mgr.allocate(cmd.object_id, loc.tile, list(loc.banks))
```

却没有检查返回值，也没有在失败时调用 `_allocate_or_spill()`。

这意味着如果 SRAM 已满，PIM output 分配失败，仿真器仍可能继续执行：

```text
create output object
allocate failure ignored
begin_producing
commit_produce
output 被标记为 VALID_DIRTY
```

这会直接破坏严格 SRAM 容量约束。

### 影响

该问题会导致：

```text
- SRAM 容量约束失效；
- PIM output / psum 可能凭空存在；
- spill / writeback traffic 被低估；
- energy / latency 结果偏乐观；
- output_allocation_spill 测试与实现不一致。
```

### 修改目标

`_ensure_output_object()` 必须：

```text
1. 创建 output object；
2. 解析 dst SRAM 位置；
3. 如果 output object 还未分配 SRAM 位置，则尝试 allocate；
4. 如果 allocate 失败，调用 _allocate_or_spill()；
5. 返回 output allocation / spill 额外 latency；
6. PIM_MAC / PIM_EW_OP / PIM_REDUCE / PIM_NL 都必须把该 latency 加入 command latency。
```

### 建议接口

```python
def _ensure_output_object(
    self,
    cmd: TraceCommand,
    obj_type: ObjType,
    precision: str,
) -> int:
    """
    Ensure output object exists and has SRAM space.

    Return:
        extra_latency_cycles caused by output allocation spill/writeback.
    """
```

### 建议实现伪代码

```python
def _ensure_output_object(self, cmd, obj_type, precision) -> int:
    extra_lat = 0

    # 1. Create object metadata if missing.
    if cmd.object_id not in self.mem_mgr.objects:
        out_bytes = cmd.attrs.get("out_bytes", cmd.bytes or 1024)
        out_type = _obj_type_from_str(cmd.attrs.get("out_type", obj_type.name))
        out_precision = cmd.attrs.get("out_precision", precision)

        out_obj = MemoryObject(
            object_id=cmd.object_id,
            obj_type=out_type,
            bytes=out_bytes,
            precision=out_precision,
            valid_in_dram=False,
            valid_in_sram=False,
        )
        self.mem_mgr.register_object(out_obj)

    obj = self.mem_mgr.objects[cmd.object_id]

    # 2. Parse destination SRAM location.
    if cmd.dst and cmd.dst.startswith("SRAM:"):
        loc = parse_sram_loc(cmd.dst)

        # 3. Allocate only if object is not placed in SRAM yet.
        if obj.sram_tile < 0:
            ok = self.mem_mgr.allocate(
                cmd.object_id,
                loc.tile,
                list(loc.banks),
                make_valid=False,
            )

            # 4. Allocation failure must trigger spill/evict/writeback.
            if not ok:
                extra_lat += self._allocate_or_spill(
                    cmd.object_id,
                    loc.tile,
                    list(loc.banks),
                )

        # 5. Optional strict consistency check.
        if obj.sram_tile != loc.tile:
            raise RuntimeError(
                f"Output object {cmd.object_id} is already placed at tile "
                f"{obj.sram_tile}, but command dst requires tile {loc.tile}"
            )

    return extra_lat
```

### PIM issue path 修改

所有产生 output 的 PIM 命令都需要加入 allocation latency。

#### `PIM_MAC`

```python
reload_lat = self._require_inputs_valid(cmd)
alloc_lat = self._ensure_output_object(cmd, ObjType.PSUM, "int32")

lat = (
    reload_lat
    + alloc_lat
    + self._compute_pim_mac_latency(cmd)
)

obj = self.mem_mgr.objects[cmd.object_id]
obj.begin_producing()
return lat
```

#### `PIM_EW_OP`

```python
reload_lat = self._require_inputs_valid(cmd)
alloc_lat = self._ensure_output_object(
    cmd,
    ObjType.ACTIVATION,
    cmd.attrs.get("out_precision", "int8"),
)

lat = reload_lat + alloc_lat + self._compute_ew_latency(cmd)
```

#### `PIM_REDUCE`

```python
reload_lat = self._require_inputs_valid(cmd)
alloc_lat = self._ensure_output_object(
    cmd,
    ObjType.PSUM,
    cmd.attrs.get("out_precision", "int32"),
)

lat = reload_lat + alloc_lat + self._compute_reduce_latency(cmd)
```

#### `PIM_NL`

```python
reload_lat = self._require_inputs_valid(cmd)
alloc_lat = self._ensure_output_object(
    cmd,
    ObjType.ACTIVATION,
    cmd.attrs.get("out_precision", "int8"),
)

lat = reload_lat + alloc_lat + self._compute_nl_latency(cmd)
```

### 新增/修正测试

应保证以下测试通过：

```bash
pytest tests/test_output_allocation_spill.py -q
```

建议额外添加：

```python
def test_pim_output_allocation_failure_without_victim_raises():
    """
    如果 SRAM 已满且没有可 evict victim，
    strict mode 下 PIM output allocation 必须 raise，
    不能继续生成 output。
    """
```

验收标准：

```text
- output allocation 失败时 eviction_count 增加；
- dirty victim 被 evict 时 writeback_count 增加；
- DRAM write bytes 增加；
- 无 victim 时 strict mode raise；
- output object 不允许在未成功 allocate 的情况下 VALID_DIRTY。
```

---

## P0-B. `auto_reload` 配置和测试存在，但主逻辑未实现

### 问题描述

配置层已经声明：

```python
correctness_mode: str = "strict"  # strict | warn | auto_reload
```

但 simulator 初始化如果仍只保留：

```python
self.strict = config.system.correctness_mode == "strict"
```

而没有在 `_require_inputs_valid()` 中实现 `auto_reload` 分支，则当前语义实际是：

```text
strict:
  非 resident input 报错

warn:
  非 resident input 计数但继续

auto_reload:
  配置存在，但没有真正 reload
```

### 影响

会导致：

```text
- test_auto_reload.py 与主实现不一致；
- evicted but valid_in_dram 的 object 无法自动恢复；
- large workload 下不能模拟 cache-like SRAM residency；
- reload_count / reload traffic / reload latency 缺失。
```

### 修改目标

`_require_inputs_valid()` 应返回额外 latency，并在 `auto_reload` 模式下执行：

```text
input 不在 SRAM
  if valid_in_dram=True:
      选择 reload SRAM 位置
      如果容量不足，则 spill/evict/writeback
      执行 DRAM read + NoC transfer + SRAM write 统计
      commit_load
      reload_count += 1
      返回 reload latency
  else:
      raise RuntimeError
```

### 建议接口

```python
def _require_inputs_valid(self, cmd: TraceCommand) -> int:
    """
    Check all input objects.

    Return:
        extra_latency_cycles caused by auto reload.
    """
```

### 建议实现伪代码

```python
def _require_inputs_valid(self, cmd: TraceCommand) -> int:
    extra_lat = 0
    input_ids = self._parse_input_object_ids(cmd)

    for iid in input_ids:
        obj = self.mem_mgr.objects.get(iid)

        if obj is None:
            self.correctness["missing_input_object"] += 1
            raise RuntimeError(f"cmd {cmd.cmd_id}: missing input object {iid}")

        # 1. SRAM residency check.
        if not obj.valid_in_sram:
            self.correctness["illegal_read_unresident_object"] += 1

            mode = self.config.system.correctness_mode

            if mode == "strict":
                raise RuntimeError(
                    f"cmd {cmd.cmd_id}: input {iid} is not valid in SRAM"
                )

            elif mode == "auto_reload":
                if not obj.valid_in_dram:
                    raise RuntimeError(
                        f"cmd {cmd.cmd_id}: input {iid} not in SRAM and DRAM copy invalid"
                    )

                extra_lat += self._auto_reload_object(obj, cmd)

            elif mode == "warn":
                # Keep warning counter but continue.
                pass

            else:
                raise ValueError(f"Unknown correctness_mode: {mode}")

        # 2. Power state check.
        if obj.power_state != "active":
            self.correctness["illegal_read_power_gated_object"] += 1
            if self.config.system.correctness_mode in ("strict", "auto_reload"):
                raise RuntimeError(
                    f"cmd {cmd.cmd_id}: input {iid} is power gated"
                )

    return extra_lat
```

### `_auto_reload_object()` 建议实现

```python
def _auto_reload_object(self, obj: MemoryObject, user_cmd: TraceCommand) -> int:
    """
    Reload an evicted object from DRAM into SRAM.

    This is a blocking architectural reload in the first version.
    """
    tile, banks = self._choose_reload_location(obj, user_cmd)

    ok = self.mem_mgr.allocate(
        obj.object_id,
        tile,
        banks,
        make_valid=False,
    )

    extra_lat = 0

    if not ok:
        extra_lat += self._allocate_or_spill(obj.object_id, tile, banks)

    # DRAM read latency
    lat = self.dram.read_latency(obj.bytes)

    # Energy and traffic accounting
    self.energy.add_dram_read(obj.bytes)
    self.energy.add_noc(obj.bytes)
    n_access = ceil_div(obj.bytes, self.config.sram_pim.word_bytes)
    self.energy.add_sram_write(n_access)

    # State update
    obj.begin_loading()
    obj.commit_load()

    self.mem_mgr.stats["reload_count"] += 1
    self.traffic["dram_read_bytes"] += obj.bytes
    self.traffic["noc_bytes"] += obj.bytes
    self.latency_breakdown["auto_reload_cycles"] += lat

    return extra_lat + lat
```

### reload location 策略

建议第一版简单采用：

```text
优先使用 object 的 historical tile/banks；
如果 object 没有 historical location：
  - 使用 command src/dst 中的 tile；
  - 或使用 default tile 0；
  - banks 根据 object size 和 bank capacity 选择。
```

建议接口：

```python
def _choose_reload_location(
    self,
    obj: MemoryObject,
    cmd: TraceCommand,
) -> tuple[int, list[int]]:
    ...
```

### 测试命令

```bash
pytest tests/test_auto_reload.py -q
```

验收标准：

```text
strict mode:
  evicted input 被 PIM 使用时 raise

auto_reload mode:
  evicted input valid_in_dram=True 时自动 reload
  reload_count >= 1
  DRAM read bytes 增加
  SRAM write energy 增加
  PIM command latency 包含 reload latency

auto_reload mode:
  evicted input valid_in_dram=False 时 raise
```

---

## P0-C. `final_dirty_policy` 已配置，但缺少仿真结束检查

### 问题描述

配置中已经加入：

```python
final_dirty_policy: str = "report"  # error | auto_writeback | report | ignore
persistent_object_types: [...]
```

但如果 simulator 结束时没有扫描 dirty object，就会出现：

```text
output/state/meta 仍 dirty in SRAM
trace 没有 DMA_STORE
simulator 仍输出 valid result
```

这不符合严格 SRAM-PIM 仿真。

### 修改目标

在 `run()` 结束、生成 report 前调用：

```python
self.finalization_report = self._finalize_dirty_objects()
```

### 建议实现

```python
def _finalize_dirty_objects(self) -> dict:
    policy = self.config.system.final_dirty_policy
    persistent_types = {
        _obj_type_from_str(t)
        for t in self.config.system.persistent_object_types
    }

    final_dirty = []

    for obj in self.mem_mgr.objects.values():
        if obj.dirty_in_sram and obj.obj_type in persistent_types:
            final_dirty.append(obj.object_id)

    report = {
        "final_dirty_policy": policy,
        "final_dirty_count": len(final_dirty),
        "final_dirty_objects": list(final_dirty),
    }

    if not final_dirty:
        return report

    if policy == "error":
        raise RuntimeError(
            f"Final dirty persistent objects exist: {final_dirty}"
        )

    if policy == "auto_writeback":
        total_lat = 0
        for oid in final_dirty:
            obj = self.mem_mgr.objects[oid]
            total_lat += self._blocking_writeback(obj)

        report["final_auto_writeback_cycles"] = total_lat
        report["final_dirty_after_policy"] = 0
        self.latency_breakdown["final_writeback_cycles"] += total_lat
        return report

    if policy == "report":
        report["final_dirty_after_policy"] = len(final_dirty)
        return report

    if policy == "ignore":
        report["final_dirty_after_policy"] = len(final_dirty)
        return report

    raise ValueError(f"Unknown final_dirty_policy: {policy}")
```

### report 中必须加入

```python
report["finalization"] = self.finalization_report
```

建议还加入顶层字段：

```python
report["valid_simulation"] = (
    self.correctness["error_count"] == 0
    and report["finalization"]["final_dirty_after_policy"] == 0
)
```

或者在 `final_dirty_policy="report"` 时不要将其视作 invalid，但必须明确：

```yaml
valid_simulation: true
final_dirty_warning: true
```

### 新增测试

```python
def test_final_dirty_policy_error_raises():
    """
    output dirty in SRAM but no DMA_STORE.
    final_dirty_policy=error should raise.
    """

def test_final_dirty_policy_report_lists_objects():
    """
    final_dirty_policy=report should not raise,
    but report must contain final_dirty_objects.
    """

def test_final_dirty_policy_auto_writeback_flushes_outputs():
    """
    final_dirty_policy=auto_writeback should perform writeback,
    increase writeback_count and DRAM write bytes.
    """
```

---

## P1-A. `spill_model=event_level` 已配置，但仍应明确 blocking 与 event-level 的边界

### 当前状态

当前 `_allocate_or_spill()` 已具备 victim selection、dirty victim writeback、eviction、retry allocation。

但 `_blocking_writeback()` 是同步写回模型：

```text
计算 writeback latency
更新 DRAM write / NoC / SRAM read energy
直接 writeback_complete
直接释放 victim
```

它不进入 event queue，也不真正占用：

```text
- DMA engine
- DRAM channel
- NoC bandwidth
- SRAM read port
- bank resource
```

### 建议短期策略

短期可以保留 blocking model，但 report 中必须明确：

```yaml
spill_model: blocking
timing_fidelity: architectural_blocking_spill
```

不要把它称作 event-level timing。

### 建议中期策略

实现：

```python
def _writeback_victim(self, victim: MemoryObject) -> int:
    if self.config.system.spill_model == "blocking":
        return self._blocking_writeback(victim)

    if self.config.system.spill_model == "event_level":
        return self._event_level_writeback(victim)

    raise ValueError(...)
```

### event-level writeback 语义

```text
ALLOC fails
  -> select victim
  -> if victim dirty:
       create INTERNAL_DMA_STORE command
       enqueue event
       wait until event complete
  -> evict victim
  -> retry allocation
```

第一版 event-level 可以仍然是 blocking wait，但必须通过统一 DMA issue path，从而占用资源和统计口径一致。

### 建议 internal command

```python
TraceCommand(
    cmd_id=self._new_internal_cmd_id(),
    op=OpCode.DMA_STORE,
    object_id=victim.object_id,
    src=f"SRAM:T{victim.sram_tile}:B{...}",
    dst=f"DRAM:{victim.object_id}",
    bytes=victim.bytes,
    attrs={"internal": True, "reason": "spill_writeback"},
)
```

---

## P1-B. report provenance 必须更明确

当前 simulator 如果用于论文实验，report 必须明确：

```yaml
simulator:
  correctness_mode: strict
  final_dirty_policy: error/report/auto_writeback
  spill_model: blocking/event_level
  valid_simulation: true/false

sram:
  source: analytical/destiny
  read_pj_per_access: ...
  write_pj_per_access: ...
  leakage_mw_per_bank: ...
  destiny_binary: ...
  destiny_config: ...
  destiny_parse_success: true/false

dram:
  model: analytical/trace/dramsim3/ramulator
  fixed_latency_cycles: ...
  bandwidth_bytes_per_cycle: ...

pim:
  mode: digital_near_sram/sram_cim
  mac_energy_source: analytical/rtl
  timing_source: analytical/rtl
```

这样可以避免在论文中被质疑：

```text
你的 SRAM 参数到底来自哪里？
DRAM 是不是 cycle accurate？
PIM MAC energy 是 RTL 还是手填？
capacity spill 是不是 event-level？
```

---

## P1-C. DRAM timing 仍应表述为 analytical，除非接入 DRAMSim3/Ramulator

当前 DRAM model 如果仍是：

```text
fixed latency + bandwidth-limited transfer
```

那么论文中应该写：

```text
We use an analytical DRAM backing-store model with fixed access latency and bandwidth-limited transfer time. For sensitivity analysis, DRAM latency and bandwidth are swept.
```

不要写：

```text
cycle-accurate DRAM timing
```

除非真正接入：

```text
DRAMSim3 / Ramulator
```

并将 DMA request trace 回填到 SRAM-PIM event queue。

---

## P1-D. PIM energy 仍应表述为 digital near-SRAM PIM

如果当前配置仍是：

```yaml
pim:
  mode: digital_near_sram
```

那么论文中建议写：

```text
We model a digital near-SRAM PIM architecture where SRAM banks provide resident storage and local digital lanes perform MAC, element-wise, reduction, and nonlinear operations.
```

不要写成：

```text
bitline-level SRAM-CIM
analog SRAM-CIM
ADC/DAC-aware SRAM-CIM
```

除非补齐：

```text
- wordline activation energy
- bitline charge/discharge energy
- sense amplifier energy
- ADC/DAC energy
- bit-serial cycles
- local shift-add energy
```

---

## 4. 推荐立即执行顺序

### Step 1：跑新增 P0 测试

```bash
pytest tests/test_output_allocation_spill.py -q
pytest tests/test_auto_reload.py -q
```

如果失败，不要继续扩展 workload，先修 simulator 主逻辑。

---

### Step 2：修 `_ensure_output_object()`

目标：

```text
PIM output allocation 失败时必须 spill/evict/writeback。
```

验收：

```bash
pytest tests/test_output_allocation_spill.py -q
```

---

### Step 3：修 `_require_inputs_valid()` 的 `auto_reload`

目标：

```text
auto_reload 模式下 evicted but valid_in_dram object 自动 reload。
```

验收：

```bash
pytest tests/test_auto_reload.py -q
```

---

### Step 4：实现 final dirty finalize

目标：

```text
仿真结束时不能静默遗留 dirty persistent output/state。
```

验收：

```bash
pytest tests/test_final_dirty_policy.py -q
```

---

### Step 5：补充 report provenance

目标：

```text
所有实验结果都能说明参数来源和 timing/energy fidelity。
```

建议在所有 JSON/YAML report 中增加：

```yaml
provenance:
  correctness_mode: strict
  final_dirty_policy: report
  spill_model: blocking
  sram_param_source: analytical
  dram_model: analytical
  pim_energy_source: analytical
```

---

## 5. 最小验收测试集合

建议 CI 至少包含：

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

如果这些测试全部通过，才建议继续跑 workload benchmark。

---

## 6. 修完后的预期状态

完成本文 P0 项后，框架可以升级为：

```text
严格 SRAM-PIM architectural simulator prototype
```

其可信能力包括：

```text
- SRAM_ALLOC 与 valid 状态分离；
- DMA_LOAD 完成后才 valid；
- PIM 输入必须 resident 或 auto_reload；
- output allocation 必须受 SRAM 容量约束；
- dirty victim 必须写回；
- final dirty object 有明确 policy；
- capacity spill 有统计；
- report 中明确 timing/energy provenance。
```

仍然不能声称：

```text
- cycle-accurate DRAM；
- circuit-level SRAM-CIM；
- ADC/DAC-aware CIM energy；
- full transformer/SSM operator-accurate trace；
```

除非后续接入 DRAMSim3/Ramulator、RTL/DESTINY 校准、以及更精细的 workload trace generator。

---

## 7. 最终建议

当前最重要的不是扩展更多模型，而是先让实现和测试语义完全一致。

最优先修改顺序：

```text
1. `_ensure_output_object()` allocation failure -> `_allocate_or_spill()`
2. `_require_inputs_valid()` 实现 `auto_reload`
3. `run()` 结束时加入 `_finalize_dirty_objects()`
4. report 增加 correctness/provenance/finalization 字段
5. 将 blocking spill 明确标记为 architectural blocking model
6. 后续再做 event-level spill 和 DRAMSim3/Ramulator 闭环
```

完成前三项后，当前 SRAM-PIM simulator 才能比较稳地进入“论文级 architectural simulator 原型”的状态。
