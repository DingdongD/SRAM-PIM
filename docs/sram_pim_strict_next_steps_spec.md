# SRAM-PIM `fix/strict-simulator` 下一阶段严格修改指南

> 适用仓库：`https://github.com/DingdongD/SRAM-PIM/tree/fix/strict-simulator`  
> 目标版本：从“严格数据生命周期原型”升级为“可用于论文级架构探索的严格 SRAM-PIM simulator”。  
> 文档目标：给出下一步修改的**问题清单、设计原则、接口变化、伪代码、测试用例、验收标准和论文表述边界**。

---

## 0. 当前版本定位

当前 `fix/strict-simulator` 分支已经解决了上一版最关键的若干 P0 问题，包括：

1. `SRAM_ALLOC` 不再等价于 `valid_in_sram`；
2. `DMA_LOAD` 完成后才 `commit_load()`；
3. PIM 输入如果不在 SRAM，strict mode 下会报错；
4. dirty object 不能直接 `SRAM_FREE` 或 `power_gate`；
5. GEMM 的 partial-sum 写入依赖已经通过 `last_y_writer` 串起来；
6. 有 `MemoryObject` 生命周期状态、`MemoryManager` 容量管理、`ResourceModel` bank conflict、`EnergyModel` 能耗分项、`DESTINY` adapter 和 DRAM analytical model。

因此当前版本已经不是“简单 MAC 计数器”，而是具备下述基本形态：

```text
TraceCommand stream
  -> dependency scheduler
  -> resource/bank conflict check
  -> issue event
  -> completion commit state
  -> energy/traffic/statistics report
```

但是它还不能完全称为论文级严格 simulator。下一阶段需要优先解决 4 个问题：

```text
P0-A: PIM output allocation 失败路径没有 spill/evict/writeback 闭环
P0-B: 仿真结束时没有 final dirty object 检查
P0-C: auto_reload 配置存在但语义未实现
P1-A: capacity spill/writeback 目前是 blocking analytical，不是 event-level DMA
```

这四个问题会直接影响大 workload、SRAM 容量压力、长序列 attention、Mamba state、Point-cloud intermediate buffer 的仿真可信度。

---

## 1. 下一阶段必须保持的核心不变量

后续修改必须始终满足下面 10 条 invariant。所有测试和代码 review 都应该围绕这些 invariant 展开。

### I1. SRAM allocation 不代表数据有效

```text
SRAM_ALLOC / mem_mgr.allocate()
  只能表示 SRAM capacity 和 bank placement 已经预留。
  不能表示数据内容已经可读。
```

只有以下事件完成后，数据才能变成 `valid_in_sram=True`：

```text
DMA_LOAD complete
PIM compute complete
preloaded/warm object 显式声明
internal auto_reload complete
```

### I2. PIM 不能读取不在 SRAM 的数据

```text
PIM_MAC / PIM_EW_OP / PIM_REDUCE / PIM_NL
  所有 input object 必须满足：
    object exists
    valid_in_sram == True
    power_state == active
```

strict mode 下，任何不满足都必须停止仿真或触发 auto_reload；不能只计数后继续执行。

### I3. PIM output 只有完成后才有效

```text
PIM issue:
  output.state = PRODUCING
  output.valid_in_sram = False

PIM complete:
  output.state = VALID_DIRTY
  output.valid_in_sram = True
  output.dirty_in_sram = True
  output.valid_in_dram = False 或保持旧版本无效
```

### I4. dirty SRAM data 不能无声丢失

以下情况必须检查 dirty：

```text
SRAM_FREE
capacity eviction
power_gate
simulation end
forced replacement
```

如果 dirty object 没有写回 DRAM，strict mode 下必须报错；非 strict mode 下可以自动 writeback，但必须统计 latency/energy/traffic。

### I5. SRAM capacity 是硬约束

不能出现：

```text
allocate failed 但 simulator 继续执行
output object 没有实际 SRAM placement 却被 commit_produce
capacity overcommit 只作为 warning
```

strict mode 下，如果容量不足且找不到 victim，必须报错。

### I6. DRAM 是 backing store

所有 cold-start 权重、输入、LUT、常量、KV-cache 初始内容等，如果要进入 SRAM-PIM，必须显式经过：

```text
DRAM -> DMA_LOAD -> SRAM valid
```

所有需要跨 kernel、跨 layer、host 可见或最终输出的 dirty object，必须经过：

```text
SRAM dirty -> DMA_STORE/writeback -> DRAM valid
```

### I7. spill/writeback 必须消耗资源

至少需要统计：

```text
DRAM write bytes
NoC bytes
SRAM read accesses
writeback latency
spill count
victim id
```

更严格的 event-level spill 还必须占用：

```text
DMA engine
DRAM channel/bandwidth
NoC bandwidth
SRAM read port / banks
```

### I8. dependency 不能被绕过

任何内部插入的 writeback、reload、evict、retry allocation 都必须与原始 command 建立依赖关系。不能在原 command 尚未满足资源和数据条件时提前完成。

### I9. 报告必须能区分模型来源

报告必须显式记录：

```text
sram_param_source: analytical | destiny | cacti | manual
pim_energy_source: analytical | rtl | measured | manual
DRAM_model: analytical | dramsim3 | ramulator | trace_only
spill_model: blocking | event_level
correctness_mode: strict | warn | auto_reload
```

### I10. workload trace 的语义边界必须明确

如果 attention、SSM、Transformer trace 只是架构 placeholder，报告中必须注明：

```text
workload_trace_semantics: operator-level approximation
```

不能把简化 trace 的结果表述为真实模型端到端精确性能。

---

# 2. 修改总路线图

建议按下面 4 个 phase 执行。

```text
Phase 1: 修复剩余 P0 correctness 问题
  1. PIM output allocation spill 闭环
  2. final dirty object 检查
  3. auto_reload 实现或删除
  4. output object 类型/bytes/precision 严格化

Phase 2: 提升 capacity spill / writeback timing 严格性
  1. blocking spill 保留为快速模式
  2. 新增 event-level spill 模式
  3. 内部 DMA_STORE / EVICT / RETRY_ALLOC 进入 event queue

Phase 3: 提升 timing/energy 参数可信度
  1. DESTINY 参数强制 provenance
  2. DRAMSim3/Ramulator DMA trace 闭环
  3. PIM dataflow energy 更细化
  4. 区分 digital near-SRAM PIM 与 bitline SRAM-CIM

Phase 4: workload trace 和测试体系完善
  1. GEMM psum / output / accumulate 语义完善
  2. Attention softmax/KV cache/prefill-decode 区分
  3. SSM state residency / recurrence / selective scan 细化
  4. CI 测试覆盖所有 invariant
```

---

# 3. Phase 1：P0 correctness 修复

---

## P0-A. 修复 PIM output allocation 失败路径

### 3.1 问题描述

当前 `_ensure_output_object()` 会为 PIM 输出创建 object，并根据 `cmd.dst` 调用 `mem_mgr.allocate()`。但是它没有检查 `allocate()` 返回值，也没有在失败时调用 `_allocate_or_spill()`。

这会造成隐藏错误：

```text
SRAM 容量不足
  -> PIM output allocate 失败
  -> simulator 没有报错，也没有 spill
  -> output 仍然 begin_producing / commit_produce
  -> 报告中看起来计算成功
```

这等价于重新引入“输出 SRAM 无限大”的隐式假设。

### 3.2 修改目标

将 `_ensure_output_object()` 改为：

```text
1. 创建 output object，但不设 valid；
2. 如果 output 尚无 SRAM placement，则解析 cmd.dst；
3. 调用 mem_mgr.allocate(..., make_valid=False)；
4. 若失败，调用 _allocate_or_spill()；
5. 返回 output allocation / spill 额外 latency；
6. PIM command latency 必须包含这部分额外 latency；
7. 如果 strict mode 下仍分配失败，必须 raise RuntimeError。
```

### 3.3 涉及文件

```text
src/simulator.py
src/memory_manager.py  # 若需要暴露更清晰的 capacity API
src/memory_object.py   # 若需要 output state 检查
```

### 3.4 建议接口变化

当前：

```python
self._ensure_output_object(cmd, ObjType.PSUM, "int32")
```

建议改为：

```python
alloc_extra_cycles = self._ensure_output_object(cmd, ObjType.PSUM, "int32")
lat += alloc_extra_cycles
```

函数签名：

```python
def _ensure_output_object(
    self,
    cmd: TraceCommand,
    default_type: ObjType,
    default_precision: str,
) -> int:
    """
    Ensure output object exists and has reserved SRAM placement.
    Return extra latency caused by spill/writeback during allocation.
    Does not make output valid.
    """
```

### 3.5 推荐伪代码

```python
def _ensure_output_object(self, cmd, default_type, default_precision) -> int:
    extra_lat = 0

    # 1. Resolve object metadata
    out_type = _obj_type_from_str(cmd.attrs.get("out_type", default_type.name))
    out_precision = cmd.attrs.get("out_precision", default_precision)
    out_bytes = cmd.attrs.get("out_bytes", cmd.bytes or 1024)

    # 2. Create object if not exists
    if cmd.object_id not in self.mem_mgr.objects:
        out_obj = MemoryObject(
            object_id=cmd.object_id,
            obj_type=out_type,
            bytes=out_bytes,
            precision=out_precision,
            valid_in_dram=False,
            valid_in_sram=False,
        )
        self.mem_mgr.register_object(out_obj)
    else:
        out_obj = self.mem_mgr.objects[cmd.object_id]
        # Optional strict consistency check
        if out_obj.bytes != out_bytes:
            if self.strict:
                raise RuntimeError(
                    f"Output object {cmd.object_id} size mismatch: "
                    f"existing={out_obj.bytes}, cmd={out_bytes}"
                )

    # 3. If no SRAM placement, allocate it
    obj = self.mem_mgr.objects[cmd.object_id]
    if obj.sram_tile < 0:
        if not (cmd.dst and cmd.dst.startswith("SRAM:")):
            if self.strict:
                raise RuntimeError(
                    f"PIM output {cmd.object_id} has no SRAM dst"
                )
            return 0

        loc = parse_sram_loc(cmd.dst)
        ok = self.mem_mgr.allocate(
            cmd.object_id,
            loc.tile,
            list(loc.banks),
            make_valid=False,
        )
        if not ok:
            extra_lat += self._allocate_or_spill(
                cmd.object_id,
                loc.tile,
                list(loc.banks),
            )

    # 4. Final placement check
    obj = self.mem_mgr.objects[cmd.object_id]
    if obj.sram_tile < 0:
        if self.strict:
            raise RuntimeError(
                f"PIM output {cmd.object_id} allocation failed"
            )

    return extra_lat
```

### 3.6 修改 PIM issue 路径

`_issue_pim_mac()`：

```python
alloc_lat = self._ensure_output_object(cmd, ObjType.PSUM, "int32")
lat += alloc_lat
obj = self.mem_mgr.objects[cmd.object_id]
obj.begin_producing()
```

`_issue_pim_ew()`：

```python
alloc_lat = self._ensure_output_object(cmd, ObjType.ACTIVATION, "int8")
lat += alloc_lat
```

`_issue_pim_reduce()`：

```python
alloc_lat = self._ensure_output_object(cmd, ObjType.PSUM, "int32")
lat += alloc_lat
```

`_issue_pim_nl()`：

```python
alloc_lat = self._ensure_output_object(cmd, ObjType.ACTIVATION, "int8")
lat += alloc_lat
```

### 3.7 必须新增测试

新增文件：

```text
tests/test_output_allocation_spill.py
```

测试 1：output allocation 失败时必须 spill victim。

```python
def test_pim_output_allocation_triggers_spill():
    # SRAM capacity deliberately small
    # Load one clean victim + one dirty victim
    # PIM output needs capacity
    # Expect spill/eviction count > 0
    # Expect output valid after completion
```

测试 2：dirty victim 必须 writeback。

```python
def test_pim_output_allocation_dirty_victim_writeback():
    # victim.dirty_in_sram = True
    # output allocation causes eviction
    # Expect dram_write_bytes > 0
    # Expect writeback_count == 1
```

测试 3：没有 victim 时 strict mode 必须报错。

```python
def test_output_allocation_no_victim_raises():
    # all objects pinned or no enough space
    # PIM output allocation fails
    # Expect RuntimeError
```

### 3.8 验收标准

```text
[ ] _ensure_output_object() 返回 int extra latency
[ ] 所有 PIM issue 函数将 alloc extra latency 加入 command latency
[ ] output allocate failed 时 strict mode 不允许继续
[ ] output allocate failed 时可触发 spill/writeback/evict
[ ] 新增测试全部通过
[ ] report 中 spill_count / writeback_count / dram_write_bytes 正确变化
```

---

## P0-B. 增加 simulation end final dirty object 检查

### 3.9 问题描述

当前版本在 `SRAM_FREE` 和 `POWER_SET` 时会保护 dirty object，但如果 trace 结束时还有 dirty output/state 留在 SRAM 中，且没有显式 `DMA_STORE`，仿真报告可能仍然显示 `valid_simulation=True`。

这会让错误 trace 漏检：

```text
PIM_MAC -> output dirty in SRAM
trace ends
没有 DMA_STORE
report valid_simulation=True
```

但从系统角度看，如果 output 是最终输出或需要被下一阶段 host/DRAM 消费，那么这条 trace 不完整。

### 3.10 修改目标

在仿真结束阶段增加 final dirty policy。

新增配置：

```python
@dataclass
class SystemConfig:
    final_dirty_policy: str = "report"  # error | auto_writeback | report | ignore
    persistent_object_types: list[str] = field(
        default_factory=lambda: ["OUTPUT", "STATE", "META", "DIST"]
    )
```

语义：

```text
error:
  如果 persistent dirty object 存在，strict mode 下 raise RuntimeError

auto_writeback:
  仿真结束时自动 writeback 所有 persistent dirty object
  必须统计 latency/energy/traffic

report:
  不报错，但在 report.final_dirty_objects 中列出
  valid_simulation 建议为 False 或 valid_but_dirty=False

ignore:
  完全忽略，仅用于 debug，不建议论文实验使用
```

推荐默认：

```text
strict mode: error
non-strict mode: report
benchmark mode: auto_writeback
```

### 3.11 涉及文件

```text
src/config.py
src/simulator.py
configs/system.yaml 或 configs/sram_pim.yaml
```

### 3.12 推荐伪代码

在 `run()` 结束前调用：

```python
self._finalize_simulation()
return self._make_report()
```

实现：

```python
def _is_persistent_object(self, obj: MemoryObject) -> bool:
    persistent = set(self.config.system.persistent_object_types)
    return obj.obj_type.name in persistent or bool(getattr(obj, "pinned", False))


def _finalize_simulation(self) -> None:
    self.final_dirty_objects = []
    self.final_resident_objects = []

    for oid, obj in self.mem_mgr.objects.items():
        if obj.valid_in_sram:
            self.final_resident_objects.append({
                "object_id": oid,
                "type": obj.obj_type.name,
                "dirty": obj.dirty_in_sram,
                "tile": obj.sram_tile,
                "banks": obj.sram_banks,
            })

        if obj.dirty_in_sram and self._is_persistent_object(obj):
            self.final_dirty_objects.append({
                "object_id": oid,
                "type": obj.obj_type.name,
                "bytes": obj.bytes,
            })

    if not self.final_dirty_objects:
        return

    policy = self.config.system.final_dirty_policy

    if policy == "error":
        self.correctness["final_dirty_objects"] = len(self.final_dirty_objects)
        if self.strict:
            raise RuntimeError(
                f"Trace ended with dirty persistent objects: "
                f"{self.final_dirty_objects}"
            )

    elif policy == "auto_writeback":
        for item in self.final_dirty_objects:
            obj = self.mem_mgr.objects[item["object_id"]]
            lat = self._blocking_writeback(obj)
            self.latency_breakdown["final_writeback_cycles"] = \
                self.latency_breakdown.get("final_writeback_cycles", 0) + lat
            self.cycle += lat

    elif policy == "report":
        self.correctness["final_dirty_objects"] = len(self.final_dirty_objects)

    elif policy == "ignore":
        pass
```

### 3.13 修改 report

`_make_report()` 中新增：

```python
"final_state": {
    "final_dirty_objects": self.final_dirty_objects,
    "final_resident_objects": self.final_resident_objects,
    "final_dirty_policy": self.config.system.final_dirty_policy,
},
"model_provenance": {
    "correctness_mode": self.config.system.correctness_mode,
    "spill_model": self.config.system.spill_model,
    "dram_model": self.config.dram.model,
    "sram_param_source": self.config.energy.sram.source,
},
```

### 3.14 必须新增测试

新增文件：

```text
tests/test_final_dirty_policy.py
```

测试 1：strict + error policy 下 dirty output trace 必须失败。

```python
def test_final_dirty_output_error_policy_raises():
    # trace: alloc/load/PIM output, no DMA_STORE
    # final_dirty_policy = error
    # expect RuntimeError
```

测试 2：auto_writeback 会增加 DRAM write traffic。

```python
def test_final_dirty_auto_writeback_counts_energy():
    # trace leaves dirty output
    # final_dirty_policy = auto_writeback
    # expect dram_write_bytes == output bytes
    # expect final_dirty_objects empty or written_back list populated
```

测试 3：report policy 不报错但 valid_simulation 应显示 dirty 状态。

```python
def test_final_dirty_report_policy_records_objects():
    # final_dirty_policy = report
    # expect report["final_state"]["final_dirty_objects"] not empty
```

### 3.15 验收标准

```text
[ ] config 中有 final_dirty_policy
[ ] report 中有 final_dirty_objects 和 final_resident_objects
[ ] strict + error policy 能抓住缺 DMA_STORE 的 trace
[ ] auto_writeback policy 会统计 DRAM/SRAM/NoC energy 和 latency
[ ] final dirty 不再被 silent pass
```

---

## P0-C. 实现或删除 `auto_reload`

### 3.16 问题描述

当前 `correctness_mode` 中声明了：

```text
strict | warn | auto_reload
```

但是 simulator 里主要只区分 `strict` 与非 strict，没有真正实现：

```text
input evicted from SRAM but valid in DRAM
  -> auto DMA_LOAD
  -> wait reload complete
  -> continue PIM
```

如果保留未实现的 `auto_reload`，用户会误以为 simulator 支持 cache-like SRAM backing reload。

### 3.17 两种可选路线

#### 路线 A：删除 auto_reload

如果短期目标是严格可控，建议只保留：

```text
correctness_mode = strict | warn
```

这样最简单、最干净。

#### 路线 B：实现 auto_reload

如果目标是模拟 SRAM scratchpad / cache-like 混合行为，则实现：

```text
PIM input 不在 SRAM:
  if correctness_mode == strict:
      raise
  elif correctness_mode == warn:
      counter += 1, continue or invalid
  elif correctness_mode == auto_reload:
      if obj.valid_in_dram:
          allocate_or_spill obj
          issue/reload DMA_LOAD
          add reload latency
          continue PIM
      else:
          raise
```

推荐长期采用路线 B，但要明确 `auto_reload` 是 cache-like 行为，和严格 scratchpad trace 行为不同。

### 3.18 推荐设计

将 `_require_inputs_valid()` 从“只检查/raise”改成“返回额外 reload latency”。

当前：

```python
def _require_inputs_valid(self, cmd) -> None:
    ...
```

建议：

```python
def _ensure_inputs_ready(self, cmd) -> int:
    """
    Ensure PIM inputs are SRAM-resident and active.
    Return extra latency caused by auto reload.
    In strict mode, raise on missing/unresident/power-gated inputs.
    """
```

### 3.19 推荐伪代码

```python
def _ensure_inputs_ready(self, cmd: TraceCommand) -> int:
    extra_lat = 0
    mode = self.config.system.correctness_mode

    for iid in parse_src_ids(cmd.src):
        obj = self.mem_mgr.objects.get(iid)

        if obj is None:
            self.correctness["missing_input_object"] += 1
            raise RuntimeError(f"Missing input object {iid}")

        if obj.power_state != "active":
            self.correctness["illegal_read_power_gated_object"] += 1
            if mode in {"strict", "auto_reload"}:
                raise RuntimeError(f"Input {iid} is power-gated")
            continue

        if obj.valid_in_sram:
            continue

        self.correctness["illegal_read_unresident_object"] += 1

        if mode == "strict":
            raise RuntimeError(
                f"Input {iid} not resident in SRAM; "
                f"state={obj.state.value}"
            )

        if mode == "warn":
            continue

        if mode == "auto_reload":
            if not obj.valid_in_dram:
                raise RuntimeError(
                    f"Input {iid} not in SRAM and DRAM copy invalid"
                )
            extra_lat += self._blocking_reload(obj)

    return extra_lat
```

新增：

```python
def _blocking_reload(self, obj: MemoryObject) -> int:
    """Synchronously reload a clean object from DRAM into SRAM."""
    if obj.sram_tile < 0:
        # Need a placement policy
        tile, banks = self._choose_reload_location(obj)
        ok = self.mem_mgr.allocate(obj.object_id, tile, banks, make_valid=False)
        if not ok:
            extra = self._allocate_or_spill(obj.object_id, tile, banks)
        else:
            extra = 0
    else:
        extra = 0

    obj.begin_loading()
    lat = self.dram.get_read_latency(obj.bytes)
    self.energy.add_dram_read(obj.bytes)
    self.energy.add_noc(obj.bytes)
    n_accesses = ceil_div(obj.bytes, self.config.sram_pim.word_bytes)
    self.energy.add_sram_write(n_accesses)
    obj.commit_load()
    self.mem_mgr.stats["reload_count"] += 1
    return extra + lat
```

注意：`_choose_reload_location()` 必须有明确策略。例如：

```text
1. 使用 obj.last_sram_tile / last_sram_banks；
2. 使用 cmd.attrs["reload_dst"]；
3. 使用简单 first-fit；
4. 如果没有任何策略，auto_reload 不允许启用。
```

### 3.20 修改 PIM issue 路径

```python
reload_lat = self._ensure_inputs_ready(cmd)
lat += reload_lat
```

适用于：

```text
_issue_pim_mac
_issue_pim_ew
_issue_pim_reduce
_issue_pim_nl
```

### 3.21 必须新增测试

新增文件：

```text
tests/test_auto_reload.py
```

测试 1：strict mode 下 unresident input 仍然报错。

```python
def test_strict_unresident_input_raises():
    ...
```

测试 2：auto_reload 会触发 DRAM read。

```python
def test_auto_reload_reads_from_dram():
    # Object valid_in_dram=True, valid_in_sram=False
    # PIM uses object
    # Expect reload_count == 1
    # Expect dram_read_bytes increased
```

测试 3：DRAM copy invalid 时 auto_reload 也必须报错。

```python
def test_auto_reload_invalid_dram_copy_raises():
    # valid_in_dram=False, valid_in_sram=False
    # PIM uses object
    # Expect RuntimeError
```

### 3.22 验收标准

```text
[ ] correctness_mode 文档与实现一致
[ ] auto_reload 若保留，必须真的 reload
[ ] reload 计入 latency/energy/traffic
[ ] reload 使用明确 placement policy
[ ] tests/test_auto_reload.py 全部通过
```

---

## P0-D. 严格化 output object 类型、bytes、precision

### 3.23 问题描述

当前多个 PIM op 默认使用：

```text
ObjType.PSUM
precision = int32
out_bytes = cmd.attrs.get("out_bytes", cmd.bytes or 1024)
```

这对 GEMM psum 合理，但对其他算子不一定合理：

```text
PIM_EW_OP 可能输出 activation/state
PIM_REDUCE 可能输出 scalar/meta/dist
PIM_NL 可能输出 activation/softmax score
PIM_MAC final output 可能是 OUTPUT 而不是 PSUM
```

如果 output object 类型和精度错误，会影响：

```text
final dirty policy
capacity accounting
energy accounting
writeback policy
trace validation
```

### 3.24 修改目标

所有 PIM trace command 必须显式携带 output metadata：

```python
attrs = {
    "out_type": "PSUM" | "ACTIVATION" | "STATE" | "OUTPUT" | "META" | "DIST",
    "out_precision": "int8" | "int16" | "int32" | "fp16" | "fp32",
    "out_bytes": ...,
    "accumulate": True | False,
}
```

如果缺失，则由 opcode 给出安全默认值，但 validator 应给 warning。

### 3.25 修改位置

```text
src/tracegen/gen_gemm_trace.py
src/tracegen/gen_attention_trace.py
src/tracegen/gen_ssm_trace.py
src/tracegen/gen_transformer_trace.py
src/simulator.py
src/trace_validator.py
```

### 3.26 推荐默认规则

| Op | 默认 out_type | 默认 precision | 说明 |
|---|---|---|---|
| PIM_MAC with accumulate=True | PSUM | int32 | tiled GEMM 中间累加 |
| PIM_MAC final | OUTPUT/ACTIVATION | int8/int16/fp16 | 由 trace 指定 |
| PIM_EW_OP | ACTIVATION 或 STATE | 输入精度 | element-wise 输出 |
| PIM_REDUCE | META/PSUM/DIST | int32/fp16 | 取决于 reduce 类型 |
| PIM_NL | ACTIVATION | 输入精度 | softmax/activation 输出 |

### 3.27 测试

新增：

```text
tests/test_output_metadata.py
```

测试内容：

```text
[ ] PIM_MAC accumulate=True 输出 PSUM/int32
[ ] PIM_EW_OP 输出 STATE 时 final dirty policy 能识别 STATE
[ ] PIM_NL 输出 ACTIVATION 不被误当作 PSUM
[ ] 缺 out_bytes 时 validator warning 或 strict error
```

---

# 4. Phase 2：event-level spill/writeback

---

## P1-A. 将 blocking spill 升级为 event-level spill

### 4.1 当前 blocking spill 的问题

当前 `_allocate_or_spill()` 中 dirty victim 会调用 `_blocking_writeback()`。这会同步增加：

```text
DRAM write latency
DRAM write energy
NoC energy
SRAM read energy
```

但它没有真正占用：

```text
DMA engine
DRAM queue/channel
NoC bandwidth
SRAM read bank/resource
```

因此当 spill 与正常 DMA/PIM 并发时，当前模型会偏乐观。

### 4.2 修改目标

新增 `spill_model` 配置：

```python
@dataclass
class SystemConfig:
    spill_model: str = "blocking"  # blocking | event_level
```

语义：

```text
blocking:
  保留当前同步 writeback，适合快速探索。

event_level:
  spill/writeback 被转成 internal DMA_STORE event，进入 event queue 和 ResourceModel。
```

### 4.3 event-level 设计思路

当某 command 需要分配 output/input/reload 空间，但容量不足时：

```text
original command C
  -> allocation fails
  -> select victim V
  -> if V dirty:
       issue INTERNAL_DMA_STORE(V)
       original C waits
  -> INTERNAL_DMA_STORE complete:
       V.writeback_complete()
       evict V
       retry allocation for C
  -> allocation success:
       issue original C
```

### 4.4 推荐最小实现方式

不要一开始重构完整 scheduler。可以先实现一个“stall and retry”机制。

新增结构：

```python
@dataclass
class PendingAllocation:
    cmd_id: int
    object_id: str
    tile: int
    banks: list[int]
    victim_ids: list[str]
    state: str  # waiting_writeback | ready_retry
```

在 `Simulator` 中新增：

```python
self.pending_allocations: dict[int, PendingAllocation] = {}
self.internal_cmd_counter = -1
```

当 allocation 失败：

```python
if self.config.system.spill_model == "blocking":
    return self._allocate_or_spill_blocking(...)
else:
    return self._allocate_or_spill_event_level(...)
```

### 4.5 event-level 伪代码

```python
def _allocate_or_spill_event_level(self, object_id, tile, banks, parent_cmd):
    needed = self._estimate_needed_bytes(object_id, tile)
    victims = self.mem_mgr.find_eviction_candidate(tile, needed)

    if not victims:
        raise RuntimeError("No eviction candidate")

    dirty_victims = [v for v in victims if self.mem_mgr.objects[v].dirty_in_sram]

    if not dirty_victims:
        for vid in victims:
            self._evict_clean_object(vid)
        ok = self.mem_mgr.allocate(object_id, tile, banks, make_valid=False)
        if not ok:
            raise RuntimeError("Allocation still fails")
        return 0

    # Dirty victims: create internal DMA_STORE commands
    for vid in dirty_victims:
        victim = self.mem_mgr.objects[vid]
        internal = TraceCommand(
            cmd_id=self._next_internal_cmd_id(),
            op=OpCode.DMA_STORE,
            object_id=vid,
            src=f"SRAM:T{victim.sram_tile}:B{','.join(map(str, victim.sram_banks))}",
            dst="DRAM",
            bytes=victim.bytes,
            attrs={"internal": True, "evict_after": True},
            deps=[],
        )
        self._issue_internal_command(internal)

    # Parent cmd is not issued now; it will retry after internal cmds complete
    self.pending_allocations[parent_cmd.cmd_id] = PendingAllocation(...)
    return None  # Means parent command not issued yet
```

Scheduler 需要识别：

```text
None latency -> command remains pending, not completed
```

或者更简单：在正式重构前，先不要让 `_ensure_output_object()` 在 issue 阶段做 event-level spill，而是新增显式 `SRAM_ALLOC` trace，让 spill 在 allocate command 处完成。这个方案侵入性更小。

### 4.6 推荐分两步实现

#### Step 1：保持 blocking，但报告 spill_model

短期先把 blocking 模型标清楚：

```yaml
model_provenance:
  spill_model: blocking
```

#### Step 2：新增 event-level 实验模式

只有在测试充分后，才把默认切到 event-level。

### 4.7 必须新增测试

```text
tests/test_event_level_spill.py
```

测试：

```text
[ ] dirty victim spill 会生成 internal DMA_STORE
[ ] internal DMA_STORE 占用 DRAM/NoC/SRAM read resource
[ ] original command 等待 spill 完成后再 issue
[ ] blocking 和 event_level 在 traffic/energy 上一致，但 cycles 可能不同
[ ] event_level report 中能看到 internal command count
```

### 4.8 验收标准

```text
[ ] spill_model 可配置
[ ] blocking 模式仍兼容原测试
[ ] event_level 模式不会绕过 ResourceModel
[ ] victim writeback completion 后才能 evict
[ ] original command 不会在 allocation 成功前 issue
```

---

# 5. Phase 3：timing / energy 参数可信度增强

---

## P1-B. DRAMSim3/Ramulator DMA trace 闭环

### 5.1 当前状态

当前 DRAM 主要是：

```text
fixed_latency + ceil(bytes / bandwidth)
```

这适合快速 architecture exploration，但不是严格 DRAM timing。

### 5.2 修改目标

保留 analytical 模式，同时新增外部 DRAM simulator 闭环：

```text
SRAM-PIM simulator
  -> 生成 DMA trace
  -> 调用 DRAMSim3 或 Ramulator
  -> 解析 completion cycles
  -> 回填 DMA latency
```

### 5.3 配置建议

```yaml
dram:
  model: analytical        # analytical | dramsim3 | ramulator | trace_only
  fixed_latency_ns: 50
  bandwidth_GBps: 25.6
  external_binary: ./third_party/DRAMsim3/build/dramsim3main
  config_file: ./configs/dram/DDR4_3200.yaml
  trace_output: ./outputs/dma_trace.txt
  allow_fallback: false
```

### 5.4 Trace 格式建议

```text
cycle,op,addr,bytes,object_id,cmd_id
100,READ,0x10000000,4096,W0,12
240,WRITE,0x20000000,4096,Y0,89
```

如果没有真实地址，可以先使用 object-id based pseudo address allocator：

```python
self.dram_address_map[object_id] = base_addr
base_addr += align(obj.bytes, 64)
```

### 5.5 report provenance

```yaml
model_provenance:
  dram_model: analytical | dramsim3 | ramulator
  dram_config_file: ...
  dram_trace_file: ...
  dram_fallback_used: false
```

### 5.6 验收标准

```text
[ ] analytical 模式可快速运行
[ ] external 模式能生成 DMA trace
[ ] external 模式能解析 request completion
[ ] external binary 缺失时，如果 allow_fallback=false 必须报错
[ ] report 明确记录 DRAM 模型来源
```

---

## P1-C. DESTINY 参数接入与 provenance 强化

### 5.7 当前状态

当前有 `DestinyAdapter`，也支持将 SRAM read/write energy 和 leakage 写回 config。但默认仍可以 fallback 到 analytical 参数。

### 5.8 修改目标

让 SRAM 参数来源可审计、可复现。

新增配置：

```yaml
sram_macro:
  source: destiny          # analytical | destiny | cacti | manual
  destiny_binary: ./third_party/DESTINY/destiny
  cache_result: ./outputs/destiny_cache.json
  allow_fallback: false
  tech_node_nm: 28
  capacity_kb_per_bank: 64
  banks_per_tile: 32
```

### 5.9 修改要求

如果用户指定：

```text
source = destiny
allow_fallback = false
```

则 DESTINY binary 缺失或解析失败时必须直接报错，不允许静默使用默认参数。

如果：

```text
allow_fallback = true
```

则 report 必须显示：

```yaml
sram_param_source: analytical_fallback
fallback_reason: destiny_binary_not_found
```

### 5.10 report 字段

```yaml
sram_macro:
  source: destiny
  read_latency_cycles: ...
  write_latency_cycles: ...
  read_energy_pj: ...
  write_energy_pj: ...
  leakage_mw_per_bank: ...
  area_mm2_per_bank: ...
  destiny_config_hash: ...
  fallback_used: false
```

### 5.11 验收标准

```text
[ ] --sram-source destiny 缺 binary 时不会静默 fallback
[ ] cached DESTINY result 可复现加载
[ ] report 中包含所有 SRAM macro 参数
[ ] 所有实验数据能追溯参数来源
```

---

## P1-D. PIM dataflow energy 更细化

### 5.12 当前问题

当前 `_account_pim_data_energy()` 主要根据 input object bytes 估算 SRAM read，并根据 output bytes 估算 SRAM write。这是合理的一阶模型，但对 tiled dataflow 可能过粗：

```text
可能过估：每次 PIM op 读完整 object，而真实只读 tile slice
可能低估：accumulate 时 psum read-modify-write 没有按精度/次数细分
可能混淆：digital near-SRAM PIM 与 bitline SRAM-CIM 能耗
```

### 5.13 修改目标

Trace command 中增加显式 dataflow bytes：

```python
attrs = {
    "act_read_bytes": ...,
    "weight_read_bytes": ...,
    "psum_read_bytes": ...,
    "out_write_bytes": ...,
    "local_reduce_bytes": ...,
    "noc_bytes_override": ...,
}
```

EnergyModel 按这些字段统计，而不是默认读完整 object。

### 5.14 推荐能耗拆分

```text
E_total =
  E_dram_read + E_dram_write
  + E_sram_array_read_act
  + E_sram_array_read_weight
  + E_sram_array_read_psum
  + E_sram_array_write_psum_or_out
  + E_pim_mac_datapath
  + E_pim_reduce
  + E_pim_nonlinear
  + E_noc
  + E_control
  + E_leakage
```

### 5.15 PIM 类型区分

配置中必须明确：

```yaml
pim:
  mode: digital_near_sram  # digital_near_sram | digital_sram_cim | analog_sram_cim
```

如果 `mode=digital_near_sram`：

```text
SRAM read/write energy from DESTINY/analytical
MAC datapath energy from analytical/RTL
```

如果 `mode=digital_sram_cim` 或 `analog_sram_cim`，必须额外提供：

```yaml
cim:
  bit_serial_cycles: ...
  wordline_energy_pj: ...
  bitline_energy_pj: ...
  sense_amp_energy_pj: ...
  adc_energy_pj: ...
  dac_energy_pj: ...
  shift_add_energy_pj: ...
```

否则 strict config validation 应该报错。

### 5.16 验收标准

```text
[ ] PIM data energy 不再只能按完整 object bytes 估计
[ ] report 能显示 SRAM act/weight/psum/out 访问分项
[ ] PIM mode 与 energy formula 一致
[ ] analog/digital SRAM-CIM 缺参数时不能运行 strict experiment
```

---

# 6. Phase 4：workload trace 严格化

---

## P2-A. GEMM trace 增强

### 6.1 当前状态

GEMM 已经有 K tile partial-sum dependency，这是正确方向。

### 6.2 建议增强

每个 `PIM_MAC` 显式写：

```python
attrs = {
    "mac_count": M_tile * N_tile * K_tile,
    "accumulate": ki > 0,
    "out_type": "PSUM" if not final else "OUTPUT",
    "out_precision": "int32",
    "out_bytes": M_tile * N_tile * 4,
    "act_read_bytes": M_tile * K_tile * act_bytes,
    "weight_read_bytes": K_tile * N_tile * weight_bytes,
    "psum_read_bytes": M_tile * N_tile * 4 if ki > 0 else 0,
    "out_write_bytes": M_tile * N_tile * 4,
}
```

最后如果需要 requant：

```text
PIM_REDUCE / PIM_EW_OP / PIM_NL / PIM_WRITEBACK
```

不要直接把 int32 psum 当 int8 output store。

---

## P2-B. Attention trace 增强

### 6.3 当前问题

Attention trace 中 softmax 和 KV-cache 仍然偏粗。

### 6.4 建议拆解

对于 decode attention：

```text
Q resident/load
K/V cache resident check
QK^T score PIM_MAC
causal mask / metadata op
softmax max-reduce
exp approximation / PIM_NL
sum-reduce
division / normalize
context = softmax * V PIM_MAC
output store or remain resident
```

建议新增 opcode 或 attrs：

```text
PIM_REDUCE_MAX
PIM_NL_EXP
PIM_REDUCE_SUM
PIM_EW_DIV
```

如果不想增加 opcode，可以用 `PIM_REDUCE` / `PIM_NL` / `PIM_EW_OP`，但 attrs 必须写清楚：

```python
attrs = {
    "kind": "softmax_max" | "softmax_exp" | "softmax_sum" | "softmax_div",
    "count": ...,
}
```

### 6.5 KV-cache residency

增加：

```text
KV_LOAD_POLICY = cold | warm_resident | sliding_window | spill_reload
```

Trace 必须明确 K/V 是：

```text
preloaded resident
每层从 DRAM load
因 SRAM 容量不足 spill/reload
```

---

## P2-C. SSM / Mamba trace 增强

### 6.6 当前问题

SSM trace 中 `h_t = A*h_{t-1} + B*u_t` 已经有基本形式，但 selective scan 真实逻辑可能还需要：

```text
dt projection
A/B/C dynamic generation
exp(dt*A)
dB*u
gated output
state precision
chunked scan or recurrent scan
```

### 6.7 推荐 trace 语义

对于每个 token 或 chunk：

```text
DMA_LOAD / resident check: u_t, state h_{t-1}, parameters
PIM_EW_OP: dt*A
PIM_NL: exp(dt*A)
PIM_EW_OP: dB*u_t
PIM_EW_OP: h_t = dA*h_{t-1} + dB*u_t
PIM_EW_OP/PIM_MAC: y_t = C*h_t + D*u_t
state h_t dirty in SRAM
optional DMA_STORE state if layer boundary / spill / final
```

### 6.8 state residency policy

新增配置：

```yaml
ssm:
  state_policy: resident | spill_each_layer | auto_reload
  scan_mode: recurrent | chunk_parallel
  state_precision: fp16 | int16 | int8
```

### 6.9 测试

```text
tests/test_ssm_state_lifecycle.py
```

必须覆盖：

```text
[ ] h_{t} 依赖 h_{t-1}
[ ] state dirty 不能 power-gate
[ ] state spill 后 auto_reload 可恢复
[ ] recurrent 与 chunk_parallel cycle 行为不同
```

---

# 7. Trace validator 强化

建议新增或强化 `TraceValidator`，在仿真前检查静态错误。

## 7.1 必查规则

```text
[ ] 每个 object_id 先 register/alloc 再使用
[ ] PIM input 必须有来源：preloaded、DMA_LOAD、previous PIM output
[ ] PIM output 必须有 dst SRAM location
[ ] PIM output 必须有 out_bytes
[ ] DMA_STORE 的 object 必须可能 dirty 或 valid_in_sram
[ ] 所有 deps 指向已存在 cmd_id
[ ] 不允许循环依赖
[ ] 同一 object 多 writer 必须有 dependency ordering
[ ] persistent output 最终必须 DMA_STORE 或 final_dirty_policy != error
```

## 7.2 Validator 输出

```python
@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    object_defs: dict
    writer_map: dict
    reader_map: dict
```

## 7.3 CI 要求

strict experiment 必须满足：

```text
len(validation.errors) == 0
```

---

# 8. Report 格式标准化

建议 `_make_report()` 最终输出采用统一 schema。

```yaml
latency:
  total_cycles: ...
  total_ns: ...
  dram_load_cycles: ...
  dram_store_cycles: ...
  pim_compute_cycles: ...
  pim_reduce_cycles: ...
  stall_dependency_cycles: ...
  bank_conflict_stall_cycles: ...
  spill_writeback_cycles: ...
  final_writeback_cycles: ...

energy:
  total_pj: ...
  dram_read_pj: ...
  dram_write_pj: ...
  sram_read_pj: ...
  sram_write_pj: ...
  pim_mac_pj: ...
  pim_ew_pj: ...
  pim_reduce_pj: ...
  pim_nl_pj: ...
  noc_pj: ...
  leakage_pj: ...

traffic:
  dram_read_bytes: ...
  dram_write_bytes: ...
  sram_read_accesses: ...
  sram_write_accesses: ...
  noc_bytes: ...

memory_lifecycle:
  spill_count: ...
  eviction_count: ...
  writeback_count: ...
  reload_count: ...
  final_dirty_count: ...

final_state:
  final_dirty_policy: error | auto_writeback | report | ignore
  final_dirty_objects: [...]
  final_resident_objects: [...]

correctness:
  valid_simulation: true | false
  illegal_read_unresident_object: ...
  missing_input_object: ...
  illegal_read_power_gated_object: ...
  illegal_evict_dirty_without_writeback: ...
  final_dirty_objects: ...
  capacity_overcommit_events: ...
  dependency_violations: ...
  deadlock_events: ...

model_provenance:
  correctness_mode: strict | warn | auto_reload
  spill_model: blocking | event_level
  dram_model: analytical | dramsim3 | ramulator
  sram_param_source: analytical | destiny | cacti | manual
  pim_energy_source: analytical | rtl | measured | manual
  workload_trace_semantics: operator_level_approximation | exact_operator_trace
```

---

# 9. 新增测试文件清单

建议下一轮至少新增以下测试。

```text
tests/test_output_allocation_spill.py
  - test_pim_output_allocation_triggers_spill
  - test_pim_output_allocation_dirty_victim_writeback
  - test_output_allocation_no_victim_raises


tests/test_final_dirty_policy.py
  - test_final_dirty_output_error_policy_raises
  - test_final_dirty_auto_writeback_counts_energy
  - test_final_dirty_report_policy_records_objects


tests/test_auto_reload.py
  - test_strict_unresident_input_raises
  - test_auto_reload_reads_from_dram
  - test_auto_reload_invalid_dram_copy_raises


tests/test_output_metadata.py
  - test_pim_mac_accumulate_outputs_psum_int32
  - test_pim_ew_outputs_activation_or_state
  - test_missing_out_bytes_validator_warning


tests/test_report_schema.py
  - test_report_contains_model_provenance
  - test_report_contains_final_state
  - test_report_spill_model_recorded


tests/test_event_level_spill.py  # Phase 2
  - test_internal_dma_store_generated_for_dirty_spill
  - test_original_command_waits_for_internal_writeback
  - test_blocking_and_event_level_match_traffic
```

---

# 10. 推荐执行顺序

最推荐的修改顺序如下。

```text
Step 1: 修 P0-A output allocation spill
  因为这是当前最直接的 correctness hole。

Step 2: 加 P0-B final dirty policy
  让 trace 结束状态不再 silent pass。

Step 3: 处理 P0-C auto_reload
  要么完整实现，要么从配置和文档中删除。

Step 4: 加 report schema / model provenance
  让实验结果能说明参数来源和模型边界。

Step 5: 强化 output metadata 和 trace validator
  避免 workload generator 产生模糊语义。

Step 6: 做 event-level spill
  从 blocking architecture model 升级到更严格的 resource contention model。

Step 7: DRAMSim3/Ramulator 和 DESTINY provenance
  用于论文实验可信度增强。

Step 8: Attention / SSM workload trace 精细化
  用于支持真实 Transformer/Mamba/PointMamba 结论。
```

---

# 11. 论文表述建议

在完成 Phase 1 后，可以这样表述：

```text
We implement a trace-driven architectural simulator for a digital near-SRAM PIM system. The simulator explicitly models DRAM-backed SRAM residency, DMA load/store, SRAM valid/dirty states, bank-level resource conflicts, PIM command latency, capacity spill/writeback, and energy breakdown. SRAM allocation is separated from data validity, and strict correctness checks prevent PIM commands from reading non-resident or power-gated objects.
```

不要写：

```text
cycle-accurate DRAM simulator
circuit-level SRAM-CIM simulator
exact Transformer/Mamba execution simulator
```

除非已经完成：

```text
DRAMSim3/Ramulator 闭环
DESTINY/CACTI/RTL 参数校准
event-level spill/writeback
真实 workload trace semantics
```

在完成 Phase 2/3 后，可以增强为：

```text
The simulator supports both analytical and event-level spill models. In event-level mode, dirty victim writebacks are inserted as internal DMA events and contend for SRAM banks, NoC bandwidth, and DRAM bandwidth. SRAM macro parameters can be imported from DESTINY, while DRAM DMA traces can be exported to an external DRAM simulator for timing calibration.
```

---

# 12. 最终验收 checklist

完成下一阶段后，必须满足以下 checklist。

## Correctness

```text
[ ] SRAM_ALLOC 不会让 object valid
[ ] DMA_LOAD complete 后才 valid
[ ] PIM input 不在 SRAM 时 strict mode 报错
[ ] auto_reload 若启用，会真实 reload 并统计 traffic
[ ] PIM output allocation 失败会 spill/evict/writeback 或报错
[ ] PIM output complete 后才 dirty valid
[ ] dirty object 不能 free/power_gate/evict 而不 writeback
[ ] simulation end 能检测 final dirty output/state
[ ] 所有 dependency violation 都会报错或进入 correctness counter
```

## Capacity

```text
[ ] tile capacity 和 bank capacity 都是硬约束
[ ] pinned object 不会被 victim 选择
[ ] dirty victim writeback 后才能 evict
[ ] clean victim 可以直接 evict，但后续读取需要 reload 或报错
[ ] no victim 时 strict mode 报错
```

## Timing

```text
[ ] PIM latency 使用 ceil_div
[ ] DMA latency 使用 ceil_div
[ ] output allocation spill latency 计入 command latency
[ ] bank conflict stall 被统计
[ ] event-level spill 不绕过 ResourceModel
```

## Energy / Traffic

```text
[ ] DMA_LOAD 统计 DRAM read + NoC + SRAM write
[ ] DMA_STORE 统计 SRAM read + NoC + DRAM write
[ ] PIM MAC 统计 SRAM input read + psum RMW + datapath energy
[ ] spill/writeback 统计完整 traffic/energy
[ ] final auto_writeback 统计完整 traffic/energy
[ ] report 中有完整 breakdown
```

## Provenance

```text
[ ] report 中记录 SRAM 参数来源
[ ] report 中记录 PIM 能耗来源
[ ] report 中记录 DRAM 模型来源
[ ] report 中记录 spill model
[ ] report 中记录 workload trace semantics
```

## Testing

```text
[ ] pytest -q 全部通过
[ ] output allocation spill 测试通过
[ ] final dirty policy 测试通过
[ ] auto_reload 测试通过或配置中删除 auto_reload
[ ] report schema 测试通过
[ ] trace validator 测试通过
```

---

# 13. 最小可合并 PR 范围建议

如果希望快速合并下一版，建议先做一个最小 PR：

```text
PR-1: Strict correctness closure
  1. _ensure_output_object() 返回 alloc_extra_cycles
  2. PIM issue latency 加 alloc_extra_cycles
  3. final_dirty_policy + final_state report
  4. 删除或实现 auto_reload
  5. 新增 4 个测试文件：
       test_output_allocation_spill.py
       test_final_dirty_policy.py
       test_auto_reload.py 或删除模式测试
       test_report_schema.py
```

这个 PR 合并后，框架可以更有底气称为：

```text
strict DRAM-backed SRAM-PIM architectural simulator prototype
```

然后再开：

```text
PR-2: Event-level spill and DRAM/Destiny provenance
PR-3: Workload trace semantic refinement
PR-4: External DRAMSim3/Ramulator integration
```

