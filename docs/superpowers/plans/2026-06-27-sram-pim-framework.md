# SRAM-PIM Simulation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trace-driven SRAM-PIM simulator that models DRAM backing store, SRAM residency, DMA traffic, bank conflicts, PIM compute, and energy breakdown for LLM inference.

**Architecture:** Python event-driven simulator with YAML config. Trace generator maps neural network layers to system-level commands (DMA_LOAD, PIM_MAC, etc.). Simulator processes commands cycle-by-cycle checking dependencies, resource availability, and power constraints. Energy model integrates DESTINY SRAM params with analytical PIM/DRAM models.

**Tech Stack:** Python 3.10+, PyYAML, dataclasses, heapq (event queue), pytest, numpy (optional for matrix ops)

## Global Constraints

- Pure Python, no compiled extensions for core simulator
- All configuration via YAML files
- Energy units: pJ; latency units: cycles and ns; capacity: bytes/KB
- All SRAM access must check valid_in_sram before read/PIM
- All dirty evictions must emit DMA_STORE or raise error
- Trace format: text-based, one command per line

---

### Task 1: Project Scaffold and Configuration System

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `configs/system.yaml`
- Create: `configs/dram.yaml`
- Create: `configs/sram_pim.yaml`
- Create: `configs/energy.yaml`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `main.py`
- Create: `requirements.txt`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `SimConfig` dataclass with `system`, `dram`, `sram_pim`, `energy` sub-configs; `load_config(config_dir: str) -> SimConfig`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_config.py
import pytest
from src.config import load_config, SimConfig

def test_load_config_returns_simconfig():
    config = load_config("configs")
    assert isinstance(config, SimConfig)
    assert config.system.frequency_hz == 1_000_000_000
    assert config.system.mode == "warm_resident"

def test_sram_pim_config():
    config = load_config("configs")
    assert config.sram_pim.tiles == 16
    assert config.sram_pim.banks_per_tile == 32
    assert config.sram_pim.bank_capacity_kb == 8
    assert config.sram_pim.total_capacity_kb == 4096

def test_dram_config():
    config = load_config("configs")
    assert config.dram.model == "analytical"
    assert config.dram.effective_bandwidth_gbps == 25.6

def test_energy_config():
    config = load_config("configs")
    assert config.energy.sram.read_pj_per_access == 3.2
    assert config.energy.pim.mac_pj_per_op == 0.08
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sram_pim && python -m pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create requirements.txt**

```
pyyaml>=6.0
pytest>=7.0
numpy>=1.21
```

- [ ] **Step 4: Install dependencies**

Run: `cd /home/sram_pim && pip install -r requirements.txt`

- [ ] **Step 5: Create YAML config files**

```yaml
# configs/system.yaml
system:
  frequency_hz: 1000000000
  mode: warm_resident
  num_queries_for_amortization: 1000
  count_initial_preload: false
  count_final_writeback: true

workload:
  model: GEMM_test
  batch_size: 1
  precision:
    activation: int8
    weight: int8
    psum: int32
```

```yaml
# configs/dram.yaml
dram:
  model: analytical
  type: LPDDR4
  channels: 1
  bus_width_bits: 64
  burst_bytes: 64
  effective_bandwidth_gbps: 25.6
  fixed_latency_ns: 80
  energy:
    read_pj_per_byte: 15.0
    write_pj_per_byte: 18.0
    io_pj_per_byte: 8.0
```

```yaml
# configs/sram_pim.yaml
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
    mode: digital_near_sram
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

```yaml
# configs/energy.yaml
energy:
  sram:
    source: analytical
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

- [ ] **Step 6: Implement config loader**

```python
# src/__init__.py
# empty

# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class PrecisionConfig:
    activation: str = "int8"
    weight: str = "int8"
    psum: str = "int32"


@dataclass
class WorkloadConfig:
    model: str = "GEMM_test"
    batch_size: int = 1
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)


@dataclass
class SystemConfig:
    frequency_hz: int = 1_000_000_000
    mode: str = "warm_resident"
    num_queries_for_amortization: int = 1000
    count_initial_preload: bool = False
    count_final_writeback: bool = True


@dataclass
class DRAMEnergyConfig:
    read_pj_per_byte: float = 15.0
    write_pj_per_byte: float = 18.0
    io_pj_per_byte: float = 8.0


@dataclass
class DRAMConfig:
    model: str = "analytical"
    type: str = "LPDDR4"
    channels: int = 1
    bus_width_bits: int = 64
    burst_bytes: int = 64
    effective_bandwidth_gbps: float = 25.6
    fixed_latency_ns: float = 80.0
    energy: DRAMEnergyConfig = field(default_factory=DRAMEnergyConfig)


@dataclass
class PortModelConfig:
    read_ports_per_bank: int = 1
    write_ports_per_bank: int = 1
    pim_ports_per_bank: int = 1
    pim_exclusive_with_read: bool = True
    pim_exclusive_with_write: bool = True


@dataclass
class PIMHardwareConfig:
    mode: str = "digital_near_sram"
    lanes_per_bank: int = 128
    mac_latency_cycles: int = 2
    mac_issue_interval_cycles: int = 1
    reduce_latency_cycles: int = 4
    nonlinear_latency_cycles: int = 4
    max_parallel_banks: int = 16
    power_throttle: bool = True
    tile_power_budget_mw: float = 100.0


@dataclass
class BufferConfig:
    input_buffer_kb_per_tile: int = 16
    psum_buffer_kb_per_tile: int = 32
    output_buffer_kb_per_tile: int = 16
    double_buffering: bool = True


@dataclass
class PowerStateConfig:
    allow_clock_gating: bool = True
    allow_power_gating: bool = False
    wakeup_latency_cycles: int = 20
    wakeup_energy_pj: float = 500.0


@dataclass
class SRAMPIMConfig:
    tiles: int = 16
    banks_per_tile: int = 32
    bank_capacity_kb: int = 8
    total_capacity_kb: int = 4096
    word_bytes: int = 16
    port_model: PortModelConfig = field(default_factory=PortModelConfig)
    pim: PIMHardwareConfig = field(default_factory=PIMHardwareConfig)
    buffers: BufferConfig = field(default_factory=BufferConfig)
    power_state: PowerStateConfig = field(default_factory=PowerStateConfig)


@dataclass
class SRAMEnergyConfig:
    source: str = "analytical"
    read_pj_per_access: float = 3.2
    write_pj_per_access: float = 3.8
    leakage_mw_per_bank: float = 0.39


@dataclass
class PIMEnergyConfig:
    mac_pj_per_op: float = 0.08
    reduce_pj_per_op: float = 0.04
    nonlinear_pj_per_elem: float = 0.12
    control_pj_per_command: float = 1.0


@dataclass
class NoCEnergyConfig:
    pj_per_byte_per_hop: float = 0.2
    average_hops: int = 2


@dataclass
class DMAEnergyConfig:
    pj_per_byte: float = 0.5


@dataclass
class EnergyConfig:
    sram: SRAMEnergyConfig = field(default_factory=SRAMEnergyConfig)
    pim: PIMEnergyConfig = field(default_factory=PIMEnergyConfig)
    noc: NoCEnergyConfig = field(default_factory=NoCEnergyConfig)
    dma: DMAEnergyConfig = field(default_factory=DMAEnergyConfig)


@dataclass
class SimConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    dram: DRAMConfig = field(default_factory=DRAMConfig)
    sram_pim: SRAMPIMConfig = field(default_factory=SRAMPIMConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)


def _dict_to_dataclass(cls, data):
    if data is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            if isinstance(ft, str):
                ft = eval(ft)
            if hasattr(ft, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(config_dir: str) -> SimConfig:
    config_path = Path(config_dir)
    raw = {}
    for name in ["system", "dram", "sram_pim", "energy"]:
        fpath = config_path / f"{name}.yaml"
        if fpath.exists():
            with open(fpath) as f:
                raw[name] = yaml.safe_load(f)

    system_raw = raw.get("system", {})
    system_cfg = _dict_to_dataclass(SystemConfig, system_raw.get("system", {}))
    workload_cfg = _dict_to_dataclass(WorkloadConfig, system_raw.get("workload", {}))
    dram_cfg = _dict_to_dataclass(DRAMConfig, raw.get("dram", {}).get("dram", {}))
    sram_pim_cfg = _dict_to_dataclass(SRAMPIMConfig, raw.get("sram_pim", {}).get("sram_pim", {}))
    energy_cfg = _dict_to_dataclass(EnergyConfig, raw.get("energy", {}).get("energy", {}))

    return SimConfig(
        system=system_cfg,
        workload=workload_cfg,
        dram=dram_cfg,
        sram_pim=sram_pim_cfg,
        energy=energy_cfg,
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/sram_pim && python -m pytest tests/test_config.py -v`
Expected: 4 PASSED

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: project scaffold with YAML config system"
```

---

### Task 2: Memory Object Table and Trace IR

**Files:**
- Create: `src/trace_ir.py`
- Create: `src/memory_object.py`
- Create: `tests/test_memory_object.py`
- Create: `tests/test_trace_ir.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MemoryObject` dataclass with fields: `object_id: str`, `obj_type: ObjType`, `bytes: int`, `precision: str`, `dram_addr: int`, `sram_tile: int`, `sram_banks: list[int]`, `valid_in_dram: bool`, `valid_in_sram: bool`, `dirty_in_sram: bool`, `pinned: bool`, `power_state: str`, `first_use: str`, `last_use: str`, `reuse_count: int`
  - `ObjType` enum: WEIGHT, ACTIVATION, STATE, PSUM, LUT, META, OUTPUT, COORD, DIST
  - `OpCode` enum: SRAM_ALLOC, SRAM_FREE, DMA_LOAD, DMA_STORE, DMA_PREFETCH, SRAM_RD, SRAM_WR, PIM_MAC, PIM_EW_OP, PIM_REDUCE, PIM_NL, PIM_WRITEBACK, BARRIER, POWER_SET
  - `TraceCommand` dataclass: `cmd_id: int`, `op: OpCode`, `object_id: str`, `src: str`, `dst: str`, `bytes: int`, `attrs: dict`, `deps: list[int]`
  - `parse_trace(path: str) -> list[TraceCommand]`
  - `write_trace(commands: list[TraceCommand], path: str) -> None`

- [ ] **Step 1: Write failing tests for trace IR**

```python
# tests/test_trace_ir.py
import tempfile, os
from src.trace_ir import OpCode, TraceCommand, parse_trace, write_trace

def test_opcode_enum():
    assert OpCode.DMA_LOAD.name == "DMA_LOAD"
    assert OpCode.PIM_MAC.name == "PIM_MAC"

def test_trace_command_creation():
    cmd = TraceCommand(
        cmd_id=0, op=OpCode.DMA_LOAD, object_id="W0_tile0",
        src="DRAM:0x1000", dst="SRAM:T0:B0-7",
        bytes=65536, attrs={"stream": "weight"}, deps=[]
    )
    assert cmd.op == OpCode.DMA_LOAD
    assert cmd.bytes == 65536

def test_trace_roundtrip():
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0_tile0", "-", "SRAM:T0:B0-7", 65536, {"pinned": 1}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0_tile0", "DRAM:0x1000", "SRAM:T0:B0-7", 65536, {"stream": "weight"}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0_psum", "W0_tile0,X0_tile0", "SRAM:T2", 0, {"mode": "bank"}, [1]),
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trace', delete=False) as f:
        path = f.name
    try:
        write_trace(cmds, path)
        loaded = parse_trace(path)
        assert len(loaded) == 3
        assert loaded[0].op == OpCode.SRAM_ALLOC
        assert loaded[1].deps == [0]
        assert loaded[2].object_id == "Y0_psum"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Write failing tests for memory object**

```python
# tests/test_memory_object.py
from src.memory_object import MemoryObject, ObjType

def test_memory_object_creation():
    obj = MemoryObject(
        object_id="W_l3_tile_07", obj_type=ObjType.WEIGHT,
        bytes=65536, precision="int8"
    )
    assert obj.valid_in_dram is True
    assert obj.valid_in_sram is False
    assert obj.dirty_in_sram is False
    assert obj.pinned is False

def test_memory_object_load_to_sram():
    obj = MemoryObject("W0", ObjType.WEIGHT, 65536, "int8")
    obj.load_to_sram(sram_tile=0, sram_banks=[0, 1, 2, 3])
    assert obj.valid_in_sram is True
    assert obj.sram_tile == 0
    assert obj.sram_banks == [0, 1, 2, 3]

def test_memory_object_mark_dirty():
    obj = MemoryObject("P0", ObjType.PSUM, 1024, "int32")
    obj.load_to_sram(0, [0])
    obj.mark_dirty()
    assert obj.dirty_in_sram is True

def test_memory_object_evict_clean():
    obj = MemoryObject("X0", ObjType.ACTIVATION, 1024, "int8")
    obj.load_to_sram(0, [0])
    needs_writeback = obj.evict()
    assert needs_writeback is False
    assert obj.valid_in_sram is False

def test_memory_object_evict_dirty():
    obj = MemoryObject("P0", ObjType.PSUM, 1024, "int32")
    obj.load_to_sram(0, [0])
    obj.mark_dirty()
    needs_writeback = obj.evict()
    assert needs_writeback is True

def test_memory_object_pinned_cannot_evict():
    obj = MemoryObject("W0", ObjType.WEIGHT, 1024, "int8", pinned=True)
    obj.load_to_sram(0, [0])
    import pytest
    with pytest.raises(RuntimeError):
        obj.evict()

def test_power_gate_clears_valid():
    obj = MemoryObject("W0", ObjType.WEIGHT, 1024, "int8")
    obj.load_to_sram(0, [0])
    obj.power_gate()
    assert obj.valid_in_sram is False
    assert obj.power_state == "power_gated"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/sram_pim && python -m pytest tests/test_trace_ir.py tests/test_memory_object.py -v`
Expected: FAIL

- [ ] **Step 4: Implement memory_object.py**

```python
# src/memory_object.py
from dataclasses import dataclass, field
from enum import Enum


class ObjType(Enum):
    WEIGHT = "WEIGHT"
    ACTIVATION = "ACTIVATION"
    STATE = "STATE"
    PSUM = "PSUM"
    LUT = "LUT"
    META = "META"
    OUTPUT = "OUTPUT"
    COORD = "COORD"
    DIST = "DIST"


@dataclass
class MemoryObject:
    object_id: str
    obj_type: ObjType
    bytes: int
    precision: str
    dram_addr: int = 0
    sram_tile: int = -1
    sram_banks: list = field(default_factory=list)
    valid_in_dram: bool = True
    valid_in_sram: bool = False
    dirty_in_sram: bool = False
    pinned: bool = False
    power_state: str = "active"
    first_use: str = ""
    last_use: str = ""
    reuse_count: int = 0

    def load_to_sram(self, sram_tile: int, sram_banks: list):
        self.sram_tile = sram_tile
        self.sram_banks = list(sram_banks)
        self.valid_in_sram = True
        self.power_state = "active"

    def mark_dirty(self):
        self.dirty_in_sram = True

    def evict(self) -> bool:
        if self.pinned:
            raise RuntimeError(f"Cannot evict pinned object {self.object_id}")
        needs_writeback = self.dirty_in_sram
        self.valid_in_sram = False
        self.dirty_in_sram = False
        self.sram_tile = -1
        self.sram_banks = []
        return needs_writeback

    def writeback_complete(self):
        self.valid_in_dram = True
        self.dirty_in_sram = False

    def power_gate(self):
        if self.dirty_in_sram:
            raise RuntimeError(f"Cannot power-gate bank with dirty object {self.object_id}")
        self.valid_in_sram = False
        self.power_state = "power_gated"
        self.sram_tile = -1
        self.sram_banks = []

    def wakeup(self):
        self.power_state = "active"
```

- [ ] **Step 5: Implement trace_ir.py**

```python
# src/trace_ir.py
from dataclasses import dataclass, field
from enum import Enum
import json


class OpCode(Enum):
    SRAM_ALLOC = "SRAM_ALLOC"
    SRAM_FREE = "SRAM_FREE"
    DMA_LOAD = "DMA_LOAD"
    DMA_STORE = "DMA_STORE"
    DMA_PREFETCH = "DMA_PREFETCH"
    SRAM_RD = "SRAM_RD"
    SRAM_WR = "SRAM_WR"
    PIM_MAC = "PIM_MAC"
    PIM_EW_OP = "PIM_EW_OP"
    PIM_REDUCE = "PIM_REDUCE"
    PIM_NL = "PIM_NL"
    PIM_WRITEBACK = "PIM_WRITEBACK"
    BARRIER = "BARRIER"
    POWER_SET = "POWER_SET"


@dataclass
class TraceCommand:
    cmd_id: int
    op: OpCode
    object_id: str
    src: str
    dst: str
    bytes: int
    attrs: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)


def write_trace(commands: list, path: str) -> None:
    with open(path, 'w') as f:
        for cmd in commands:
            deps_str = ",".join(str(d) for d in cmd.deps) if cmd.deps else "-"
            attrs_str = json.dumps(cmd.attrs) if cmd.attrs else "{}"
            line = f"{cmd.cmd_id}\t{cmd.op.value}\t{cmd.object_id}\t{cmd.src}\t{cmd.dst}\t{cmd.bytes}\t{attrs_str}\t{deps_str}\n"
            f.write(line)


def parse_trace(path: str) -> list:
    commands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            cmd_id = int(parts[0])
            op = OpCode(parts[1])
            object_id = parts[2]
            src = parts[3]
            dst = parts[4]
            nbytes = int(parts[5])
            attrs = json.loads(parts[6]) if parts[6] != "{}" else {}
            deps_str = parts[7]
            deps = [] if deps_str == "-" else [int(d) for d in deps_str.split(",")]
            commands.append(TraceCommand(cmd_id, op, object_id, src, dst, nbytes, attrs, deps))
    return commands
```

- [ ] **Step 6: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_trace_ir.py tests/test_memory_object.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: memory object model and trace IR with ISA"
```

---

### Task 3: SRAM Capacity Manager

**Files:**
- Create: `src/memory_manager.py`
- Create: `tests/test_memory_manager.py`

**Interfaces:**
- Consumes: `MemoryObject`, `ObjType` from `src/memory_object.py`; `SRAMPIMConfig` from `src/config.py`
- Produces:
  - `MemoryManager.__init__(self, config: SRAMPIMConfig)`
  - `MemoryManager.register_object(self, obj: MemoryObject) -> None`
  - `MemoryManager.allocate(self, object_id: str, sram_tile: int, sram_banks: list[int]) -> bool`
  - `MemoryManager.free(self, object_id: str) -> None`
  - `MemoryManager.get_object(self, object_id: str) -> MemoryObject`
  - `MemoryManager.get_free_bytes(self, tile: int) -> int`
  - `MemoryManager.find_eviction_candidate(self, tile: int, needed_bytes: int) -> list[str]`
  - `MemoryManager.stats: dict` with alloc_bytes, free_bytes, spill_count, writeback_count, etc.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_manager.py
import pytest
from src.memory_manager import MemoryManager
from src.memory_object import MemoryObject, ObjType
from src.config import SRAMPIMConfig

def make_config():
    return SRAMPIMConfig(tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64)

def test_allocate_object():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("W0", ObjType.WEIGHT, 8192, "int8")
    mgr.register_object(obj)
    ok = mgr.allocate("W0", sram_tile=0, sram_banks=[0, 1])
    assert ok is True
    assert mgr.get_object("W0").valid_in_sram is True

def test_allocate_exceeds_capacity():
    mgr = MemoryManager(make_config())
    # tile has 4 banks * 8KB = 32KB
    obj = MemoryObject("big", ObjType.ACTIVATION, 33 * 1024, "int8")
    mgr.register_object(obj)
    ok = mgr.allocate("big", sram_tile=0, sram_banks=[0, 1, 2, 3])
    assert ok is False

def test_free_object():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", sram_tile=0, sram_banks=[0])
    mgr.free("X0")
    assert mgr.get_object("X0").valid_in_sram is False

def test_free_bytes_tracking():
    cfg = make_config()
    mgr = MemoryManager(cfg)
    total_per_tile = cfg.banks_per_tile * cfg.bank_capacity_kb * 1024
    assert mgr.get_free_bytes(0) == total_per_tile
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", sram_tile=0, sram_banks=[0])
    assert mgr.get_free_bytes(0) == total_per_tile - 4096

def test_eviction_candidate():
    mgr = MemoryManager(make_config())
    obj1 = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    obj2 = MemoryObject("W0", ObjType.WEIGHT, 4096, "int8", pinned=True)
    mgr.register_object(obj1)
    mgr.register_object(obj2)
    mgr.allocate("X0", 0, [0])
    mgr.allocate("W0", 0, [1])
    candidates = mgr.find_eviction_candidate(tile=0, needed_bytes=4096)
    assert "X0" in candidates
    assert "W0" not in candidates

def test_read_unresident_raises():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    with pytest.raises(RuntimeError):
        mgr.check_valid_for_read("X0")

def test_stats_tracking():
    mgr = MemoryManager(make_config())
    obj = MemoryObject("X0", ObjType.ACTIVATION, 4096, "int8")
    mgr.register_object(obj)
    mgr.allocate("X0", 0, [0])
    assert mgr.stats["alloc_bytes"] == 4096
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_memory_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement memory_manager.py**

```python
# src/memory_manager.py
from src.memory_object import MemoryObject, ObjType
from src.config import SRAMPIMConfig


class MemoryManager:
    def __init__(self, config: SRAMPIMConfig):
        self.config = config
        self.objects: dict[str, MemoryObject] = {}
        self.tile_usage: dict[int, int] = {}
        self.tile_capacity: dict[int, int] = {}
        for t in range(config.tiles):
            cap = config.banks_per_tile * config.bank_capacity_kb * 1024
            self.tile_capacity[t] = cap
            self.tile_usage[t] = 0
        self.stats = {
            "alloc_bytes": 0,
            "free_bytes": 0,
            "spill_count": 0,
            "writeback_count": 0,
            "eviction_count": 0,
        }

    def register_object(self, obj: MemoryObject):
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> MemoryObject:
        return self.objects[object_id]

    def allocate(self, object_id: str, sram_tile: int, sram_banks: list) -> bool:
        obj = self.objects[object_id]
        if self.tile_usage[sram_tile] + obj.bytes > self.tile_capacity[sram_tile]:
            return False
        obj.load_to_sram(sram_tile, sram_banks)
        self.tile_usage[sram_tile] += obj.bytes
        self.stats["alloc_bytes"] += obj.bytes
        return True

    def free(self, object_id: str):
        obj = self.objects[object_id]
        if obj.valid_in_sram and obj.sram_tile >= 0:
            self.tile_usage[obj.sram_tile] -= obj.bytes
            self.stats["free_bytes"] += obj.bytes
        obj.valid_in_sram = False
        obj.sram_tile = -1
        obj.sram_banks = []

    def get_free_bytes(self, tile: int) -> int:
        return self.tile_capacity[tile] - self.tile_usage[tile]

    def find_eviction_candidate(self, tile: int, needed_bytes: int) -> list:
        candidates = []
        freed = 0
        resident = [
            oid for oid, obj in self.objects.items()
            if obj.valid_in_sram and obj.sram_tile == tile and not obj.pinned
        ]
        # Sort: clean activations first, then clean weights, then dirty
        def evict_priority(oid):
            obj = self.objects[oid]
            type_pri = 0 if obj.obj_type == ObjType.ACTIVATION else 1
            dirty_pri = 0 if not obj.dirty_in_sram else 1
            return (dirty_pri, type_pri)

        resident.sort(key=evict_priority)
        for oid in resident:
            if freed >= needed_bytes:
                break
            candidates.append(oid)
            freed += self.objects[oid].bytes
        return candidates

    def check_valid_for_read(self, object_id: str):
        obj = self.objects[object_id]
        if not obj.valid_in_sram:
            raise RuntimeError(f"Object {object_id} not valid in SRAM, cannot read/PIM")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_memory_manager.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: SRAM capacity manager with allocation/eviction"
```

---

### Task 4: DRAM Model and Energy Model

**Files:**
- Create: `src/dram_model.py`
- Create: `src/energy_model.py`
- Create: `tests/test_dram_model.py`
- Create: `tests/test_energy_model.py`

**Interfaces:**
- Consumes: `DRAMConfig`, `EnergyConfig`, `SRAMPIMConfig` from `src/config.py`
- Produces:
  - `DRAMModel.__init__(self, config: DRAMConfig)`
  - `DRAMModel.get_read_latency(self, nbytes: int) -> int` (cycles)
  - `DRAMModel.get_write_latency(self, nbytes: int) -> int` (cycles)
  - `EnergyModel.__init__(self, config: EnergyConfig, sram_config: SRAMPIMConfig, dram_config: DRAMConfig, freq_hz: int)`
  - `EnergyModel.add_dram_read(self, nbytes: int)`
  - `EnergyModel.add_dram_write(self, nbytes: int)`
  - `EnergyModel.add_sram_read(self, count: int)`
  - `EnergyModel.add_sram_write(self, count: int)`
  - `EnergyModel.add_pim_mac(self, count: int)`
  - `EnergyModel.add_pim_reduce(self, count: int)`
  - `EnergyModel.add_noc(self, nbytes: int)`
  - `EnergyModel.add_leakage(self, cycles: int)`
  - `EnergyModel.get_breakdown(self) -> dict`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dram_model.py
from src.dram_model import DRAMModel
from src.config import DRAMConfig

def test_analytical_read_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    # 64KB at 25.6 GB/s = 65536 / 25.6e9 * 1e9 = 2.56 us = 2560 ns = 2560 cycles at 1GHz
    # plus 80ns fixed = 2640 cycles
    lat = model.get_read_latency(65536)
    assert lat == 2640

def test_analytical_write_latency():
    config = DRAMConfig(effective_bandwidth_gbps=25.6, fixed_latency_ns=80)
    model = DRAMModel(config, freq_hz=1_000_000_000)
    lat = model.get_write_latency(65536)
    assert lat == 2640
```

```python
# tests/test_energy_model.py
from src.energy_model import EnergyModel
from src.config import EnergyConfig, SRAMPIMConfig, DRAMConfig

def test_dram_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_read(1000)
    bd = energy.get_breakdown()
    # 1000 * 15.0 = 15000 pJ read + 1000 * 8.0 = 8000 pJ io
    assert bd["dram_read_pj"] == 15000.0
    assert bd["offchip_io_pj"] == 8000.0

def test_sram_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_sram_read(100)
    energy.add_sram_write(50)
    bd = energy.get_breakdown()
    assert bd["sram_read_pj"] == 100 * 3.2
    assert bd["sram_write_pj"] == 50 * 3.8

def test_pim_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_pim_mac(1000)
    energy.add_pim_reduce(500)
    bd = energy.get_breakdown()
    assert bd["pim_mac_pj"] == 1000 * 0.08
    assert bd["pim_reduce_pj"] == 500 * 0.04

def test_leakage_energy():
    cfg = EnergyConfig()
    sram_cfg = SRAMPIMConfig(tiles=2, banks_per_tile=4)
    energy = EnergyModel(cfg, sram_cfg, DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_leakage(1000)  # 1000 cycles at 1GHz = 1us
    bd = energy.get_breakdown()
    # 8 banks * 0.39 mW * 1us = 8 * 0.39 * 1e-3 * 1e-6 W*s = 8 * 0.39e-9 J = 8 * 0.39e-9 * 1e12 pJ = 8 * 390 * 1e-3 pJ
    # = 8 * 0.39 * 1e3 * 1e-6 pJ per cycle... let me recalc:
    # P_leak = 0.39 mW/bank = 0.39e-3 W/bank
    # 8 banks total, 1000 cycles at 1GHz = 1e-6 s
    # E = 8 * 0.39e-3 * 1e-6 J = 3.12e-9 J = 3.12e-9 * 1e12 pJ = 3120 pJ
    assert abs(bd["sram_leakage_pj"] - 3120.0) < 1.0

def test_total_energy():
    energy = EnergyModel(EnergyConfig(), SRAMPIMConfig(), DRAMConfig(), freq_hz=1_000_000_000)
    energy.add_dram_read(100)
    energy.add_pim_mac(100)
    bd = energy.get_breakdown()
    assert bd["total_pj"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_dram_model.py tests/test_energy_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dram_model.py**

```python
# src/dram_model.py
import math
from src.config import DRAMConfig


class DRAMModel:
    def __init__(self, config: DRAMConfig, freq_hz: int):
        self.config = config
        self.freq_hz = freq_hz
        bw_bytes_per_sec = config.effective_bandwidth_gbps * 1e9
        self.bytes_per_cycle = bw_bytes_per_sec / freq_hz
        self.fixed_latency_cycles = int(config.fixed_latency_ns * freq_hz / 1e9)

    def get_read_latency(self, nbytes: int) -> int:
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle)
        return self.fixed_latency_cycles + transfer_cycles

    def get_write_latency(self, nbytes: int) -> int:
        transfer_cycles = math.ceil(nbytes / self.bytes_per_cycle)
        return self.fixed_latency_cycles + transfer_cycles
```

- [ ] **Step 4: Implement energy_model.py**

```python
# src/energy_model.py
from src.config import EnergyConfig, SRAMPIMConfig, DRAMConfig


class EnergyModel:
    def __init__(self, config: EnergyConfig, sram_config: SRAMPIMConfig,
                 dram_config: DRAMConfig, freq_hz: int):
        self.config = config
        self.sram_config = sram_config
        self.dram_config = dram_config
        self.freq_hz = freq_hz
        self.total_banks = sram_config.tiles * sram_config.banks_per_tile
        self._counters = {
            "dram_read_bytes": 0,
            "dram_write_bytes": 0,
            "sram_read_count": 0,
            "sram_write_count": 0,
            "pim_mac_count": 0,
            "pim_reduce_count": 0,
            "pim_nl_count": 0,
            "noc_bytes": 0,
            "leakage_cycles": 0,
            "command_count": 0,
        }

    def add_dram_read(self, nbytes: int):
        self._counters["dram_read_bytes"] += nbytes

    def add_dram_write(self, nbytes: int):
        self._counters["dram_write_bytes"] += nbytes

    def add_sram_read(self, count: int):
        self._counters["sram_read_count"] += count

    def add_sram_write(self, count: int):
        self._counters["sram_write_count"] += count

    def add_pim_mac(self, count: int):
        self._counters["pim_mac_count"] += count

    def add_pim_reduce(self, count: int):
        self._counters["pim_reduce_count"] += count

    def add_pim_nl(self, count: int):
        self._counters["pim_nl_count"] += count

    def add_noc(self, nbytes: int):
        self._counters["noc_bytes"] += nbytes

    def add_leakage(self, cycles: int):
        self._counters["leakage_cycles"] += cycles

    def add_command(self):
        self._counters["command_count"] += 1

    def get_breakdown(self) -> dict:
        c = self._counters
        ec = self.config
        dc = self.dram_config

        dram_read = c["dram_read_bytes"] * dc.energy.read_pj_per_byte
        dram_write = c["dram_write_bytes"] * dc.energy.write_pj_per_byte
        io = (c["dram_read_bytes"] + c["dram_write_bytes"]) * dc.energy.io_pj_per_byte
        sram_read = c["sram_read_count"] * ec.sram.read_pj_per_access
        sram_write = c["sram_write_count"] * ec.sram.write_pj_per_access

        # Leakage: P_leak(W) * time(s) -> J -> pJ
        time_s = c["leakage_cycles"] / self.freq_hz
        leak_pj = self.total_banks * ec.sram.leakage_mw_per_bank * 1e-3 * time_s * 1e12

        pim_mac = c["pim_mac_count"] * ec.pim.mac_pj_per_op
        pim_reduce = c["pim_reduce_count"] * ec.pim.reduce_pj_per_op
        pim_nl = c["pim_nl_count"] * ec.pim.nonlinear_pj_per_elem
        noc = c["noc_bytes"] * ec.noc.pj_per_byte_per_hop * ec.noc.average_hops
        dma = (c["dram_read_bytes"] + c["dram_write_bytes"]) * ec.dma.pj_per_byte
        control = c["command_count"] * ec.pim.control_pj_per_command

        total = (dram_read + dram_write + io + sram_read + sram_write +
                 leak_pj + pim_mac + pim_reduce + pim_nl + noc + dma + control)

        return {
            "dram_read_pj": dram_read,
            "dram_write_pj": dram_write,
            "offchip_io_pj": io,
            "dma_pj": dma,
            "sram_read_pj": sram_read,
            "sram_write_pj": sram_write,
            "sram_leakage_pj": leak_pj,
            "pim_mac_pj": pim_mac,
            "pim_reduce_pj": pim_reduce,
            "pim_nl_pj": pim_nl,
            "noc_pj": noc,
            "control_pj": control,
            "total_pj": total,
            **{k: v for k, v in c.items()},
        }
```

- [ ] **Step 5: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_dram_model.py tests/test_energy_model.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: analytical DRAM model and energy breakdown model"
```

---

### Task 5: Event-Driven Simulator Core

**Files:**
- Create: `src/simulator.py`
- Create: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `SimConfig` from config; `TraceCommand`, `OpCode` from trace_ir; `MemoryManager` from memory_manager; `DRAMModel` from dram_model; `EnergyModel` from energy_model
- Produces:
  - `Simulator.__init__(self, config: SimConfig)`
  - `Simulator.load_trace(self, commands: list[TraceCommand])`
  - `Simulator.run(self) -> dict` — returns full report with latency_breakdown, energy_breakdown, traffic, correctness stats
  - `Simulator.cycle: int` — current cycle

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulator.py
import pytest
from src.simulator import Simulator
from src.config import load_config, SimConfig, SRAMPIMConfig, SystemConfig
from src.trace_ir import TraceCommand, OpCode
from src.memory_object import MemoryObject, ObjType


def make_simple_config():
    return SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64),
    )


def test_empty_trace():
    sim = Simulator(make_simple_config())
    sim.load_trace([])
    report = sim.run()
    assert report["latency"]["total_cycles"] == 0


def test_dma_load_then_pim_mac():
    """Basic: alloc -> load weight -> load activation -> PIM MAC -> store output"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000", "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7", 4096, {"type": "ACTIVATION"}, []),
        TraceCommand(3, OpCode.DMA_LOAD, "X0", "DRAM:0x9000", "SRAM:T0:B4-7", 4096, {}, [2]),
        TraceCommand(4, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1", 0, {"mac_count": 1024}, [1, 3]),
        TraceCommand(5, OpCode.DMA_STORE, "Y0", "SRAM:T1:B0-1", "DRAM:0xA000", 2048, {}, [4]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["latency"]["total_cycles"] > 0
    assert report["energy"]["total_pj"] > 0
    assert report["traffic"]["dram_read_bytes"] == 8192 + 4096
    assert report["traffic"]["dram_write_bytes"] == 2048


def test_dependency_enforcement():
    """PIM_MAC cannot run before its DMA_LOAD dependency completes"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT"}, []),
        TraceCommand(1, OpCode.DMA_LOAD, "W0", "DRAM:0x1000", "SRAM:T0:B0-3", 8192, {}, [0]),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0", "SRAM:T1:B0", 0, {"mac_count": 100}, [1]),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    # PIM_MAC starts after DMA_LOAD finishes
    assert report["latency"]["total_cycles"] > report["latency"]["dram_load_cycles"]


def test_unresident_read_detected():
    """Correctness: reading object not in SRAM should be flagged"""
    cmds = [
        # PIM_MAC without DMA_LOAD — object not in SRAM
        TraceCommand(0, OpCode.PIM_MAC, "Y0", "W0_missing", "SRAM:T0:B0", 0, {"mac_count": 100}, []),
    ]
    sim = Simulator(make_simple_config())
    sim.load_trace(cmds)
    report = sim.run()
    assert report["correctness"]["illegal_read_unresident_object"] > 0


def test_warm_resident_mode():
    """In warm_resident mode, pre-loaded weights skip DMA cost"""
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-3", 8192, {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B4-7", 4096, {"type": "ACTIVATION"}, []),
        TraceCommand(2, OpCode.DMA_LOAD, "X0", "DRAM:0x9000", "SRAM:T0:B4-7", 4096, {}, [1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T1:B0-1", 0, {"mac_count": 1024}, [0, 2]),
    ]
    config = make_simple_config()
    config.system.mode = "warm_resident"
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert report["traffic"]["dram_read_bytes"] == 4096  # Only activation loaded
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_simulator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement simulator.py**

```python
# src/simulator.py
import heapq
from src.config import SimConfig
from src.trace_ir import TraceCommand, OpCode
from src.memory_object import MemoryObject, ObjType
from src.memory_manager import MemoryManager
from src.dram_model import DRAMModel
from src.energy_model import EnergyModel


def _parse_sram_loc(loc_str: str):
    """Parse 'SRAM:T0:B0-3' -> (tile=0, banks=[0,1,2,3])"""
    parts = loc_str.split(":")
    tile = int(parts[1][1:])
    bank_str = parts[2][1:]
    if "-" in bank_str:
        lo, hi = bank_str.split("-")
        banks = list(range(int(lo), int(hi) + 1))
    else:
        banks = [int(b) for b in bank_str.split(",")]
    return tile, banks


def _obj_type_from_str(s: str) -> ObjType:
    mapping = {
        "WEIGHT": ObjType.WEIGHT, "ACTIVATION": ObjType.ACTIVATION,
        "STATE": ObjType.STATE, "PSUM": ObjType.PSUM, "LUT": ObjType.LUT,
        "META": ObjType.META, "OUTPUT": ObjType.OUTPUT,
        "COORD": ObjType.COORD, "DIST": ObjType.DIST,
    }
    return mapping.get(s, ObjType.ACTIVATION)


class Simulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.mem_mgr = MemoryManager(config.sram_pim)
        self.dram = DRAMModel(config.dram, config.system.frequency_hz)
        self.energy = EnergyModel(config.energy, config.sram_pim,
                                  config.dram, config.system.frequency_hz)
        self.cycle = 0
        self.commands = []
        self.completed = set()
        self.event_queue = []  # (finish_cycle, cmd_id)
        self.latency_breakdown = {
            "dram_load_cycles": 0,
            "dram_store_cycles": 0,
            "pim_compute_cycles": 0,
            "pim_reduce_cycles": 0,
            "stall_dependency_cycles": 0,
        }
        self.correctness = {
            "illegal_read_unresident_object": 0,
            "illegal_evict_dirty_without_writeback": 0,
            "capacity_overcommit_events": 0,
            "dependency_violations": 0,
        }

    def load_trace(self, commands: list):
        self.commands = list(commands)

    def _deps_ready(self, cmd: TraceCommand) -> bool:
        return all(d in self.completed for d in cmd.deps)

    def _issue_command(self, cmd: TraceCommand) -> int:
        self.energy.add_command()

        if cmd.op == OpCode.SRAM_ALLOC:
            return self._issue_alloc(cmd)
        elif cmd.op == OpCode.DMA_LOAD:
            return self._issue_dma_load(cmd)
        elif cmd.op == OpCode.DMA_STORE:
            return self._issue_dma_store(cmd)
        elif cmd.op == OpCode.PIM_MAC:
            return self._issue_pim_mac(cmd)
        elif cmd.op == OpCode.PIM_EW_OP:
            return self._issue_pim_ew(cmd)
        elif cmd.op == OpCode.PIM_REDUCE:
            return self._issue_pim_reduce(cmd)
        elif cmd.op == OpCode.PIM_NL:
            return self._issue_pim_nl(cmd)
        elif cmd.op == OpCode.SRAM_FREE:
            return self._issue_free(cmd)
        elif cmd.op == OpCode.BARRIER:
            return 0
        elif cmd.op == OpCode.PIM_WRITEBACK:
            return self._issue_pim_writeback(cmd)
        elif cmd.op == OpCode.POWER_SET:
            return 0
        return 0

    def _issue_alloc(self, cmd: TraceCommand) -> int:
        tile, banks = _parse_sram_loc(cmd.dst)
        obj_type_str = cmd.attrs.get("type", "ACTIVATION")
        pinned = bool(cmd.attrs.get("pinned", False))
        preloaded = bool(cmd.attrs.get("preloaded", False))
        obj = MemoryObject(
            object_id=cmd.object_id,
            obj_type=_obj_type_from_str(obj_type_str),
            bytes=cmd.bytes,
            precision="int8",
            pinned=pinned,
        )
        self.mem_mgr.register_object(obj)
        ok = self.mem_mgr.allocate(cmd.object_id, tile, banks)
        if not ok:
            self.correctness["capacity_overcommit_events"] += 1
        if preloaded:
            obj.valid_in_sram = True
        return 0

    def _issue_dma_load(self, cmd: TraceCommand) -> int:
        lat = self.dram.get_read_latency(cmd.bytes)
        self.energy.add_dram_read(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.valid_in_sram = True
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_write(n_accesses)
        self.latency_breakdown["dram_load_cycles"] += lat
        return lat

    def _issue_dma_store(self, cmd: TraceCommand) -> int:
        lat = self.dram.get_write_latency(cmd.bytes)
        self.energy.add_dram_write(cmd.bytes)
        self.energy.add_noc(cmd.bytes)
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_read(n_accesses)
            obj.writeback_complete()
        self.latency_breakdown["dram_store_cycles"] += lat
        return lat

    def _check_inputs_valid(self, cmd: TraceCommand) -> bool:
        input_ids = [s.strip() for s in cmd.src.split(",") if s.strip() and s.strip() != "-"]
        for iid in input_ids:
            obj = self.mem_mgr.objects.get(iid)
            if obj is None or not obj.valid_in_sram:
                self.correctness["illegal_read_unresident_object"] += 1
                return False
        return True

    def _issue_pim_mac(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        mac_count = cmd.attrs.get("mac_count", 0)
        lat = self.config.sram_pim.pim.mac_latency_cycles
        if mac_count > 0:
            lat = max(lat, mac_count * self.config.sram_pim.pim.mac_issue_interval_cycles)
        self.energy.add_pim_mac(mac_count)
        # Create output object if not exists
        if cmd.object_id not in self.mem_mgr.objects:
            out_obj = MemoryObject(cmd.object_id, ObjType.PSUM, cmd.bytes or 1024, "int32")
            self.mem_mgr.register_object(out_obj)
            if cmd.dst.startswith("SRAM:"):
                tile, banks = _parse_sram_loc(cmd.dst)
                self.mem_mgr.allocate(cmd.object_id, tile, banks)
            out_obj.mark_dirty()
        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_ew(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = max(1, count // self.config.sram_pim.pim.lanes_per_bank)
        self.energy.add_pim_mac(count)
        self.latency_breakdown["pim_compute_cycles"] += lat
        return lat

    def _issue_pim_reduce(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.reduce_latency_cycles
        self.energy.add_pim_reduce(count)
        self.latency_breakdown["pim_reduce_cycles"] += lat
        return lat

    def _issue_pim_nl(self, cmd: TraceCommand) -> int:
        self._check_inputs_valid(cmd)
        count = cmd.attrs.get("count", 0)
        lat = self.config.sram_pim.pim.nonlinear_latency_cycles
        self.energy.add_pim_nl(count)
        return lat

    def _issue_pim_writeback(self, cmd: TraceCommand) -> int:
        obj = self.mem_mgr.objects.get(cmd.object_id)
        if obj:
            obj.mark_dirty()
            n_accesses = max(1, cmd.bytes // self.config.sram_pim.word_bytes)
            self.energy.add_sram_write(n_accesses)
        return 1

    def _issue_free(self, cmd: TraceCommand) -> int:
        self.mem_mgr.free(cmd.object_id)
        return 0

    def run(self) -> dict:
        if not self.commands:
            return self._make_report()

        pending = {cmd.cmd_id: cmd for cmd in self.commands}
        issued = {}
        self.cycle = 0
        max_cycles = 100_000_000

        while (pending or self.event_queue) and self.cycle < max_cycles:
            # Complete events at this cycle
            while self.event_queue and self.event_queue[0][0] <= self.cycle:
                _, cmd_id = heapq.heappop(self.event_queue)
                self.completed.add(cmd_id)

            # Find and issue ready commands
            ready = [cid for cid, cmd in pending.items() if self._deps_ready(cmd)]
            for cid in ready:
                cmd = pending.pop(cid)
                latency = self._issue_command(cmd)
                if latency > 0:
                    heapq.heappush(self.event_queue, (self.cycle + latency, cid))
                else:
                    self.completed.add(cid)

            # Add leakage for this cycle
            self.energy.add_leakage(1)

            if not self.event_queue and not any(
                self._deps_ready(cmd) for cmd in pending.values()
            ):
                if pending:
                    # Deadlock or unreachable deps — force complete remaining
                    for cid in list(pending.keys()):
                        self.completed.add(cid)
                    pending.clear()
                break

            self.cycle += 1

        return self._make_report()

    def _make_report(self) -> dict:
        eb = self.energy.get_breakdown()
        freq = self.config.system.frequency_hz
        total_ns = self.cycle * 1e9 / freq

        return {
            "latency": {
                "total_cycles": self.cycle,
                "total_ns": total_ns,
                **self.latency_breakdown,
            },
            "energy": eb,
            "traffic": {
                "dram_read_bytes": eb["dram_read_bytes"],
                "dram_write_bytes": eb["dram_write_bytes"],
                "sram_read_accesses": eb["sram_read_count"],
                "sram_write_accesses": eb["sram_write_count"],
                "noc_bytes": eb["noc_bytes"],
            },
            "correctness": self.correctness,
            "pim": {
                "mac_count": eb["pim_mac_count"],
                "reduce_count": eb["pim_reduce_count"],
            },
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_simulator.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: event-driven simulator core with dependency scheduling"
```

---

### Task 6: GEMM Trace Generator

**Files:**
- Create: `src/tracegen/gen_gemm_trace.py`
- Create: `src/tracegen/__init__.py`
- Create: `tests/test_gen_gemm.py`

**Interfaces:**
- Consumes: `TraceCommand`, `OpCode` from trace_ir; `SRAMPIMConfig` from config
- Produces:
  - `gen_gemm_trace(M: int, N: int, K: int, Tm: int, Tn: int, Tk: int, sram_config: SRAMPIMConfig, precision: str = "int8", mode: str = "cold_start") -> list[TraceCommand]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gen_gemm.py
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig

def make_sram_config():
    return SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256)

def test_gemm_trace_has_all_phases():
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=64, Tn=64, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    ops = [c.op for c in cmds]
    assert OpCode.SRAM_ALLOC in ops
    assert OpCode.DMA_LOAD in ops
    assert OpCode.PIM_MAC in ops
    assert OpCode.DMA_STORE in ops

def test_gemm_trace_dependency_chain():
    cmds = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="cold_start")
    cmd_map = {c.cmd_id: c for c in cmds}
    # Every PIM_MAC should depend on DMA_LOADs
    for c in cmds:
        if c.op == OpCode.PIM_MAC:
            assert len(c.deps) > 0
            for d in c.deps:
                assert cmd_map[d].op in (OpCode.DMA_LOAD, OpCode.SRAM_ALLOC, OpCode.PIM_MAC)

def test_gemm_trace_cold_vs_warm():
    cold = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="cold_start")
    warm = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=make_sram_config(), mode="warm_resident")
    cold_loads = sum(1 for c in cold if c.op == OpCode.DMA_LOAD)
    warm_loads = sum(1 for c in warm if c.op == OpCode.DMA_LOAD)
    # Warm mode skips weight loads
    assert warm_loads < cold_loads

def test_gemm_trace_tiled():
    cmds = gen_gemm_trace(M=256, N=256, K=256, Tm=64, Tn=64, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    mac_cmds = [c for c in cmds if c.op == OpCode.PIM_MAC]
    # Should have (M/Tm) * (N/Tn) * (K/Tk) = 4 * 4 * 2 = 32 MAC commands
    assert len(mac_cmds) == 32

def test_gemm_output_bytes():
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=128, Tn=128, Tk=128,
                          sram_config=make_sram_config(), mode="cold_start")
    stores = [c for c in cmds if c.op == OpCode.DMA_STORE]
    total_store = sum(c.bytes for c in stores)
    # Output Y[128,128] in int8 = 16384 bytes
    assert total_store == 128 * 128 * 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_gemm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gen_gemm_trace.py**

```python
# src/tracegen/__init__.py
# empty

# src/tracegen/gen_gemm_trace.py
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
        # Warm mode: weights pre-loaded
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
            x_banks = [0, 1]

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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_gemm.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: GEMM trace generator with tiling and cold/warm modes"
```

---

### Task 7: End-to-End CLI and Report Generation

**Files:**
- Create: `src/report.py`
- Modify: `main.py`
- Create: `tests/test_e2e.py`

**Interfaces:**
- Consumes: All previous modules
- Produces:
  - `generate_report(result: dict, config: SimConfig) -> str` — YAML-formatted report
  - CLI: `python main.py --config configs/ --workload gemm --M 256 --N 256 --K 256`

- [ ] **Step 1: Write failing e2e test**

```python
# tests/test_e2e.py
import subprocess
import json
import yaml
from src.config import load_config, SimConfig, SystemConfig, SRAMPIMConfig
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.simulator import Simulator
from src.report import generate_report

def test_gemm_e2e_pipeline():
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256),
    )
    cmds = gen_gemm_trace(M=128, N=128, K=128, Tm=64, Tn=64, Tk=128,
                          sram_config=config.sram_pim, mode="cold_start")
    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)
    assert "total_cycles" in report
    assert "total_pj" in report
    assert "dram_read_bytes" in report

def test_report_yaml_parseable():
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="cold_start"),
        sram_pim=SRAMPIMConfig(tiles=4, banks_per_tile=8, bank_capacity_kb=8, total_capacity_kb=256),
    )
    cmds = gen_gemm_trace(M=64, N=64, K=64, Tm=64, Tn=64, Tk=64,
                          sram_config=config.sram_pim, mode="cold_start")
    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)
    parsed = yaml.safe_load(report)
    assert parsed["latency"]["total_cycles"] > 0
    assert parsed["energy"]["total_pj"] > 0
    assert parsed["correctness"]["illegal_read_unresident_object"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_e2e.py -v`
Expected: FAIL

- [ ] **Step 3: Implement report.py**

```python
# src/report.py
import yaml
from src.config import SimConfig


def generate_report(result: dict, config: SimConfig) -> str:
    report = {
        "simulation_config": {
            "mode": config.system.mode,
            "frequency_hz": config.system.frequency_hz,
            "sram_tiles": config.sram_pim.tiles,
            "sram_total_kb": config.sram_pim.total_capacity_kb,
            "pim_mode": config.sram_pim.pim.mode,
        },
        "latency": result["latency"],
        "energy": {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in result["energy"].items()},
        "traffic": result["traffic"],
        "correctness": result["correctness"],
        "pim": result["pim"],
    }
    return yaml.dump(report, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Implement main.py**

```python
# main.py
import argparse
import sys
import yaml

from src.config import load_config
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.simulator import Simulator
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="SRAM-PIM Simulator")
    parser.add_argument("--config", type=str, default="configs", help="Config directory")
    parser.add_argument("--workload", type=str, default="gemm", help="Workload type: gemm")
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--Tm", type=int, default=64)
    parser.add_argument("--Tn", type=int, default=64)
    parser.add_argument("--Tk", type=int, default=128)
    parser.add_argument("--mode", type=str, default=None, help="Override: cold_start|warm_resident|amortized")
    parser.add_argument("--output", type=str, default=None, help="Output YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = args.mode or config.system.mode

    if args.workload == "gemm":
        cmds = gen_gemm_trace(
            M=args.M, N=args.N, K=args.K,
            Tm=args.Tm, Tn=args.Tn, Tk=args.Tk,
            sram_config=config.sram_pim,
            mode=mode,
        )
    else:
        print(f"Unknown workload: {args.workload}", file=sys.stderr)
        sys.exit(1)

    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)

    print(report)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_e2e.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run CLI smoke test**

Run: `cd /home/sram_pim && python main.py --workload gemm --M 128 --N 128 --K 128 --mode cold_start`
Expected: YAML report printed with latency, energy, traffic sections

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: end-to-end CLI with GEMM workload and YAML report"
```

---

### Task 8: Bank Conflict and Resource Constraint Model

**Files:**
- Create: `src/resource_model.py`
- Modify: `src/simulator.py` — add resource checking to command scheduling
- Create: `tests/test_resource_model.py`
- Create: `tests/test_bank_conflict.py`

**Interfaces:**
- Consumes: `SRAMPIMConfig` from config
- Produces:
  - `ResourceModel.__init__(self, config: SRAMPIMConfig)`
  - `ResourceModel.can_issue(self, op: OpCode, banks: list[int], nbytes: int) -> bool`
  - `ResourceModel.reserve(self, op: OpCode, banks: list[int], nbytes: int, duration: int)`
  - `ResourceModel.release_at(self, cycle: int)`
  - `ResourceModel.get_bank_conflicts(self) -> int`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_resource_model.py
from src.resource_model import ResourceModel
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig, PortModelConfig

def make_config():
    return SRAMPIMConfig(
        tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64,
        port_model=PortModelConfig(
            pim_exclusive_with_read=True,
            pim_exclusive_with_write=True,
        )
    )

def test_can_issue_no_conflict():
    rm = ResourceModel(make_config())
    assert rm.can_issue(OpCode.PIM_MAC, banks=[0], nbytes=0) is True

def test_bank_conflict_pim_and_read():
    rm = ResourceModel(make_config())
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=2)
    # Same bank, PIM exclusive with read
    assert rm.can_issue(OpCode.SRAM_RD, banks=[0], nbytes=0) is False
    # Different bank OK
    assert rm.can_issue(OpCode.SRAM_RD, banks=[1], nbytes=0) is True

def test_release_frees_banks():
    rm = ResourceModel(make_config())
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=2)
    rm.release_at(cycle=2)
    assert rm.can_issue(OpCode.SRAM_RD, banks=[0], nbytes=0) is True

def test_power_throttle():
    cfg = make_config()
    cfg.pim.max_parallel_banks = 2
    rm = ResourceModel(cfg)
    rm.reserve(OpCode.PIM_MAC, banks=[0, 1], nbytes=0, duration=2)
    # Already 2 banks active, 3rd should be throttled
    assert rm.can_issue(OpCode.PIM_MAC, banks=[2], nbytes=0) is False
```

```python
# tests/test_bank_conflict.py
from src.simulator import Simulator
from src.config import SimConfig, SystemConfig, SRAMPIMConfig, PortModelConfig
from src.trace_ir import TraceCommand, OpCode

def test_bank_conflict_stall():
    """Two PIM_MACs on same bank should serialize"""
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="warm_resident"),
        sram_pim=SRAMPIMConfig(
            tiles=2, banks_per_tile=4, bank_capacity_kb=8, total_capacity_kb=64,
            port_model=PortModelConfig(pim_exclusive_with_read=True, pim_exclusive_with_write=True),
        ),
    )
    cmds = [
        TraceCommand(0, OpCode.SRAM_ALLOC, "W0", "-", "SRAM:T0:B0-0", 1024,
                     {"pinned": True, "type": "WEIGHT", "preloaded": True}, []),
        TraceCommand(1, OpCode.SRAM_ALLOC, "X0", "-", "SRAM:T0:B1-1", 1024,
                     {"type": "ACTIVATION", "preloaded": True}, []),
        TraceCommand(2, OpCode.PIM_MAC, "Y0", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
        TraceCommand(3, OpCode.PIM_MAC, "Y1", "W0,X0", "SRAM:T0:B0-0", 0,
                     {"mac_count": 100, "banks": [0]}, [0, 1]),
    ]
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    # Two MACs on same bank should take longer than one
    assert report["latency"]["total_cycles"] >= 200
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_resource_model.py tests/test_bank_conflict.py -v`
Expected: FAIL

- [ ] **Step 3: Implement resource_model.py**

```python
# src/resource_model.py
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig


class ResourceModel:
    def __init__(self, config: SRAMPIMConfig):
        self.config = config
        total_banks = config.tiles * config.banks_per_tile
        self.bank_busy_until = [0] * total_banks
        self.bank_op_type = [None] * total_banks
        self.pim_active_banks = 0
        self.current_cycle = 0
        self.bank_conflict_count = 0
        self._reservations = []  # (release_cycle, banks)

    def can_issue(self, op: OpCode, banks: list, nbytes: int = 0) -> bool:
        port = self.config.port_model

        for b in banks:
            if b >= len(self.bank_busy_until):
                continue
            if self.bank_busy_until[b] > self.current_cycle:
                current_op = self.bank_op_type[b]
                if op in (OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE):
                    if port.pim_exclusive_with_read or port.pim_exclusive_with_write:
                        self.bank_conflict_count += 1
                        return False
                if current_op in (OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE):
                    if op in (OpCode.SRAM_RD, OpCode.DMA_LOAD) and port.pim_exclusive_with_read:
                        self.bank_conflict_count += 1
                        return False
                    if op in (OpCode.SRAM_WR, OpCode.DMA_STORE) and port.pim_exclusive_with_write:
                        self.bank_conflict_count += 1
                        return False

        # Power throttle
        if op in (OpCode.PIM_MAC, OpCode.PIM_EW_OP):
            new_active = self.pim_active_banks + len(banks)
            if new_active > self.config.pim.max_parallel_banks:
                return False

        return True

    def reserve(self, op: OpCode, banks: list, nbytes: int, duration: int):
        release_cycle = self.current_cycle + duration
        for b in banks:
            if b < len(self.bank_busy_until):
                self.bank_busy_until[b] = release_cycle
                self.bank_op_type[b] = op

        if op in (OpCode.PIM_MAC, OpCode.PIM_EW_OP):
            self.pim_active_banks += len(banks)

        self._reservations.append((release_cycle, banks, op))

    def release_at(self, cycle: int):
        self.current_cycle = cycle
        remaining = []
        for release_cycle, banks, op in self._reservations:
            if release_cycle <= cycle:
                if op in (OpCode.PIM_MAC, OpCode.PIM_EW_OP):
                    self.pim_active_banks = max(0, self.pim_active_banks - len(banks))
            else:
                remaining.append((release_cycle, banks, op))
        self._reservations = remaining

    def get_bank_conflicts(self) -> int:
        return self.bank_conflict_count
```

- [ ] **Step 4: Integrate ResourceModel into Simulator**

Modify `src/simulator.py` to use `ResourceModel` for scheduling:
- Add `self.resource = ResourceModel(config.sram_pim)` in `__init__`
- In the main loop, check `resource.can_issue()` before issuing
- Call `resource.reserve()` after issue
- Call `resource.release_at(self.cycle)` at start of each cycle
- Track bank_conflict stall cycles in latency_breakdown

- [ ] **Step 5: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_resource_model.py tests/test_bank_conflict.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run full test suite**

Run: `cd /home/sram_pim && python -m pytest tests/ -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: bank conflict and resource constraint model"
```

---

### Task 9: DESTINY/CACTI Adapter for SRAM Parameters

**Files:**
- Create: `src/energy/destiny_adapter.py`
- Create: `src/energy/__init__.py`
- Create: `tests/test_destiny_adapter.py`

**Interfaces:**
- Consumes: DESTINY config template at `/home/Destiny-Memory-simulator/config/3D/SRAM/`
- Produces:
  - `DestinyAdapter.generate_config(self, tech_nm: int, capacity_kb: int, banks: int, word_bits: int) -> str` — returns cfg file content
  - `DestinyAdapter.run(self) -> dict` — runs DESTINY and parses output
  - `DestinyAdapter.parse_output(self, output: str) -> dict` — returns `{read_latency_ns, write_latency_ns, read_energy_pj, write_energy_pj, leakage_mw, area_mm2}`
  - `SRAMParams` dataclass

- [ ] **Step 1: Write failing tests**

```python
# tests/test_destiny_adapter.py
from src.energy.destiny_adapter import DestinyAdapter, SRAMParams

def test_generate_sram_config():
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    cfg = adapter.generate_config(tech_nm=28, capacity_kb=256, banks=32, word_bits=128)
    assert "-MemoryCellInputFile:" in cfg
    assert "SRAM" in cfg
    assert "-Capacity (KB): 256" in cfg

def test_parse_output():
    # Simulated DESTINY output
    sample = """
 - Read Latency (ns): 0.534
 - Write Latency (ns): 0.612
 - Read Dynamic Energy (pJ): 3.21
 - Write Dynamic Energy (pJ): 3.85
 - Leakage Power (mW): 12.5
 - Area (mm2): 0.42
"""
    adapter = DestinyAdapter(destiny_dir="/home/Destiny-Memory-simulator")
    params = adapter.parse_output(sample)
    assert isinstance(params, SRAMParams)
    assert abs(params.read_latency_ns - 0.534) < 0.01
    assert abs(params.read_energy_pj - 3.21) < 0.01
    assert abs(params.leakage_mw - 12.5) < 0.01

def test_default_params():
    params = SRAMParams()
    assert params.read_energy_pj > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_destiny_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement destiny_adapter.py**

```python
# src/energy/__init__.py
# empty

# src/energy/destiny_adapter.py
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class SRAMParams:
    read_latency_ns: float = 0.534
    write_latency_ns: float = 0.612
    read_energy_pj: float = 3.2
    write_energy_pj: float = 3.8
    leakage_mw: float = 12.5
    area_mm2: float = 0.42
    tech_nm: int = 28
    capacity_kb: int = 256
    banks: int = 32


class DestinyAdapter:
    def __init__(self, destiny_dir: str = "/home/Destiny-Memory-simulator"):
        self.destiny_dir = destiny_dir
        self.cell_file = os.path.join(destiny_dir, "config", "3D", "SRAM", "SRAM.cell")

    def generate_config(self, tech_nm: int = 28, capacity_kb: int = 256,
                        banks: int = 32, word_bits: int = 128) -> str:
        cfg = f"""-DesignTarget: cache

-CacheAccessMode: Normal
-Associativity (for cache only): 1

-ProcessNode: {tech_nm}

-Capacity (KB): {capacity_kb}
-WordWidth (bit): {word_bits}

-DeviceRoadmap: HP

-LocalWireType: LocalAggressive
-LocalWireRepeaterType: RepeatedNone
-LocalWireUseLowSwing: No

-GlobalWireType: GlobalAggressive
-GlobalWireRepeaterType: RepeatedNone
-GlobalWireUseLowSwing: No

-Routing: H-tree

-InternalSensing: true

-MemoryCellInputFile: {self.cell_file}

-Temperature (K): 350

-OptimizationTarget: ReadLatency
-EnablePruning: Yes

-BufferDesignOptimization: latency

-StackedDieCount: 1
-LocalTSVProjection: 0
-GlobalTSVProjection: 0
-TSVRedundancy: 1.0
"""
        return cfg

    def run(self, tech_nm: int = 28, capacity_kb: int = 256,
            banks: int = 32, word_bits: int = 128) -> SRAMParams:
        cfg_content = self.generate_config(tech_nm, capacity_kb, banks, word_bits)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False,
                                         dir=self.destiny_dir) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            exe = os.path.join(self.destiny_dir, "destiny")
            if not os.path.exists(exe):
                return SRAMParams(tech_nm=tech_nm, capacity_kb=capacity_kb, banks=banks)

            result = subprocess.run(
                [exe, cfg_path],
                capture_output=True, text=True, timeout=120,
                cwd=self.destiny_dir,
            )
            params = self.parse_output(result.stdout)
            params.tech_nm = tech_nm
            params.capacity_kb = capacity_kb
            params.banks = banks
            return params
        finally:
            os.unlink(cfg_path)

    def parse_output(self, output: str) -> SRAMParams:
        params = SRAMParams()

        patterns = {
            "read_latency_ns": r"Read Latency.*?(\d+\.?\d*)",
            "write_latency_ns": r"Write Latency.*?(\d+\.?\d*)",
            "read_energy_pj": r"Read Dynamic Energy.*?(\d+\.?\d*)",
            "write_energy_pj": r"Write Dynamic Energy.*?(\d+\.?\d*)",
            "leakage_mw": r"Leakage Power.*?(\d+\.?\d*)",
            "area_mm2": r"Area.*?(\d+\.?\d*)",
        }

        for field, pattern in patterns.items():
            m = re.search(pattern, output)
            if m:
                setattr(params, field, float(m.group(1)))

        return params
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_destiny_adapter.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: DESTINY adapter for SRAM macro parameters"
```

---

### Task 10: LLM Attention Trace Generator

**Files:**
- Create: `src/tracegen/gen_attention_trace.py`
- Create: `tests/test_gen_attention.py`

**Interfaces:**
- Consumes: `TraceCommand`, `OpCode` from trace_ir; `SRAMPIMConfig` from config
- Produces:
  - `gen_attention_trace(batch: int, seq_len: int, num_heads: int, d_head: int, sram_config: SRAMPIMConfig, precision: str, mode: str, stage: str) -> list[TraceCommand]`
  - `stage` can be "summarization" (prefill) or "generation" (decode)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gen_attention.py
from src.tracegen.gen_attention_trace import gen_attention_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig

def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)

def test_attention_has_score_softmax_context():
    cmds = gen_attention_trace(
        batch=1, seq_len=128, num_heads=8, d_head=64,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    ops = [c.op for c in cmds]
    assert OpCode.PIM_MAC in ops     # score = Q @ K^T
    assert OpCode.PIM_NL in ops      # softmax
    assert OpCode.PIM_REDUCE in ops  # reduction

def test_attention_qkv_load():
    cmds = gen_attention_trace(
        batch=1, seq_len=64, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="cold_start", stage="generation"
    )
    loads = [c for c in cmds if c.op == OpCode.DMA_LOAD]
    # Should load Q, K, V at minimum
    assert len(loads) >= 3

def test_attention_generation_kv_scales_with_seq():
    short = gen_attention_trace(
        batch=1, seq_len=64, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    long = gen_attention_trace(
        batch=1, seq_len=256, num_heads=4, d_head=32,
        sram_config=make_config(), precision="int8",
        mode="warm_resident", stage="generation"
    )
    short_kv_bytes = sum(c.bytes for c in short if c.op == OpCode.DMA_LOAD and "K" in c.object_id or "V" in c.object_id)
    long_kv_bytes = sum(c.bytes for c in long if c.op == OpCode.DMA_LOAD and "K" in c.object_id or "V" in c.object_id)
    assert long_kv_bytes > short_kv_bytes
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_attention.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gen_attention_trace.py**

```python
# src/tracegen/gen_attention_trace.py
import math
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4, "fp32": 4}.get(precision, 1)


def gen_attention_trace(batch: int, seq_len: int, num_heads: int, d_head: int,
                        sram_config: SRAMPIMConfig, precision: str = "int8",
                        mode: str = "warm_resident", stage: str = "generation") -> list:
    dbyte = _precision_bytes(precision)
    cmds = []
    cmd_id = 0
    dram_addr = 0x2000_0000

    if stage == "generation":
        q_len = 1  # decode: single token query
        kv_len = seq_len
    else:
        q_len = seq_len  # prefill: full sequence
        kv_len = seq_len

    for h in range(num_heads):
        # Allocate and load Q (batch x q_len x d_head)
        q_id = f"Q_h{h}"
        q_bytes = batch * q_len * d_head * dbyte
        q_tile = h % sram_config.tiles
        alloc_q = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, q_id, "-",
                    f"SRAM:T{q_tile}:B0-1", q_bytes, {"type": "ACTIVATION"}, []))
        cmd_id += 1

        load_q = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, q_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{q_tile}:B0-1",
                    q_bytes, {}, [alloc_q]))
        cmd_id += 1
        dram_addr += q_bytes

        # Load K (batch x kv_len x d_head)
        k_id = f"K_h{h}"
        k_bytes = batch * kv_len * d_head * dbyte
        k_tile = (h + 1) % sram_config.tiles
        alloc_k = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, k_id, "-",
                    f"SRAM:T{k_tile}:B0-3", k_bytes, {"type": "STATE"}, []))
        cmd_id += 1

        load_k = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, k_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{k_tile}:B0-3",
                    k_bytes, {}, [alloc_k]))
        cmd_id += 1
        dram_addr += k_bytes

        # Load V (batch x kv_len x d_head)
        v_id = f"V_h{h}"
        v_bytes = batch * kv_len * d_head * dbyte
        v_tile = (h + 2) % sram_config.tiles
        alloc_v = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, v_id, "-",
                    f"SRAM:T{v_tile}:B0-3", v_bytes, {"type": "STATE"}, []))
        cmd_id += 1

        load_v = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_LOAD, v_id,
                    f"DRAM:0x{dram_addr:X}", f"SRAM:T{v_tile}:B0-3",
                    v_bytes, {}, [alloc_v]))
        cmd_id += 1
        dram_addr += v_bytes

        # Score = Q @ K^T: (q_len x kv_len)
        score_id = f"Score_h{h}"
        mac_count = batch * q_len * kv_len * d_head
        score_tile = (h + 3) % sram_config.tiles
        mac_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, score_id,
                    f"{q_id},{k_id}", f"SRAM:T{score_tile}:B0-1",
                    0, {"mac_count": mac_count}, [load_q, load_k]))
        cmd_id += 1

        # Softmax on score
        softmax_id = f"Softmax_h{h}"
        nl_count = batch * q_len * kv_len
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_NL, softmax_id,
                    score_id, f"SRAM:T{score_tile}:B0-1",
                    0, {"count": nl_count, "op": "softmax"}, [mac_cmd]))
        cmd_id += 1
        softmax_cmd = cmd_id - 1

        # Reduce (for numerical stability / partial sum)
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_REDUCE, softmax_id,
                    softmax_id, f"SRAM:T{score_tile}:B0-1",
                    0, {"count": nl_count}, [softmax_cmd]))
        cmd_id += 1
        reduce_cmd = cmd_id - 1

        # Context = Softmax(Score) @ V: (q_len x d_head)
        context_id = f"Context_h{h}"
        ctx_mac_count = batch * q_len * d_head * kv_len
        ctx_cmd = cmd_id
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_MAC, context_id,
                    f"{softmax_id},{v_id}", f"SRAM:T{score_tile}:B2-3",
                    0, {"mac_count": ctx_mac_count}, [reduce_cmd, load_v]))
        cmd_id += 1

        # Store context output
        ctx_bytes = batch * q_len * d_head * dbyte
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, context_id,
                    f"SRAM:T{score_tile}:B2-3", f"DRAM:0x{dram_addr:X}",
                    ctx_bytes, {}, [ctx_cmd]))
        cmd_id += 1
        dram_addr += ctx_bytes

    return cmds
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_attention.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Add attention workload to main.py**

Add to `main.py` argument parser: `--num-heads`, `--d-head`, `--seq-len`, `--stage`
Add attention branch in workload dispatch.

- [ ] **Step 6: Run full test suite**

Run: `cd /home/sram_pim && python -m pytest tests/ -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: LLM attention trace generator (prefill + decode)"
```

---

### Task 11: DRAMsim3 Integration

**Files:**
- Create: `src/dramsim3_wrapper.py`
- Modify: `src/dram_model.py` — add DRAMsim3 backend option
- Create: `tests/test_dramsim3_wrapper.py`

**Interfaces:**
- Consumes: `DRAMConfig` from config; DRAMsim3 at `/home/DRAMsim3`
- Produces:
  - `DRAMsim3Wrapper.__init__(self, config_file: str, dramsim3_dir: str)`
  - `DRAMsim3Wrapper.issue_read(self, addr: int) -> None`
  - `DRAMsim3Wrapper.issue_write(self, addr: int) -> None`
  - `DRAMsim3Wrapper.tick(self) -> list[int]` — returns completed addresses
  - `DRAMsim3Wrapper.get_cycles(self) -> int`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dramsim3_wrapper.py
import os
import pytest
from src.dramsim3_wrapper import DRAMsim3Wrapper, generate_dram_trace

def test_generate_trace():
    trace = generate_dram_trace(
        dram_addr=0x1000, nbytes=256, burst_bytes=64, is_write=False
    )
    assert len(trace) == 4  # 256 / 64 = 4 reads
    assert all(line.endswith("READ") for line in trace)

def test_generate_write_trace():
    trace = generate_dram_trace(
        dram_addr=0x2000, nbytes=128, burst_bytes=64, is_write=True
    )
    assert len(trace) == 2
    assert all(line.endswith("WRITE") for line in trace)

@pytest.mark.skipif(not os.path.exists("/home/DRAMsim3/dramsim3main.out"),
                    reason="DRAMsim3 binary not available")
def test_dramsim3_run():
    wrapper = DRAMsim3Wrapper(
        config_file="/home/DRAMsim3/configs/DDR4_8Gb_x8_2400.ini",
        dramsim3_dir="/home/DRAMsim3"
    )
    cycles = wrapper.run_trace_file(
        [f"0x{0x1000 + i*64:X} READ" for i in range(4)]
    )
    assert cycles > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_dramsim3_wrapper.py -v`
Expected: FAIL (at least for non-skipif tests)

- [ ] **Step 3: Implement dramsim3_wrapper.py**

```python
# src/dramsim3_wrapper.py
import os
import subprocess
import tempfile
import re


def generate_dram_trace(dram_addr: int, nbytes: int, burst_bytes: int = 64,
                        is_write: bool = False) -> list:
    n_bursts = max(1, nbytes // burst_bytes)
    op = "WRITE" if is_write else "READ"
    trace = []
    for i in range(n_bursts):
        addr = dram_addr + i * burst_bytes
        trace.append(f"0x{addr:X} {op}")
    return trace


class DRAMsim3Wrapper:
    def __init__(self, config_file: str, dramsim3_dir: str = "/home/DRAMsim3"):
        self.config_file = config_file
        self.dramsim3_dir = dramsim3_dir
        self.exe = os.path.join(dramsim3_dir, "dramsim3main.out")

    def run_trace_file(self, trace_lines: list) -> int:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.trace',
                                         delete=False, dir="/tmp") as f:
            for line in trace_lines:
                f.write(line + "\n")
            trace_path = f.name

        try:
            result = subprocess.run(
                [self.exe, self.config_file, trace_path],
                capture_output=True, text=True, timeout=60,
                cwd=self.dramsim3_dir,
            )
            # Parse cycles from output
            cycles = 0
            for line in result.stdout.split("\n"):
                m = re.search(r"num_cycles\s*[:=]\s*(\d+)", line)
                if m:
                    cycles = int(m.group(1))
            if cycles == 0:
                # Fallback: count lines as approximate
                m = re.search(r"(\d+)", result.stdout.split("\n")[-2] if result.stdout.strip() else "0")
                if m:
                    cycles = int(m.group(1))
            return cycles
        finally:
            os.unlink(trace_path)

    def get_latency_for_transfer(self, dram_addr: int, nbytes: int,
                                  is_write: bool = False, burst_bytes: int = 64) -> int:
        trace = generate_dram_trace(dram_addr, nbytes, burst_bytes, is_write)
        return self.run_trace_file(trace)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_dramsim3_wrapper.py -v`
Expected: First 2 PASSED, 3rd SKIPPED if binary missing

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: DRAMsim3 wrapper for validated DRAM timing"
```

---

### Task 12: SSM/Mamba and Transformer FFN Trace Generators

**Files:**
- Create: `src/tracegen/gen_ssm_trace.py`
- Create: `src/tracegen/gen_ffn_trace.py`
- Create: `tests/test_gen_ssm.py`
- Create: `tests/test_gen_ffn.py`

**Interfaces:**
- Consumes: `TraceCommand`, `OpCode`, `SRAMPIMConfig`
- Produces:
  - `gen_ssm_trace(batch, seq_len, d_model, d_state, sram_config, precision, mode) -> list[TraceCommand]`
  - `gen_ffn_trace(batch, seq_len, d_model, d_ff, sram_config, precision, mode) -> list[TraceCommand]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gen_ssm.py
from src.tracegen.gen_ssm_trace import gen_ssm_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig

def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)

def test_ssm_has_sequential_dependency():
    cmds = gen_ssm_trace(batch=1, seq_len=8, d_model=64, d_state=16,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    ew_ops = [c for c in cmds if c.op == OpCode.PIM_EW_OP]
    # Each timestep depends on previous
    for i in range(1, len(ew_ops)):
        assert any(d < ew_ops[i].cmd_id for d in ew_ops[i].deps)

def test_ssm_state_persistence():
    cmds = gen_ssm_trace(batch=1, seq_len=4, d_model=32, d_state=8,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    # Should have state objects (h_t)
    state_ops = [c for c in cmds if "h_" in c.object_id]
    assert len(state_ops) > 0
```

```python
# tests/test_gen_ffn.py
from src.tracegen.gen_ffn_trace import gen_ffn_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig

def make_config():
    return SRAMPIMConfig(tiles=8, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=1024)

def test_ffn_two_gemms():
    cmds = gen_ffn_trace(batch=1, seq_len=128, d_model=256, d_ff=1024,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    macs = [c for c in cmds if c.op == OpCode.PIM_MAC]
    # At least 2 GEMM layers (up-project + down-project)
    assert len(macs) >= 2

def test_ffn_has_activation():
    cmds = gen_ffn_trace(batch=1, seq_len=64, d_model=128, d_ff=512,
                         sram_config=make_config(), precision="int8", mode="warm_resident")
    nl_ops = [c for c in cmds if c.op == OpCode.PIM_NL]
    assert len(nl_ops) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_ssm.py tests/test_gen_ffn.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gen_ssm_trace.py**

```python
# src/tracegen/gen_ssm_trace.py
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def _precision_bytes(precision: str) -> int:
    return {"int8": 1, "int16": 2, "fp16": 2, "int32": 4}.get(precision, 1)


def gen_ssm_trace(batch: int, seq_len: int, d_model: int, d_state: int,
                  sram_config: SRAMPIMConfig, precision: str = "int8",
                  mode: str = "warm_resident") -> list:
    dbyte = _precision_bytes(precision)
    cmds = []
    cmd_id = 0
    dram_addr = 0x5000_0000

    # Load A, B, C, D parameters (warm: preloaded, cold: DMA_LOAD)
    param_ids = {}
    for name in ["A", "B", "C", "D"]:
        p_id = f"SSM_{name}"
        p_bytes = d_model * d_state * dbyte if name != "D" else d_model * dbyte
        tile = cmd_id % sram_config.tiles
        preloaded = mode == "warm_resident"
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

    # Initialize h_0 state
    h_bytes = batch * d_model * d_state * dbyte
    h_prev_id = "h_0"
    tile = cmd_id % sram_config.tiles
    cmds.append(TraceCommand(cmd_id, OpCode.SRAM_ALLOC, h_prev_id, "-",
                f"SRAM:T{tile}:B0-3", h_bytes, {"type": "STATE"}, []))
    h_alloc = cmd_id
    cmd_id += 1

    prev_dep = h_alloc

    # Sequential scan: h_t = A_t * h_{t-1} + B_t * u_t
    for t in range(seq_len):
        u_id = f"u_{t}"
        u_bytes = batch * d_model * dbyte
        u_tile = cmd_id % sram_config.tiles

        # Load input u_t
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

        # A * h_{t-1} (element-wise)
        ah_id = f"Ah_{t}"
        count = batch * d_model * d_state
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, ah_id,
                    f"SSM_A,{h_prev_id}", f"SRAM:T{u_tile}:B1-1",
                    0, {"count": count, "op": "MUL"},
                    [param_ids["A"], prev_dep]))
        ah_cmd = cmd_id
        cmd_id += 1

        # B * u_t
        bu_id = f"Bu_{t}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, bu_id,
                    f"SSM_B,{u_id}", f"SRAM:T{u_tile}:B2-2",
                    0, {"count": count, "op": "MUL"},
                    [param_ids["B"], u_load]))
        bu_cmd = cmd_id
        cmd_id += 1

        # h_t = Ah + Bu
        h_id = f"h_{t+1}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, h_id,
                    f"{ah_id},{bu_id}", f"SRAM:T{u_tile}:B3-3",
                    0, {"count": count, "op": "ADD"},
                    [ah_cmd, bu_cmd]))
        prev_dep = cmd_id
        h_prev_id = h_id
        cmd_id += 1

        # y_t = C * h_t + D * u_t
        y_id = f"y_{t}"
        cmds.append(TraceCommand(cmd_id, OpCode.PIM_EW_OP, y_id,
                    f"SSM_C,{h_id}", f"SRAM:T{u_tile}:B4-4",
                    0, {"count": batch * d_model, "op": "MUL_ADD"},
                    [param_ids["C"], prev_dep, param_ids["D"], u_load]))
        y_cmd = cmd_id
        cmd_id += 1

        # Store y_t
        y_bytes = batch * d_model * dbyte
        cmds.append(TraceCommand(cmd_id, OpCode.DMA_STORE, y_id,
                    f"SRAM:T{u_tile}:B4-4", f"DRAM:0x{dram_addr:X}",
                    y_bytes, {}, [y_cmd]))
        cmd_id += 1
        dram_addr += y_bytes

    return cmds
```

- [ ] **Step 4: Implement gen_ffn_trace.py**

```python
# src/tracegen/gen_ffn_trace.py
from src.tracegen.gen_gemm_trace import gen_gemm_trace, _precision_bytes
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def gen_ffn_trace(batch: int, seq_len: int, d_model: int, d_ff: int,
                  sram_config: SRAMPIMConfig, precision: str = "int8",
                  mode: str = "warm_resident") -> list:
    dbyte = _precision_bytes(precision)
    M = batch * seq_len
    Tm = min(64, M)
    Tk = min(128, d_model)

    # Up-projection: X[M, d_model] @ W1[d_model, d_ff] -> H[M, d_ff]
    up_cmds = gen_gemm_trace(M=M, N=d_ff, K=d_model,
                             Tm=Tm, Tn=min(64, d_ff), Tk=Tk,
                             sram_config=sram_config, precision=precision, mode=mode)

    # Offset cmd_ids
    max_id = max(c.cmd_id for c in up_cmds) + 1

    # Activation (GeLU/SiLU)
    nl_id = f"FFN_act"
    nl_count = M * d_ff
    nl_tile = 0
    last_up_mac = max(c.cmd_id for c in up_cmds if c.op == OpCode.PIM_MAC)
    nl_cmd = TraceCommand(max_id, OpCode.PIM_NL, nl_id,
                          f"Y_m0_n0", f"SRAM:T{nl_tile}:B0-1",
                          0, {"count": nl_count, "op": "gelu"},
                          [last_up_mac])
    max_id += 1

    # Down-projection: H[M, d_ff] @ W2[d_ff, d_model] -> O[M, d_model]
    down_cmds = gen_gemm_trace(M=M, N=d_model, K=d_ff,
                               Tm=Tm, Tn=min(64, d_model), Tk=min(128, d_ff),
                               sram_config=sram_config, precision=precision, mode=mode)

    # Remap down cmd_ids to avoid collision
    id_offset = max_id
    id_map = {}
    for c in down_cmds:
        old_id = c.cmd_id
        c.cmd_id = old_id + id_offset
        id_map[old_id] = c.cmd_id
        c.deps = [id_map.get(d, d + id_offset) for d in c.deps]

    # First down command depends on activation
    if down_cmds:
        down_cmds[0].deps.append(nl_cmd.cmd_id)

    # Rename objects to avoid collision
    for c in down_cmds:
        if c.object_id.startswith("W_"):
            c.object_id = "FFN_down_" + c.object_id
        elif c.object_id.startswith("X_"):
            c.object_id = "FFN_down_" + c.object_id
        elif c.object_id.startswith("Y_"):
            c.object_id = "FFN_out_" + c.object_id

    return up_cmds + [nl_cmd] + down_cmds
```

- [ ] **Step 5: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_ssm.py tests/test_gen_ffn.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run full suite**

Run: `cd /home/sram_pim && python -m pytest tests/ -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: SSM/Mamba and Transformer FFN trace generators"
```

---

### Task 13: Full Transformer Layer and Multi-Layer Pipeline

**Files:**
- Create: `src/tracegen/gen_transformer_trace.py`
- Modify: `main.py` — add transformer workload
- Create: `tests/test_gen_transformer.py`

**Interfaces:**
- Consumes: `gen_attention_trace`, `gen_ffn_trace`, `gen_gemm_trace`
- Produces:
  - `gen_transformer_trace(model_config: dict, sram_config: SRAMPIMConfig, precision: str, mode: str, stage: str) -> list[TraceCommand]`
  - model_config: `{name, ndec, hdim, num_heads, d_head, ff_scale, batch, seq_len, gen_len}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gen_transformer.py
from src.tracegen.gen_transformer_trace import gen_transformer_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig
from src.simulator import Simulator
from src.config import SimConfig, SystemConfig

def make_config():
    return SRAMPIMConfig(tiles=16, banks_per_tile=8, bank_capacity_kb=16, total_capacity_kb=2048)

def test_single_layer_decode():
    model = {
        "name": "test", "ndec": 1, "hdim": 128, "num_heads": 4,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 64, "gen_len": 1,
    }
    cmds = gen_transformer_trace(model, make_config(), "int8", "warm_resident", "generation")
    ops = set(c.op for c in cmds)
    # Should have attention (PIM_MAC, PIM_NL) and FFN (PIM_MAC, PIM_NL)
    assert OpCode.PIM_MAC in ops
    assert OpCode.PIM_NL in ops
    assert OpCode.DMA_LOAD in ops

def test_multi_layer():
    model = {
        "name": "test", "ndec": 2, "hdim": 64, "num_heads": 2,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 32, "gen_len": 1,
    }
    cmds = gen_transformer_trace(model, make_config(), "int8", "warm_resident", "generation")
    # 2 layers should produce more commands than 1
    single = {
        "name": "test", "ndec": 1, "hdim": 64, "num_heads": 2,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 32, "gen_len": 1,
    }
    cmds_1 = gen_transformer_trace(single, make_config(), "int8", "warm_resident", "generation")
    assert len(cmds) > len(cmds_1)

def test_transformer_e2e_simulation():
    model = {
        "name": "tiny", "ndec": 1, "hdim": 64, "num_heads": 2,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 32, "gen_len": 1,
    }
    sram_cfg = make_config()
    cmds = gen_transformer_trace(model, sram_cfg, "int8", "warm_resident", "generation")
    config = SimConfig(
        system=SystemConfig(frequency_hz=1_000_000_000, mode="warm_resident"),
        sram_pim=sram_cfg,
    )
    sim = Simulator(config)
    sim.load_trace(cmds)
    report = sim.run()
    assert report["latency"]["total_cycles"] > 0
    assert report["energy"]["total_pj"] > 0
    assert report["correctness"]["illegal_read_unresident_object"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_transformer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gen_transformer_trace.py**

```python
# src/tracegen/gen_transformer_trace.py
from src.tracegen.gen_attention_trace import gen_attention_trace
from src.tracegen.gen_ffn_trace import gen_ffn_trace
from src.tracegen.gen_gemm_trace import _precision_bytes
from src.trace_ir import TraceCommand, OpCode
from src.config import SRAMPIMConfig


def gen_transformer_trace(model_config: dict, sram_config: SRAMPIMConfig,
                          precision: str = "int8", mode: str = "warm_resident",
                          stage: str = "generation") -> list:
    ndec = model_config["ndec"]
    hdim = model_config["hdim"]
    num_heads = model_config["num_heads"]
    d_head = model_config["d_head"]
    ff_scale = model_config["ff_scale"]
    batch = model_config["batch"]
    seq_len = model_config["seq_len"]
    d_ff = int(hdim * ff_scale)

    all_cmds = []
    global_id = 0

    for layer in range(ndec):
        # QKV projection: [batch*q_len, hdim] @ [hdim, 3*hdim] -> [batch*q_len, 3*hdim]
        q_len = 1 if stage == "generation" else seq_len
        qkv_cmds = gen_attention_trace(
            batch=batch, seq_len=seq_len, num_heads=num_heads,
            d_head=d_head, sram_config=sram_config,
            precision=precision, mode=mode, stage=stage
        )

        # Remap IDs
        id_map = {}
        for c in qkv_cmds:
            old_id = c.cmd_id
            c.cmd_id = global_id
            id_map[old_id] = global_id
            c.deps = [id_map.get(d, d) for d in c.deps]
            # Prefix object IDs with layer
            c.object_id = f"L{layer}_{c.object_id}"
            global_id += 1

        # Connect to previous layer's output
        if all_cmds:
            last_store = max(c.cmd_id for c in all_cmds if c.op == OpCode.DMA_STORE)
            qkv_cmds[0].deps.append(last_store)

        all_cmds.extend(qkv_cmds)

        # FFN
        ffn_cmds = gen_ffn_trace(
            batch=batch, seq_len=q_len, d_model=hdim, d_ff=d_ff,
            sram_config=sram_config, precision=precision, mode=mode
        )

        # Remap FFN IDs
        ffn_id_map = {}
        last_attn = max(c.cmd_id for c in all_cmds)
        for c in ffn_cmds:
            old_id = c.cmd_id
            c.cmd_id = global_id
            ffn_id_map[old_id] = global_id
            c.deps = [ffn_id_map.get(d, d) for d in c.deps]
            c.object_id = f"L{layer}_FFN_{c.object_id}"
            global_id += 1

        # FFN depends on attention output
        if ffn_cmds:
            ffn_cmds[0].deps.append(last_attn)

        all_cmds.extend(ffn_cmds)

    return all_cmds
```

- [ ] **Step 4: Add transformer workload to main.py**

Add argument `--model-config` and `--stage` to CLI. Add transformer dispatch:

```python
# Add to main.py argument parser
parser.add_argument("--ndec", type=int, default=1)
parser.add_argument("--hdim", type=int, default=256)
parser.add_argument("--num-heads", type=int, default=8)
parser.add_argument("--d-head", type=int, default=32)
parser.add_argument("--ff-scale", type=float, default=4.0)
parser.add_argument("--seq-len", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--stage", type=str, default="generation")

# Add to workload dispatch
elif args.workload == "transformer":
    model_config = {
        "name": "custom", "ndec": args.ndec, "hdim": args.hdim,
        "num_heads": args.num_heads, "d_head": args.d_head,
        "ff_scale": args.ff_scale, "batch": args.batch,
        "seq_len": args.seq_len, "gen_len": 1,
    }
    from src.tracegen.gen_transformer_trace import gen_transformer_trace
    cmds = gen_transformer_trace(model_config, config.sram_pim, "int8", mode, args.stage)
```

- [ ] **Step 5: Run tests**

Run: `cd /home/sram_pim && python -m pytest tests/test_gen_transformer.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Run full test suite and CLI smoke test**

Run: `cd /home/sram_pim && python -m pytest tests/ -v && python main.py --workload transformer --ndec 2 --hdim 128 --num-heads 4 --seq-len 64 --mode warm_resident --stage generation`
Expected: ALL PASSED, YAML report printed

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: full Transformer layer trace generator with multi-layer pipeline"
```
