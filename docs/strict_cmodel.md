# Strict stacked-SRAM NPU C-model

This package adds a fail-closed, cycle-level Python C-model for comparing two
architectures that use the same systolic-array model:

1. **Centralized SRAM-on-logic NPU** — stacked SRAM banks feed centralized
   arrays through vertical links and a global buffer.
2. **Bank-local near-SRAM NPU** — each SRAM bank group owns a local systolic
   array and local buffer. The model supports output-channel (`N`) partitioning
   with activation multicast and reduction-dimension (`K`) partitioning with
   explicit cross-group reduction.

The distinction is made by data ownership and resource paths, not by using
different MAC formulas.

## Accuracy contract

The C-model provides:

- bit-exact integer GEMM reference semantics;
- a cycle-tagged micro-operation DAG;
- explicit SRAM-bank, vertical-link, buffer, NoC, array, and reduction resources;
- deterministic resource contention and critical-path reconstruction;
- strict external backend contracts.

It is cycle-accurate under the declared architectural model. Absolute silicon
accuracy still requires calibration against RTL, synthesis, SRAM macros, and
physical implementation.

## No fallback behavior

Production runs are fail-closed:

- a missing SCALE-Sim installation fails;
- a missing or ambiguous compute report fails;
- missing input/filter SRAM traces fail;
- unsupported YAML fields fail;
- an incompatible recorded result fails;
- unknown resources or invalid mappings fail;
- tensors that exceed stacked-SRAM capacity fail because DRAM/spill is not yet
  part of the strict model;
- tiles that exceed the declared global/local buffer fail instead of being
  silently retiled.

The `recorded` backend is an explicitly selected immutable replay backend for
regression and offline reproducibility. It is never selected automatically.

## SCALE-Sim integration

The adapter runs the documented module CLI:

```bash
python3 -m scalesim.scale \
  -c /path/to/scalesim.cfg \
  -t generated_topology.csv \
  -p output_directory
```

A GEMM `A[M,K] @ B[K,N]` is emitted as a 1x1-convolution-equivalent topology
with `M` spatial positions, `K` channels, and `N` output filters. The adapter
requires one unambiguous compute report, cycle-tagged SRAM IFMAP traces,
cycle-tagged SRAM filter traces, and accepts optional OFMAP traces.

The compute report provides the nominal array cycle window. SRAM traces are
lowered into explicit bank/link/buffer delivery chains. An array cycle cannot
execute until its operand delivery events have completed, so bank and link
contention create real array stalls rather than an added summary penalty.

## Event paths

### Centralized placement

```text
SRAM_READ -> VLINK_SEND -> GBUF_WRITE -> GBUF_READ -> ARRAY_STEP
ARRAY_STEP -> GBUF_WRITE -> GBUF_READ -> VLINK_SEND -> SRAM_WRITE
```

### Bank-local `N` partition

```text
shared INPUT: SRAM_READ -> VLINK_SEND -> NOC_MULTICAST
              -> LOCAL_BUFFER_WRITE -> LOCAL_BUFFER_READ -> ARRAY_STEP
local WEIGHT: LOCAL_SRAM_READ -> VLINK_SEND -> LOCAL_BUFFER_WRITE
              -> LOCAL_BUFFER_READ -> ARRAY_STEP
```

Output columns are independent, so `N` partitioning concatenates group outputs
without arithmetic reduction.

### Bank-local `K` partition

```text
LOCAL_SRAM_READ -> VLINK_SEND -> LOCAL_ARRAY -> LOCAL_PSUM
all groups ready -> NOC_SEND -> GLOBAL_REDUCE -> SRAM_WRITE
```

The same systolic backend is invoked for every centralized or local tile.

## Configuration

Two examples are provided:

- `configs/cmodel_centralized.yaml`
- `configs/cmodel_bank_local.yaml`

The important orthogonal fields are:

```yaml
architecture:
  compute_placement: centralized_logic  # or bank_local_logic
  arrays:
    rows: 128
    cols: 128
    dataflow: ws
```

For an apples-to-apples comparison, keep array rows, columns, dataflow, operand
precision, and SCALE-Sim configuration identical. Change only the placement,
array/group count under study, bank ownership, link resources, and mapping.

## Run one architecture

```bash
python cmodel_main.py \
  --config configs/cmodel_centralized.yaml \
  --M 128 --N 128 --K 128 \
  --output results/centralized.json
```

Add `--include-events` to retain the complete event schedule.

## Compare both architectures

```bash
python compare_cmodel.py \
  --centralized-config configs/cmodel_centralized.yaml \
  --bank-local-config configs/cmodel_bank_local.yaml \
  --M 128 --N 128 --K 128 \
  --output results/comparison.json
```

The comparison command verifies the common compute signature and backend
identity/version before reporting a cycle ratio.

## Diagnosis

For every scheduled event, the simulator records dependency-ready,
release/prefetch-ready, resource-ready, start, and finish cycles plus the event
that blocked the selected resource slot.

The terminal event is traced backward through dependency and resource blockers
to reconstruct a deterministic critical path. The report attributes critical
path cycles to resources such as:

```text
systolic_array
sram_read / sram_read_queue
vertical_link / vertical_link_queue
noc / noc_queue
reduction / reduction_queue
```

This separates resource utilization from actual end-to-end bottlenecks.

## Current boundary

This first implementation covers strict GEMM lowering and the stacked-SRAM
on-chip hierarchy. DRAM/Ramulator2 and detailed BookSim co-simulation should be
added as additional required resources, not as analytical fallbacks. The public
interfaces are separated so those backends can be integrated without changing
the functional IR or the architecture distinction.
