# Strict stacked-SRAM NPU C-model

The strict C-model is the architecture-study path rooted at `cmodel_main.py`. It is separate from the legacy `main.py` simulator.

## Modeling boundary

The model uses one SCALE-Sim systolic backend for both compute placements. Architecture differences are expressed only through placement and resource paths:

- `centralized_logic`: stacked SRAM → vertical link → BookSim2 NoC → global buffer → systolic array.
- `bank_local_logic`: stacked SRAM → vertical link → bank-local buffer/array, with explicit multicast and cross-group reduction when required.

The strict path models:

- GEMM and grouped CONV lowering;
- Attention prefill and decode with KV-cache traffic;
- Softmax and LayerNorm vector/reduction pipelines;
- per-bank SRAM ports and capacity;
- vertical-link serialization;
- persistent BookSim2 packet/flit/VC/router timing;
- persistent Ramulator2 DRAM-controller timing;
- CACTI or DESTINY SRAM macro timing and energy import;
- deterministic event scheduling, request conservation, and critical-path reporting.

No analytical backend is selected when an external backend fails. Missing files, configuration fields, repository-version mismatches, protocol mismatches, capacity overflow, unsupported mappings, and simulator failures terminate the run.

## Fixed external versions

The committed configs require these revisions:

| Backend | Commit |
| --- | --- |
| SCALE-Sim | `9f98c4371055a54c75209c2e02b640b897550532` |
| Ramulator2 | `b30320bc9385b708e86b67ebb9f48858cc66d798` |
| BookSim2 | `28f43299f1706a3160ffac721ca461d74eb6e618` |
| CACTI | `1ffd8dfb10303d306ecd8d215320aea07651e878` |

Tracked modifications inside any of these repositories are rejected.

## Bootstrap

From the repository root, install the external tools once:

```bash
./scripts/bootstrap_strict_backends.sh "$PWD"
```

The script clones the exact commits, builds Ramulator2, BookSim2 and CACTI, builds the two persistent bridges, exports the Ramulator2 External configuration, checks that tracked third-party sources remain unchanged, and runs real backend smoke tests. Existing `third_party/scalesim`, `third_party/ramulator2`, `third_party/booksim2`, or `third_party/cacti` directories cause the script to fail instead of being reused implicitly.

## Run one architecture

```bash
python cmodel_main.py \
  --config configs/cmodel_centralized.yaml \
  --output results/gemm.json \
  gemm \
  --op-id gemm0 \
  --m 4096 --n 4096 --k 4096 \
  --input-bits 8 --weight-bits 8 --output-bits 32 --accumulator-bits 32
```

All operator arguments are explicit. There are no hidden operator defaults.

## Compare compute placement

```bash
python compare_cmodel.py \
  --centralized-config configs/cmodel_centralized.yaml \
  --bank-local-config configs/cmodel_bank_local.yaml \
  --output results/gemm_compare.json \
  gemm \
  --op-id gemm0 \
  --m 4096 --n 4096 --k 4096 \
  --input-bits 8 --weight-bits 8 --output-bits 32 --accumulator-bits 32
```

The comparison rejects configs that differ in mapping, address map, backends, simulation limits, array specification, SRAM, NoC, vector unit, reduction unit, or DRAM. The only allowed architecture differences are `name` and `compute_placement`.

## SCALE-Sim contract

SCALE-Sim is used only for the systolic compute demand schedule. The adapter requires the fixed output layout:

```text
<output>/<run_name>/COMPUTE_REPORT.csv
<output>/<run_name>/layer0/IFMAP_SRAM_TRACE.csv
<output>/<run_name>/layer0/FILTER_SRAM_TRACE.csv
<output>/<run_name>/layer0/OFMAP_SRAM_TRACE.csv
```

The SCALE-Sim run must report zero internal SRAM stall cycles. A non-zero stall count is rejected because accepting it would double-count SRAM stalls when the same demands are replayed through the C-model SRAM and interconnect resources.

## Persistent Ramulator2 contract

The C-model keeps one Ramulator2 process alive for the full simulation. Requests that Ramulator2 rejects because its queue is full remain pending in the C-model and are retried on later cycles. The backend reports its transaction size and request type IDs at startup; they must match the architecture config.

## Persistent BookSim2 contract

BookSim2 remains alive for the full simulation. Each accepted C-model NoC operation is represented by a multi-flit packet with an explicitly configured VC. The bridge directly advances BookSim2 routers, VC buffers, switch allocation, links, and credits cycle by cycle. At startup, BookSim2 reports node count, VC count, and traffic-class count; incompatible C-model configs are rejected.

## Energy boundary

CACTI or DESTINY provides SRAM macro access latency, cycle time, dynamic read/write energy, leakage, and area. The current report intentionally does not synthesize energy numbers for systolic compute, vector/reduction compute, NoC, DRAM, or vertical links. Those components are listed as unmodeled instead of being filled with analytical defaults.

## Validation

The unit suite checks all operator lowerings, centralized and bank-local mapping, K-partition reduction ordering, LayerNorm parameter ordering, persistent-backend protocols, repository cleanliness, SCALE-Sim zero-stall enforcement, CACTI parsing, configuration equivalence, and forbidden access patterns.

```bash
PYTHONPATH=. pytest -q tests/test_strict_cmodel.py tests/test_strict_backends.py
```

The GitHub Actions `external-backends` job additionally builds and executes the fixed real third-party tools.
