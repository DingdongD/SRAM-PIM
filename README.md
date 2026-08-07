# SRAM-PIM

Trace-driven simulation tools for stacked-SRAM and near-SRAM neural-processing
architectures.

The repository now includes a strict Python C-model that compares centralized
SRAM-on-logic systolic NPUs with bank-local near-SRAM systolic NPUs using the
same external systolic backend. It models SRAM banks, vertical links, global or
local buffers, NoC traffic, systolic array cycles, and cross-bank reduction as
explicit resources.

See [`docs/strict_cmodel.md`](docs/strict_cmodel.md) for the architecture model,
SCALE-Sim integration, fail-closed behavior, configuration, and commands.

The original SRAM-PIM trace simulator remains available through `main.py`.
The strict architecture comparison uses:

```bash
python cmodel_main.py --config <config.yaml> --M <M> --N <N> --K <K>
```
