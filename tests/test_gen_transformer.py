from src.tracegen.gen_transformer_trace import gen_transformer_trace
from src.trace_ir import OpCode
from src.config import SRAMPIMConfig, SimConfig, SystemConfig
from src.simulator import Simulator


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


def test_cmd_ids_unique():
    model = {
        "name": "test", "ndec": 2, "hdim": 64, "num_heads": 2,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 32, "gen_len": 1,
    }
    cmds = gen_transformer_trace(model, make_config(), "int8", "warm_resident", "generation")
    ids = [c.cmd_id for c in cmds]
    assert len(ids) == len(set(ids)), "cmd_ids must be unique"


def test_deps_valid():
    """All dependency IDs should reference valid cmd_ids."""
    model = {
        "name": "test", "ndec": 2, "hdim": 64, "num_heads": 2,
        "d_head": 32, "ff_scale": 4, "batch": 1, "seq_len": 32, "gen_len": 1,
    }
    cmds = gen_transformer_trace(model, make_config(), "int8", "warm_resident", "generation")
    valid_ids = {c.cmd_id for c in cmds}
    for c in cmds:
        for dep in c.deps:
            assert dep in valid_ids, f"cmd {c.cmd_id} has invalid dep {dep}"
