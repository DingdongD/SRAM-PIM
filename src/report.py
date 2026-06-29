import yaml
from src.config import SimConfig


def generate_report(result: dict, config: SimConfig) -> str:
    lat = result["latency"]
    report = {
        "simulation_config": {
            "mode": config.system.mode,
            "correctness_mode": config.system.correctness_mode,
            "frequency_hz": config.system.frequency_hz,
            "sram_tiles": config.sram_pim.tiles,
            "sram_total_kb": config.sram_pim.total_capacity_kb,
            "pim_mode": config.sram_pim.pim.mode,
            "sram_params_source": config.energy.sram.source,
        },
        "latency": {
            "total_cycles": lat["total_cycles"],
            "total_ns": lat.get("total_ns", 0),
            "dram_load_cycles": lat.get("dram_load_cycles", 0),
            "dram_store_cycles": lat.get("dram_store_cycles", 0),
            "pim_compute_cycles": lat.get("pim_compute_cycles", 0),
            "pim_reduce_cycles": lat.get("pim_reduce_cycles", 0),
            "pim_nl_cycles": lat.get("pim_nl_cycles", 0),
        },
        "stalls": {
            "dependency_stall_cycles": lat.get("stall_dependency_cycles", 0),
            "bank_conflict_stall_cycles": lat.get("bank_conflict_stall_cycles", 0),
            "capacity_spill_cycles": lat.get("spill_writeback_cycles", 0),
            "auto_reload_cycles": lat.get("auto_reload_cycles", 0),
            "final_writeback_cycles": lat.get("final_writeback_cycles", 0),
        },
        "energy": {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in result["energy"].items()},
        "traffic": result["traffic"],
        "memory_lifecycle": result.get("memory_lifecycle", {}),
        "pim": result["pim"],
        "correctness": result["correctness"],
        "lifecycle_events": result.get("lifecycle_events", {}),
        "valid_simulation": result.get("valid_simulation", False),
        "final_state": result.get("final_state", {}),
        "finalization": result.get("finalization", {}),
        "model_provenance": result.get("model_provenance", {}),
    }
    return yaml.dump(report, default_flow_style=False, sort_keys=False)
