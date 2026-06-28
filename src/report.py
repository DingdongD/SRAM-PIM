import yaml
from src.config import SimConfig


def generate_report(result: dict, config: SimConfig) -> str:
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
        "latency": result["latency"],
        "energy": {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in result["energy"].items()},
        "traffic": result["traffic"],
        "correctness": result["correctness"],
        "valid_simulation": result.get("valid_simulation", False),
        "memory_lifecycle": result.get("memory_lifecycle", {}),
        "pim": result["pim"],
    }
    return yaml.dump(report, default_flow_style=False, sort_keys=False)
