from src.config import load_config


def main():
    config = load_config("configs")
    print(f"SRAM-PIM Simulator")
    print(f"  System frequency: {config.system.frequency_hz / 1e9:.1f} GHz")
    print(f"  Mode: {config.system.mode}")
    print(f"  SRAM-PIM tiles: {config.sram_pim.tiles}")
    print(f"  Total SRAM capacity: {config.sram_pim.total_capacity_kb} KB")
    print(f"  DRAM model: {config.dram.model} ({config.dram.effective_bandwidth_gbps} GB/s)")


if __name__ == "__main__":
    main()
