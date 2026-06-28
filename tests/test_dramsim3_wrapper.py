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


def test_generate_trace_addresses():
    """Each burst address should be offset by burst_bytes."""
    trace = generate_dram_trace(dram_addr=0x1000, nbytes=256, burst_bytes=64, is_write=False)
    expected_addrs = [f"0x{0x1000 + i * 64:X}" for i in range(4)]
    for line, expected_addr in zip(trace, expected_addrs):
        addr_part = line.split()[0]
        assert addr_part == expected_addr


def test_generate_trace_minimum_one_burst():
    """Small transfers should produce at least one burst."""
    trace = generate_dram_trace(dram_addr=0x1000, nbytes=1, burst_bytes=64, is_write=False)
    assert len(trace) == 1


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


@pytest.mark.skipif(not os.path.exists("/home/DRAMsim3/dramsim3main.out"),
                    reason="DRAMsim3 binary not available")
def test_dramsim3_get_latency():
    wrapper = DRAMsim3Wrapper(
        config_file="/home/DRAMsim3/configs/DDR4_8Gb_x8_2400.ini",
        dramsim3_dir="/home/DRAMsim3"
    )
    latency = wrapper.get_latency_for_transfer(dram_addr=0x1000, nbytes=256, is_write=False)
    assert latency > 0


def test_wrapper_exe_path():
    """DRAMsim3Wrapper should record exe path correctly."""
    wrapper = DRAMsim3Wrapper(
        config_file="/home/DRAMsim3/configs/DDR4_8Gb_x8_2400.ini",
        dramsim3_dir="/home/DRAMsim3"
    )
    assert wrapper.exe == "/home/DRAMsim3/dramsim3main.out"
    assert wrapper.config_file == "/home/DRAMsim3/configs/DDR4_8Gb_x8_2400.ini"
