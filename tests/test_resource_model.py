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


def test_get_bank_conflicts_count():
    rm = ResourceModel(make_config())
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=5)
    # Trying to read a bank that is busy with PIM should increment conflict count
    result = rm.can_issue(OpCode.SRAM_RD, banks=[0], nbytes=0)
    assert result is False
    assert rm.get_bank_conflicts() == 1


def test_pim_exclusive_with_write():
    rm = ResourceModel(make_config())
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=2)
    assert rm.can_issue(OpCode.SRAM_WR, banks=[0], nbytes=0) is False
    assert rm.can_issue(OpCode.SRAM_WR, banks=[1], nbytes=0) is True


def test_read_on_idle_bank():
    rm = ResourceModel(make_config())
    # No reservations — read should always be allowed
    assert rm.can_issue(OpCode.SRAM_RD, banks=[0, 1, 2], nbytes=0) is True


def test_pim_on_different_banks_no_conflict():
    cfg = make_config()
    cfg.pim.max_parallel_banks = 4
    rm = ResourceModel(cfg)
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=2)
    # Different bank — no exclusivity conflict
    assert rm.can_issue(OpCode.PIM_MAC, banks=[1], nbytes=0) is True


def test_release_partial():
    """release_at should only free reservations whose end_cycle <= cycle."""
    rm = ResourceModel(make_config())
    rm.reserve(OpCode.PIM_MAC, banks=[0], nbytes=0, duration=5)
    rm.reserve(OpCode.PIM_MAC, banks=[1], nbytes=0, duration=10)
    rm.release_at(cycle=5)
    # Bank 0 released, bank 1 still busy
    assert rm.can_issue(OpCode.SRAM_RD, banks=[0], nbytes=0) is True
    assert rm.can_issue(OpCode.SRAM_RD, banks=[1], nbytes=0) is False
