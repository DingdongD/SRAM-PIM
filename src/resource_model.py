from src.trace_ir import OpCode
from src.config import SRAMPIMConfig

# P1-03: From SRAM perspective, DMA_LOAD writes to SRAM, DMA_STORE reads from SRAM
_SRAM_READ_OPS = {OpCode.SRAM_RD, OpCode.DMA_STORE}
_SRAM_WRITE_OPS = {OpCode.SRAM_WR, OpCode.DMA_LOAD, OpCode.DMA_PREFETCH}
# P1-04: PIM_NL is a PIM operation and must occupy PIM resources
_PIM_OPS = {OpCode.PIM_MAC, OpCode.PIM_EW_OP, OpCode.PIM_REDUCE, OpCode.PIM_NL}


class ResourceModel:
    def __init__(self, config: SRAMPIMConfig):
        self.config = config
        total_banks = config.tiles * config.banks_per_tile
        self.bank_busy_until = [0] * total_banks
        self.bank_op_type = [None] * total_banks
        self.pim_active_banks = 0
        self.current_cycle = 0
        self.bank_conflict_count = 0
        self._reservations = []  # (release_cycle, banks, op)

    def can_issue(self, op: OpCode, banks: list, nbytes: int = 0) -> bool:
        port = self.config.port_model

        for b in banks:
            if b >= len(self.bank_busy_until):
                continue
            if self.bank_busy_until[b] > self.current_cycle:
                current_op = self.bank_op_type[b]
                # PIM requesting on busy bank
                if op in _PIM_OPS:
                    if current_op in _SRAM_READ_OPS and port.pim_exclusive_with_read:
                        self.bank_conflict_count += 1
                        return False
                    if current_op in _SRAM_WRITE_OPS and port.pim_exclusive_with_write:
                        self.bank_conflict_count += 1
                        return False
                    if current_op in _PIM_OPS:
                        self.bank_conflict_count += 1
                        return False
                # Read/write requesting on bank busy with PIM
                if current_op in _PIM_OPS:
                    if op in _SRAM_READ_OPS and port.pim_exclusive_with_read:
                        self.bank_conflict_count += 1
                        return False
                    if op in _SRAM_WRITE_OPS and port.pim_exclusive_with_write:
                        self.bank_conflict_count += 1
                        return False
                # Same port type conflict
                if current_op in _SRAM_READ_OPS and op in _SRAM_READ_OPS:
                    if port.read_ports_per_bank <= 1:
                        self.bank_conflict_count += 1
                        return False
                if current_op in _SRAM_WRITE_OPS and op in _SRAM_WRITE_OPS:
                    if port.write_ports_per_bank <= 1:
                        self.bank_conflict_count += 1
                        return False

        # Power throttle: limit simultaneous PIM banks
        if op in _PIM_OPS:
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

        if op in _PIM_OPS:
            self.pim_active_banks += len(banks)

        self._reservations.append((release_cycle, list(banks), op))

    def release_at(self, cycle: int):
        self.current_cycle = cycle
        remaining = []
        for release_cycle, banks, op in self._reservations:
            if release_cycle <= cycle:
                if op in _PIM_OPS:
                    self.pim_active_banks = max(0, self.pim_active_banks - len(banks))
            else:
                remaining.append((release_cycle, banks, op))
        self._reservations = remaining

    def get_bank_conflicts(self) -> int:
        return self.bank_conflict_count
