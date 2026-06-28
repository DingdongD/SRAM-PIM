"""DRAMsim3 wrapper for validated cycle-accurate DRAM timing.

Trace format (DRAMsim3): <hex_addr> <READ|WRITE> <clock_cycle>
The wrapper generates such traces from logical DRAM commands and invokes
the dramsim3main.out binary, then parses the JSON stats output for latency.
"""

import json
import os
import subprocess
import tempfile


def generate_dram_trace(
    dram_addr: int,
    nbytes: int,
    burst_bytes: int = 64,
    is_write: bool = False,
) -> list:
    """Generate a list of DRAMsim3 trace lines for a contiguous transfer.

    Each line is ``"0x<ADDR> <READ|WRITE>"`` (no cycle field; the wrapper
    adds a monotonically increasing cycle when writing the actual trace file).

    Args:
        dram_addr:   Base DRAM byte address.
        nbytes:      Total number of bytes to transfer.
        burst_bytes: Bytes per burst (cache-line / transaction granularity).
        is_write:    True for WRITE, False for READ.

    Returns:
        List of trace-line strings, one per burst.
    """
    n_bursts = max(1, nbytes // burst_bytes)
    op = "WRITE" if is_write else "READ"
    trace = []
    for i in range(n_bursts):
        addr = dram_addr + i * burst_bytes
        trace.append(f"0x{addr:X} {op}")
    return trace


class DRAMsim3Wrapper:
    """Thin wrapper around the DRAMsim3 stand-alone executable.

    Writes a temporary trace file, invokes ``dramsim3main.out``, parses the
    JSON stats file it produces, and returns cycle counts / latencies.

    Args:
        config_file:  Absolute path to a DRAMsim3 ``.ini`` config.
        dramsim3_dir: Root of the DRAMsim3 installation (contains the binary).
    """

    def __init__(self, config_file: str, dramsim3_dir: str = "/home/DRAMsim3"):
        self.config_file = config_file
        self.dramsim3_dir = dramsim3_dir
        self.exe = os.path.join(dramsim3_dir, "dramsim3main.out")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_trace_file(self, trace_lines: list) -> int:
        """Run DRAMsim3 with the supplied trace lines and return cycle count.

        ``trace_lines`` may contain lines with or without a cycle field.
        If the cycle field is absent the wrapper injects monotonically
        increasing cycle numbers (0, 1, 2, …).

        Args:
            trace_lines: List of strings in ``"0x<ADDR> <OP> [cycle]"`` format.

        Returns:
            ``average_read_latency`` from DRAMsim3's JSON stats (rounded to
            nearest integer), or ``num_cycles`` if no read latency is present,
            or 0 if the binary is unavailable / fails.
        """
        if not os.path.isfile(self.exe):
            return 0

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".trace", delete=False, dir="/tmp"
        ) as tf:
            for idx, line in enumerate(trace_lines):
                parts = line.split()
                # Ensure the clock-cycle field is present
                if len(parts) == 2:
                    tf.write(f"{parts[0]} {parts[1]} {idx}\n")
                else:
                    tf.write(line + "\n")
            trace_path = tf.name

        out_dir = tempfile.mkdtemp(prefix="dramsim3_out_", dir="/tmp")
        try:
            subprocess.run(
                [self.exe, self.config_file, "-t", trace_path, "-o", out_dir],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.dramsim3_dir,
            )
            return self._parse_cycles(out_dir)
        finally:
            os.unlink(trace_path)
            # Clean up output files
            for fname in os.listdir(out_dir):
                try:
                    os.unlink(os.path.join(out_dir, fname))
                except OSError:
                    pass
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

    def get_latency_for_transfer(
        self,
        dram_addr: int,
        nbytes: int,
        is_write: bool = False,
        burst_bytes: int = 64,
    ) -> int:
        """High-level helper: generate trace and return simulated latency.

        Args:
            dram_addr:   Base address of the DRAM transfer.
            nbytes:      Transfer size in bytes.
            is_write:    Direction of transfer.
            burst_bytes: Burst granularity.

        Returns:
            Simulated latency in DRAM cycles (>0), or 0 on failure.
        """
        trace = generate_dram_trace(dram_addr, nbytes, burst_bytes, is_write)
        return self.run_trace_file(trace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_cycles(self, out_dir: str) -> int:
        """Parse DRAMsim3 JSON stats and extract a representative cycle count.

        Priority:
        1. ``average_read_latency`` (float → rounded int) from channel 0.
        2. ``average_write_latency`` if no read latency.
        3. ``num_cycles`` as a last resort.

        Returns 0 if the JSON file is missing or unparseable.
        """
        json_path = os.path.join(out_dir, "dramsim3.json")
        if not os.path.isfile(json_path):
            return 0
        try:
            with open(json_path) as f:
                data = json.load(f)
            # Stats are keyed by channel index (as strings: "0", "1", …)
            ch = data.get("0", data)
            avg_read = ch.get("average_read_latency")
            if avg_read is not None and avg_read > 0:
                return round(avg_read)
            avg_write = ch.get("average_write_latency")
            if avg_write is not None and avg_write > 0:
                return round(avg_write)
            num_cycles = ch.get("num_cycles", 0)
            return int(num_cycles)
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0
