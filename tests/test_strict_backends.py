from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from src.cmodel.backends.booksim2 import BookSim2Backend
from src.cmodel.backends.process import verify_repository
from src.cmodel.backends.ramulator2 import Ramulator2Backend
from src.cmodel.backends.scalesim import ScaleSimBackend
from src.cmodel.backends.sram_macro import SRAMMacroBackend
from src.cmodel.config import DRAMSpec, NoCSpec, PersistentBackendSpec, RepositorySpec
from src.cmodel.errors import BackendProtocolError, BackendVersionError


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def write_server(path: Path, backend: str) -> None:
    if backend == "ramulator2":
        info = "INFO_RESULT\\t32\\t0\\t1"
        completion = "pending"
    else:
        info = "INFO_RESULT\\t16\\t5\\t1"
        completion = "pending"
    path.write_text(
        f'''#!/usr/bin/env python3\nimport sys\n\nbackend = "{backend}"\npending = []\nprint("HELLO\\t" + backend + "\\t1", flush=True)\nfor raw in sys.stdin:\n    fields = raw.rstrip("\\n").split("\\t")\n    if fields[0] == "INFO":\n        print("{info}", flush=True)\n    elif fields[0] == "SUBMIT":\n        item = int(fields[1])\n        pending.append(item)\n        print("SUBMIT_RESULT\\t" + fields[1] + "\\t1", flush=True)\n    elif fields[0] == "TICK":\n        if pending:\n            item = pending.pop(0)\n            suffix = str(item) if backend == "ramulator2" else str(item) + ":2"\n        else:\n            suffix = "-"\n        print("TICK_RESULT\\t" + fields[1] + "\\t" + suffix, flush=True)\n    elif fields[0] == "CLOSE":\n        print("CLOSED", flush=True)\n        raise SystemExit(0)\n    else:\n        raise RuntimeError(fields[0])\n''',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def persistent_spec(tmp_path: Path, backend: str) -> PersistentBackendSpec:
    repo, commit = make_repo(tmp_path)
    executable = tmp_path / f"{backend}_server.py"
    write_server(executable, backend)
    config = tmp_path / "config.txt"
    config.write_text("strict\n", encoding="utf-8")
    return PersistentBackendSpec(
        RepositorySpec(str(repo), commit),
        str(executable),
        str(config),
        1,
        5,
        1,
        1,
    )


def test_ramulator_protocol_and_conservation(tmp_path: Path) -> None:
    backend = Ramulator2Backend(persistent_spec(tmp_path, "ramulator2"), DRAMSpec(0, 1, 32, 0))
    assert backend.submit(request_id=7, cycle=0, request_type=0, address=4096, source_id=0, nbytes=32)
    completed = backend.tick(0)
    assert completed[0].request_id == 7
    assert not backend.pending
    backend.close()


def test_booksim_protocol_and_conservation(tmp_path: Path) -> None:
    noc = NoCSpec(16, 16, 14, 15, (0, 1, 2, 3), 0, 1, 2, 3, 4)
    backend = BookSim2Backend(persistent_spec(tmp_path, "booksim2"), noc)
    assert backend.submit(packet_id=9, cycle=0, src=15, dst=0, vc=0, flits=4, traffic_class=0)
    completed = backend.tick(0)
    assert completed[0].packet_id == 9
    assert completed[0].hops == 2
    assert not backend.pending
    backend.close()


def test_ramulator_info_mismatch_fails(tmp_path: Path) -> None:
    with pytest.raises(BackendProtocolError):
        Ramulator2Backend(persistent_spec(tmp_path, "ramulator2"), DRAMSpec(0, 1, 64, 0))


def test_booksim_info_mismatch_fails(tmp_path: Path) -> None:
    noc = NoCSpec(8, 16, 6, 7, (0, 1), 0, 1, 2, 3, 4)
    with pytest.raises(BackendProtocolError):
        BookSim2Backend(persistent_spec(tmp_path, "booksim2"), noc)


def test_repository_dirty_tracked_file_fails(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(BackendVersionError):
        verify_repository(RepositorySpec(str(repo), commit))


def test_scalesim_report_rejects_internal_memory_stalls(tmp_path: Path) -> None:
    report = tmp_path / "COMPUTE_REPORT.csv"
    report.write_text(
        "LayerID, Total Cycles, Stall Cycles, Overall Util %\n0, 10, 1, 50\n",
        encoding="utf-8",
    )
    with pytest.raises(BackendProtocolError):
        ScaleSimBackend._parse_report(report)


def test_scalesim_report_accepts_zero_stall(tmp_path: Path) -> None:
    report = tmp_path / "COMPUTE_REPORT.csv"
    report.write_text(
        "LayerID, Total Cycles, Stall Cycles, Overall Util %\n0, 10, 0, 50\n",
        encoding="utf-8",
    )
    cycles, utilization = ScaleSimBackend._parse_report(report)
    assert cycles == 10
    assert utilization == 0.5


def test_cacti_parser_requires_all_fields() -> None:
    text = """Access time (ns): 1.0\nCycle time (ns): 2.0\nTotal dynamic read energy per access (nJ): 0.003\nTotal dynamic write energy per access (nJ): 0.004\nTotal leakage power of a bank (mW): 0.5\nCache height x width (mm): 1.5 x 2.0\n"""
    result = SRAMMacroBackend._parse_cacti(text)
    assert result.read_energy_pj == 3.0
    assert result.write_energy_pj == 4.0
    assert result.area_mm2 == 3.0
