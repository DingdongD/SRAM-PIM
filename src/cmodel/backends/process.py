"""Persistent line protocol used by cycle-coupled external simulators."""

from __future__ import annotations

import os
import selectors
import subprocess
from pathlib import Path

from ..config import PersistentBackendSpec, RepositorySpec
from ..errors import BackendProtocolError, BackendVersionError


def verify_repository(repository: RepositorySpec) -> None:
    repo = Path(repository.path).resolve()
    if not repo.is_dir():
        raise BackendVersionError(f"repository does not exist: {repo}")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != repository.expected_commit:
        raise BackendVersionError(
            f"repository HEAD mismatch for {repo}: expected {repository.expected_commit}, got {head}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise BackendVersionError(f"tracked files are modified in {repo}")


class PersistentLineProcess:
    def __init__(self, spec: PersistentBackendSpec, backend_name: str):
        verify_repository(spec.repository)
        executable = Path(spec.executable).resolve()
        config_path = Path(spec.config_path).resolve()
        if not executable.is_file():
            raise BackendVersionError(f"{backend_name} executable does not exist: {executable}")
        if not os.access(executable, os.X_OK):
            raise BackendVersionError(f"{backend_name} executable is not executable: {executable}")
        if not config_path.is_file():
            raise BackendVersionError(f"{backend_name} config does not exist: {config_path}")
        self.spec = spec
        self.backend_name = backend_name
        self.process = subprocess.Popen(
            [
                str(executable),
                "--config", str(config_path),
                "--ticks-numerator", str(spec.ticks_numerator),
                "--ticks-denominator", str(spec.ticks_denominator),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            raise BackendProtocolError(f"{backend_name} failed to create stdio pipes")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self.stderr = self.process.stderr
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.stdout, selectors.EVENT_READ)
        hello = self.read_fields()
        if len(hello) != 3 or hello[0] != "HELLO":
            raise BackendProtocolError(f"{backend_name} first message must be HELLO")
        if hello[1] != backend_name:
            raise BackendProtocolError(f"backend identity mismatch: expected {backend_name}, got {hello[1]}")
        if int(hello[2]) != spec.protocol_version:
            raise BackendProtocolError(
                f"{backend_name} protocol mismatch: expected {spec.protocol_version}, got {hello[2]}"
            )

    def send_fields(self, fields: tuple[str, ...]) -> None:
        if self.process.poll() is not None:
            stderr = self.stderr.read()
            raise BackendProtocolError(
                f"{self.backend_name} exited with code {self.process.returncode}: {stderr}"
            )
        self.stdin.write("\t".join(fields) + "\n")
        self.stdin.flush()

    def read_fields(self) -> tuple[str, ...]:
        events = self.selector.select(self.spec.timeout_seconds)
        if not events:
            raise BackendProtocolError(f"{self.backend_name} response timeout")
        line = self.stdout.readline()
        if line == "":
            stderr = self.stderr.read()
            raise BackendProtocolError(f"{self.backend_name} closed stdout unexpectedly: {stderr}")
        fields = tuple(line.rstrip("\n").split("\t"))
        if not fields or fields[0] == "":
            raise BackendProtocolError(f"{self.backend_name} returned an empty protocol message")
        return fields

    def request(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        self.send_fields(fields)
        return self.read_fields()

    def close(self) -> None:
        response = self.request(("CLOSE",))
        if response != ("CLOSED",):
            raise BackendProtocolError(f"{self.backend_name} did not acknowledge close")
        self.stdin.close()
        return_code = self.process.wait(timeout=self.spec.timeout_seconds)
        if return_code != 0:
            stderr = self.stderr.read()
            raise BackendProtocolError(f"{self.backend_name} exited with code {return_code}: {stderr}")
