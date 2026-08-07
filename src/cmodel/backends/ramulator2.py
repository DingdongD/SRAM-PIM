"""Persistent Ramulator2 protocol adapter."""

from __future__ import annotations

from ..config import DRAMSpec, PersistentBackendSpec
from ..errors import BackendProtocolError
from .base import BackendCompletion
from .process import PersistentLineProcess


class Ramulator2Backend:
    def __init__(self, spec: PersistentBackendSpec, dram: DRAMSpec):
        self.process = PersistentLineProcess(spec, "ramulator2")
        info = self.process.request(("INFO",))
        if len(info) != 4 or info[0] != "INFO_RESULT":
            raise BackendProtocolError("Ramulator2 INFO response mismatch")
        if int(info[1]) != dram.transaction_bytes:
            raise BackendProtocolError(
                f"Ramulator2 transaction size mismatch: config={dram.transaction_bytes}, backend={info[1]}"
            )
        if int(info[2]) != dram.read_request_type or int(info[3]) != dram.write_request_type:
            raise BackendProtocolError("Ramulator2 request type IDs do not match C-model configuration")
        self.current_cycle = 0
        self.pending: set[int] = set()

    def submit(self, *, request_id: int, cycle: int, request_type: int, address: int, source_id: int, nbytes: int) -> bool:
        if cycle != self.current_cycle:
            raise BackendProtocolError(f"Ramulator2 submit cycle {cycle} does not match backend cycle {self.current_cycle}")
        response = self.process.request((
            "SUBMIT", str(request_id), str(cycle), str(request_type), str(address), str(source_id), str(nbytes)
        ))
        if len(response) != 3 or response[0] != "SUBMIT_RESULT" or int(response[1]) != request_id:
            raise BackendProtocolError("Ramulator2 submit response mismatch")
        if response[2] not in {"0", "1"}:
            raise BackendProtocolError("Ramulator2 accepted field must be 0 or 1")
        accepted = response[2] == "1"
        if accepted:
            if request_id in self.pending:
                raise BackendProtocolError("Ramulator2 accepted duplicate request_id")
            self.pending.add(request_id)
        return accepted

    def tick(self, cycle: int) -> tuple[BackendCompletion, ...]:
        if cycle != self.current_cycle:
            raise BackendProtocolError(f"Ramulator2 tick cycle {cycle} does not match backend cycle {self.current_cycle}")
        response = self.process.request(("TICK", str(cycle)))
        if len(response) != 3 or response[0] != "TICK_RESULT" or int(response[1]) != cycle:
            raise BackendProtocolError("Ramulator2 tick response mismatch")
        completions: list[BackendCompletion] = []
        if response[2] != "-":
            seen: set[int] = set()
            for token in response[2].split(","):
                request_id = int(token)
                if request_id in seen or request_id not in self.pending:
                    raise BackendProtocolError(f"Ramulator2 invalid completion {request_id}")
                seen.add(request_id)
                self.pending.remove(request_id)
                completions.append(BackendCompletion(request_id, cycle + 1))
        self.current_cycle += 1
        return tuple(completions)

    def close(self) -> None:
        if self.pending:
            raise BackendProtocolError(f"Ramulator2 close with pending requests: {sorted(self.pending)}")
        self.process.close()
