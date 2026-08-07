"""Persistent BookSim2 packet/flit protocol adapter."""

from __future__ import annotations

from ..config import NoCSpec, PersistentBackendSpec
from ..errors import BackendProtocolError
from .base import NoCCompletion
from .process import PersistentLineProcess


class BookSim2Backend:
    def __init__(self, spec: PersistentBackendSpec, noc: NoCSpec):
        self.process = PersistentLineProcess(spec, "booksim2")
        info = self.process.request(("INFO",))
        if len(info) != 4 or info[0] != "INFO_RESULT":
            raise BackendProtocolError("BookSim2 INFO response mismatch")
        nodes = int(info[1])
        num_vcs = int(info[2])
        classes = int(info[3])
        if nodes != noc.nodes:
            raise BackendProtocolError(f"BookSim2 node count mismatch: config={noc.nodes}, backend={nodes}")
        vcs = (noc.vc_activation, noc.vc_weight, noc.vc_output, noc.vc_partial_sum, noc.vc_control)
        if any(vc < 0 or vc >= num_vcs for vc in vcs):
            raise BackendProtocolError("C-model VC index is outside the BookSim2 VC range")
        if classes <= 0:
            raise BackendProtocolError("BookSim2 must expose at least one traffic class")
        self.current_cycle = 0
        self.pending: dict[int, int] = {}

    def submit(self, *, packet_id: int, cycle: int, src: int, dst: int, vc: int, flits: int, traffic_class: int) -> bool:
        if cycle != self.current_cycle:
            raise BackendProtocolError(f"BookSim2 submit cycle {cycle} does not match backend cycle {self.current_cycle}")
        response = self.process.request((
            "SUBMIT", str(packet_id), str(cycle), str(src), str(dst), str(vc), str(flits), str(traffic_class)
        ))
        if len(response) != 3 or response[0] != "SUBMIT_RESULT" or int(response[1]) != packet_id:
            raise BackendProtocolError("BookSim2 submit response mismatch")
        if response[2] not in {"0", "1"}:
            raise BackendProtocolError("BookSim2 accepted field must be 0 or 1")
        accepted = response[2] == "1"
        if accepted:
            if packet_id in self.pending:
                raise BackendProtocolError("BookSim2 accepted duplicate packet_id")
            self.pending[packet_id] = flits
        return accepted

    def tick(self, cycle: int) -> tuple[NoCCompletion, ...]:
        if cycle != self.current_cycle:
            raise BackendProtocolError(f"BookSim2 tick cycle {cycle} does not match backend cycle {self.current_cycle}")
        response = self.process.request(("TICK", str(cycle)))
        if len(response) != 3 or response[0] != "TICK_RESULT" or int(response[1]) != cycle:
            raise BackendProtocolError("BookSim2 tick response mismatch")
        completions: list[NoCCompletion] = []
        if response[2] != "-":
            seen: set[int] = set()
            for token in response[2].split(","):
                packet_text, hops_text = token.split(":")
                packet_id = int(packet_text)
                if packet_id in seen or packet_id not in self.pending:
                    raise BackendProtocolError(f"BookSim2 invalid completion {packet_id}")
                seen.add(packet_id)
                flits = self.pending[packet_id]
                del self.pending[packet_id]
                completions.append(NoCCompletion(packet_id, cycle + 1, flits, int(hops_text)))
        self.current_cycle += 1
        return tuple(completions)

    def close(self) -> None:
        if self.pending:
            raise BackendProtocolError(f"BookSim2 close with pending packets: {sorted(self.pending)}")
        self.process.close()
