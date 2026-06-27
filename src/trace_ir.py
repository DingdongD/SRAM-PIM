from dataclasses import dataclass, field
from enum import Enum
import json


class OpCode(Enum):
    SRAM_ALLOC = "SRAM_ALLOC"
    SRAM_FREE = "SRAM_FREE"
    DMA_LOAD = "DMA_LOAD"
    DMA_STORE = "DMA_STORE"
    DMA_PREFETCH = "DMA_PREFETCH"
    SRAM_RD = "SRAM_RD"
    SRAM_WR = "SRAM_WR"
    PIM_MAC = "PIM_MAC"
    PIM_EW_OP = "PIM_EW_OP"
    PIM_REDUCE = "PIM_REDUCE"
    PIM_NL = "PIM_NL"
    PIM_WRITEBACK = "PIM_WRITEBACK"
    BARRIER = "BARRIER"
    POWER_SET = "POWER_SET"


@dataclass
class TraceCommand:
    cmd_id: int
    op: OpCode
    object_id: str
    src: str
    dst: str
    bytes: int
    attrs: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)


def write_trace(commands: list, path: str) -> None:
    with open(path, 'w') as f:
        for cmd in commands:
            deps_str = ",".join(str(d) for d in cmd.deps) if cmd.deps else "-"
            attrs_str = json.dumps(cmd.attrs) if cmd.attrs else "{}"
            line = f"{cmd.cmd_id}\t{cmd.op.value}\t{cmd.object_id}\t{cmd.src}\t{cmd.dst}\t{cmd.bytes}\t{attrs_str}\t{deps_str}\n"
            f.write(line)


def parse_trace(path: str) -> list:
    commands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            cmd_id = int(parts[0])
            op = OpCode(parts[1])
            object_id = parts[2]
            src = parts[3]
            dst = parts[4]
            nbytes = int(parts[5])
            attrs = json.loads(parts[6]) if parts[6] != "{}" else {}
            deps_str = parts[7]
            deps = [] if deps_str == "-" else [int(d) for d in deps_str.split(",")]
            commands.append(TraceCommand(cmd_id, op, object_id, src, dst, nbytes, attrs, deps))
    return commands
