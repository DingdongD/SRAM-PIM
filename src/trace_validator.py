"""P2-02: Static trace validator — checks trace correctness before simulation."""

from dataclasses import dataclass, field
from src.trace_ir import TraceCommand, OpCode


@dataclass
class TraceError:
    cmd_id: int
    severity: str  # "error" or "warning"
    message: str


class TraceValidator:
    """Validate a trace for structural correctness before simulation."""

    def validate(self, commands: list[TraceCommand]) -> list[TraceError]:
        errors: list[TraceError] = []
        cmd_map = {c.cmd_id: c for c in commands}
        all_ids = set(cmd_map.keys())

        # Track which objects are produced/loaded/preloaded
        produced: set[str] = set()
        # Track last writer per object for WAW check
        last_writer: dict[str, int] = {}
        # Track dirty objects (produced but not stored)
        dirty: set[str] = set()
        # Track stored objects
        stored: set[str] = set()

        # Build dependency graph for cycle detection
        adj: dict[int, list[int]] = {c.cmd_id: list(c.deps) for c in commands}

        # Check for cycles
        if self._has_cycle(adj, all_ids):
            errors.append(TraceError(-1, "error", "Dependency graph contains a cycle"))

        for cmd in commands:
            # Unknown dependency IDs
            for dep in cmd.deps:
                if dep not in all_ids:
                    errors.append(TraceError(
                        cmd.cmd_id, "error",
                        f"Unknown dependency id {dep}"))

            if cmd.op == OpCode.SRAM_ALLOC:
                preloaded = cmd.attrs.get("preloaded", False)
                if preloaded:
                    produced.add(cmd.object_id)
                last_writer[cmd.object_id] = cmd.cmd_id

            elif cmd.op == OpCode.DMA_LOAD:
                produced.add(cmd.object_id)
                last_writer[cmd.object_id] = cmd.cmd_id

            elif cmd.op in {OpCode.PIM_MAC, OpCode.PIM_EW_OP,
                            OpCode.PIM_REDUCE, OpCode.PIM_NL}:
                # Check each src object has a producer
                src_objs = self._parse_src(cmd.src)
                for obj_id in src_objs:
                    if obj_id not in produced:
                        errors.append(TraceError(
                            cmd.cmd_id, "error",
                            f"Input '{obj_id}' has no producer/DMA_LOAD/preloaded alloc"))
                    # Check dependency on producer
                    if obj_id in last_writer:
                        producer_id = last_writer[obj_id]
                        if not self._depends_on(cmd.cmd_id, producer_id, adj):
                            errors.append(TraceError(
                                cmd.cmd_id, "warning",
                                f"No dependency path to producer of '{obj_id}' (cmd {producer_id})"))

                # WAW check: if output already has a writer, must depend on it
                if cmd.object_id in last_writer:
                    prev = last_writer[cmd.object_id]
                    if prev not in cmd.deps and not self._depends_on(cmd.cmd_id, prev, adj):
                        errors.append(TraceError(
                            cmd.cmd_id, "warning",
                            f"WAW hazard: '{cmd.object_id}' also written by cmd {prev} with no dependency"))

                produced.add(cmd.object_id)
                dirty.add(cmd.object_id)
                last_writer[cmd.object_id] = cmd.cmd_id

            elif cmd.op == OpCode.DMA_STORE:
                # Must depend on last producer
                if cmd.object_id in last_writer:
                    producer_id = last_writer[cmd.object_id]
                    if not self._depends_on(cmd.cmd_id, producer_id, adj):
                        errors.append(TraceError(
                            cmd.cmd_id, "warning",
                            f"DMA_STORE of '{cmd.object_id}' has no dependency on its producer (cmd {producer_id})"))
                stored.add(cmd.object_id)
                dirty.discard(cmd.object_id)

            elif cmd.op == OpCode.SRAM_FREE:
                # Freeing dirty object without prior DMA_STORE
                if cmd.object_id in dirty and cmd.object_id not in stored:
                    errors.append(TraceError(
                        cmd.cmd_id, "warning",
                        f"SRAM_FREE on dirty object '{cmd.object_id}' without prior DMA_STORE"))

        return errors

    @staticmethod
    def _parse_src(src: str) -> list[str]:
        ids = []
        for tok in src.split(","):
            tok = tok.strip()
            if (tok and tok != "-"
                    and not tok.startswith("SRAM:")
                    and not tok.startswith("DRAM:")):
                ids.append(tok)
        return ids

    @staticmethod
    def _has_cycle(adj: dict[int, list[int]], nodes: set[int]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}

        def dfs(u: int) -> bool:
            color[u] = GRAY
            for v in adj.get(u, []):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        for n in nodes:
            if color[n] == WHITE:
                if dfs(n):
                    return True
        return False

    @staticmethod
    def _depends_on(cmd_id: int, target: int, adj: dict[int, list[int]]) -> bool:
        """Check if cmd_id transitively depends on target via BFS."""
        visited = set()
        queue = [cmd_id]
        while queue:
            current = queue.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for dep in adj.get(current, []):
                queue.append(dep)
        return False
