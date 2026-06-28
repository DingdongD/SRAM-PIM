from dataclasses import dataclass


@dataclass(frozen=True)
class SRAMLoc:
    tile: int
    banks: tuple


def parse_sram_loc(s: str) -> SRAMLoc:
    """Parse 'SRAM:T0:B0-3' -> SRAMLoc(tile=0, banks=(0,1,2,3))"""
    parts = s.split(":")
    tile = int(parts[1][1:])
    bank_str = parts[2][1:]
    if "-" in bank_str:
        lo, hi = bank_str.split("-")
        banks = tuple(range(int(lo), int(hi) + 1))
    else:
        banks = tuple(int(b) for b in bank_str.split(","))
    return SRAMLoc(tile=tile, banks=banks)


def global_bank_ids(loc: SRAMLoc, banks_per_tile: int) -> list:
    return [loc.tile * banks_per_tile + b for b in loc.banks]


def parse_src_ids(src: str) -> list:
    """Parse comma-separated src field into list of object IDs, filtering out
    location strings and placeholders."""
    ids = []
    for tok in src.split(","):
        tok = tok.strip()
        if tok and tok != "-" and not tok.startswith("DRAM:") and not tok.startswith("SRAM:"):
            ids.append(tok)
    return ids
