from __future__ import annotations

import argparse

from src.cmodel.backends import BookSim2Backend, Ramulator2Backend, SRAMMacroBackend, ScaleSimBackend
from src.cmodel.config import load_cmodel_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_cmodel_config(args.config)

    macro = SRAMMacroBackend(config.backends.sram_macro).run()
    if macro.access_time_ns <= 0.0 or macro.cycle_time_ns <= 0.0:
        raise RuntimeError("SRAM macro timing must be positive")

    scalesim = ScaleSimBackend(config.backends.scalesim, config.architecture.arrays)
    scale_result = scalesim.run_gemm(
        op_id="strict_smoke",
        m=16,
        n=16,
        k=16,
        input_base=0,
        weight_base=0,
        output_base=0,
    )
    if scale_result.cycles <= 0:
        raise RuntimeError("SCALE-Sim smoke test returned non-positive cycles")

    ramulator = Ramulator2Backend(config.backends.ramulator2, config.architecture.dram)
    accepted = ramulator.submit(
        request_id=1,
        cycle=0,
        request_type=config.architecture.dram.read_request_type,
        address=config.addresses.input_base,
        source_id=config.architecture.dram.source_id,
        nbytes=config.architecture.dram.transaction_bytes,
    )
    if not accepted:
        raise RuntimeError("Ramulator2 rejected the first smoke request")
    cycle = 0
    while ramulator.pending:
        ramulator.tick(cycle)
        cycle += 1
        if cycle >= 10000:
            raise RuntimeError("Ramulator2 smoke request did not complete")
    ramulator.close()

    booksim = BookSim2Backend(config.backends.booksim2, config.architecture.noc)
    accepted = booksim.submit(
        packet_id=1,
        cycle=0,
        src=config.architecture.noc.endpoint_for_dram,
        dst=config.architecture.noc.endpoint_for_bank_group[0],
        vc=config.architecture.noc.vc_activation,
        flits=4,
        traffic_class=0,
    )
    if not accepted:
        raise RuntimeError("BookSim2 rejected the first smoke packet")
    cycle = 0
    while booksim.pending:
        booksim.tick(cycle)
        cycle += 1
        if cycle >= 10000:
            raise RuntimeError("BookSim2 smoke packet did not complete")
    booksim.close()


if __name__ == "__main__":
    main()
