import argparse
import sys

from src.config import load_config
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.simulator import Simulator
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="SRAM-PIM Simulator")
    parser.add_argument("--config", type=str, default="configs", help="Config directory")
    parser.add_argument("--workload", type=str, default="gemm", help="Workload type: gemm")
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--Tm", type=int, default=64)
    parser.add_argument("--Tn", type=int, default=64)
    parser.add_argument("--Tk", type=int, default=128)
    parser.add_argument("--mode", type=str, default=None,
                        help="Override: cold_start|warm_resident|amortized")
    parser.add_argument("--output", type=str, default=None, help="Output YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = args.mode or config.system.mode

    if args.workload == "gemm":
        cmds = gen_gemm_trace(
            M=args.M, N=args.N, K=args.K,
            Tm=args.Tm, Tn=args.Tn, Tk=args.Tk,
            sram_config=config.sram_pim,
            mode=mode,
        )
    else:
        print(f"Unknown workload: {args.workload}", file=sys.stderr)
        sys.exit(1)

    sim = Simulator(config)
    sim.load_trace(cmds)
    result = sim.run()
    report = generate_report(result, config)

    print(report)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
