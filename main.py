import argparse
import sys

from src.config import load_config
from src.tracegen.gen_gemm_trace import gen_gemm_trace
from src.tracegen.gen_attention_trace import gen_attention_trace
from src.simulator import Simulator
from src.report import generate_report
from src.trace_validator import TraceValidator


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
    parser.add_argument("--validate-trace", action="store_true",
                        help="Run static trace validation before simulation")
    # Attention-specific arguments
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-head", type=int, default=64, help="Dimension per attention head")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--stage", type=str, default="generation",
                        help="Attention stage: generation (decode) or summarization (prefill)")
    # Transformer-specific arguments
    parser.add_argument("--ndec", type=int, default=1, help="Number of decoder layers")
    parser.add_argument("--hdim", type=int, default=256, help="Hidden/model dimension")
    parser.add_argument("--ff-scale", type=float, default=4.0, help="FFN expansion ratio")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
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
    elif args.workload == "attention":
        cmds = gen_attention_trace(
            batch=config.workload.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            d_head=args.d_head,
            sram_config=config.sram_pim,
            precision=config.workload.precision.activation,
            mode=mode,
            stage=args.stage,
        )
    elif args.workload == "transformer":
        model_config = {
            "name": "custom",
            "ndec": args.ndec,
            "hdim": args.hdim,
            "num_heads": args.num_heads,
            "d_head": args.d_head,
            "ff_scale": args.ff_scale,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "gen_len": 1,
        }
        from src.tracegen.gen_transformer_trace import gen_transformer_trace
        cmds = gen_transformer_trace(
            model_config, config.sram_pim,
            precision=config.workload.precision.activation,
            mode=mode,
            stage=args.stage,
        )
    else:
        print(f"Unknown workload: {args.workload}", file=sys.stderr)
        sys.exit(1)

    if args.validate_trace:
        validator = TraceValidator()
        errors = validator.validate(cmds)
        for e in errors:
            print(f"[{e.severity.upper()}] cmd {e.cmd_id}: {e.message}",
                  file=sys.stderr)
        fatal = [e for e in errors if e.severity == "error"]
        if fatal:
            print(f"\nTrace validation failed with {len(fatal)} error(s).",
                  file=sys.stderr)
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
