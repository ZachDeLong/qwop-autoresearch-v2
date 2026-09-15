import argparse
import json
import multiprocessing


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Reproducible QWOP experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    boot = sub.add_parser("bootstrap", help="Fetch/patch local game and resolve ChromeDriver")
    boot.add_argument("--browser")
    evaluate = sub.add_parser("evaluate", help="Record deterministic validation trajectories")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--label", default="evaluation")
    evaluate.add_argument("--ledger", required=True)
    evaluate.add_argument("--cap", type=int, default=50000)
    replay = sub.add_parser("replay", help="Verify every transition and export a timed MP4")
    replay.add_argument("--input", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--label", default="Verified replay")
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--cap", type=int, default=50000)
    train = sub.add_parser("train", help="Fixed-budget PPO warm-start pilot")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--steps", type=int, default=131072)
    train.add_argument("--time-cost", type=float, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--ledger", required=True)
    train.add_argument("--cap", type=int, default=350000)
    pilot = sub.add_parser("pilot", help="Run the complete matched two-arm pilot and export videos")
    pilot.add_argument("--checkpoint", default=".runtime/checkpoints/original.pt")
    pilot.add_argument("--out", required=True)
    pilot.add_argument("--steps", type=int, default=131072)
    pilot.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "bootstrap":
        from .bootstrap import bootstrap

        result = bootstrap(args.browser)
    elif args.command == "evaluate":
        from .budget import Ledger
        from .environment import VALIDATION_CASES
        from .evaluation import evaluate

        result = evaluate(
            args.checkpoint, VALIDATION_CASES, args.out, Ledger(args.ledger, args.cap), args.label
        )
    elif args.command == "replay":
        from .budget import Ledger
        from .replay import replay

        result = replay(args.input, args.out, Ledger(args.ledger, args.cap), args.label)
    elif args.command == "train":
        from .budget import Ledger
        from .training import train

        result = str(
            train(
                args.checkpoint,
                args.out,
                Ledger(args.ledger, args.cap),
                args.steps,
                args.time_cost,
                args.seed,
            )
        )
    elif args.command == "pilot":
        from .pilot import pilot

        result = pilot(args.checkpoint, args.out, args.steps, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
