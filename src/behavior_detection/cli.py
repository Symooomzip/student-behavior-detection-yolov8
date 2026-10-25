"""Command line entry point: `behavior-detection {check,prepare,train,evaluate,app}`."""

import argparse

from .config import load_config, resolve_path
from .dataset import class_stats, load_samples


def cmd_check(cfg, args):
    samples, report = load_samples(resolve_path(cfg, "raw"), cfg["classes"])
    print(f"{len(samples)} labeled images")
    print(f"{len(report.unlabeled)} without usable labels: {report.unlabeled}")
    print(f"{len(report.problems)} invalid label lines")
    for line in report.problems:
        print("  ", line)
    print(f"\n{'class':15} {'boxes':>7} {'images':>7}")
    for name, s in class_stats(samples, cfg["classes"]).items():
        print(f"{name:15} {s['instances']:7} {s['images']:7}")


def cmd_prepare(cfg, args):
    from .prepare import prepare

    s = prepare(cfg)
    print("original images per split:", s["original_images"])
    print("augmented train images:", s["augmented_train_images"])
    print(f"data written to {resolve_path(cfg, 'prepared')}")


def cmd_train(cfg, args):
    from .train import train

    print("best weights copied to", train(cfg, device=args.device))


def cmd_evaluate(cfg, args):
    from .evaluate import evaluate

    evaluate(cfg, weights=args.weights, device=args.device)


def cmd_app(cfg, args):
    from .app import launch

    launch(cfg, weights=args.weights, share=args.share, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="behavior-detection", description=__doc__)
    parser.add_argument("--config", help="path to a config yaml (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the raw dataset and print class statistics").set_defaults(func=cmd_check)
    sub.add_parser("prepare", help="split, then balance the train split").set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train YOLOv8 on the prepared data")
    p.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate weights on val and test")
    p.add_argument("--weights", help="default: paths.weights from the config")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("app", help="launch the Gradio demo")
    p.add_argument("--weights", help="default: paths.weights from the config")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
