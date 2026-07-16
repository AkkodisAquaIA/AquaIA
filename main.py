import argparse
import os

from detection.infer import test_from_config
from detection.train import train_from_config


def build_parser():
    parser = argparse.ArgumentParser(description="AquaIA entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default=os.path.join("detection", "train_config.yaml"))
    train_parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR", help="Resume training from an existing run directory (e.g. runs/20250615_142200)")
    train_parser.set_defaults(command_handler=handle_train)

    test_parser = subparsers.add_parser("infer", help="Run inference on the specified dataset and split")
    test_parser.add_argument("--config", type=str, default=os.path.join("detection", "infer_config.yaml"))
    test_parser.set_defaults(command_handler=handle_test)

    return parser


def handle_train(args):
    return train_from_config(args.config, resume_dir=args.resume)


def handle_test(args):
    return test_from_config(args.config)


def main(args=None):
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)
    return parsed_args.command_handler(parsed_args)


if __name__ == "__main__":
    main()
