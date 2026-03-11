import argparse
import os

from test.infer import test_from_config
from training.train import train_from_config


def build_parser():
    parser = argparse.ArgumentParser(description="AquaIA entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a DETR model")
    train_parser.add_argument("--config", type=str, default=os.path.join("training","train_config.yaml"))
    train_parser.set_defaults(command_handler=handle_train)

    test_parser = subparsers.add_parser("infer", help="Run DETR inference on train and test samples")
    test_parser.add_argument("--config", type=str, default=os.path.join("test", "infer_config.yaml"))
    test_parser.set_defaults(command_handler=handle_test)

    return parser


def handle_train(args):
    return train_from_config(args.config)


def handle_test(args):
    return test_from_config(args.config)


def main(args=None):
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)
    return parsed_args.command_handler(parsed_args)


if __name__ == "__main__":
    main()
