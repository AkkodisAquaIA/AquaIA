import argparse
import os

from detection.infer import test_from_config
from detection.train import train_from_config


def build_parser():
    # Z: create the main argument parser
    parser = argparse.ArgumentParser(description="AquaIA entry point")
    # Z: create subparser system supporting subcommands
    # Z: dest="command" stores the selected subcommand in parsed_args.command
    # Z: required=True requires the user to select a subcommand
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Z: create train subparser supporting "train" subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    # Z: add --config argument to train subparser, defaulting to detection/train_config.yaml
    train_parser.add_argument("--config", type=str, default=os.path.join("detection", "train_config.yaml"))
    # Z: add --resume argument to train subparser, allowing user to specify a run directory to resume training from
    # Z: !Warning! only DINO
    train_parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR", help="Resume training from an existing run directory (e.g. runs/20250615_142200)")
    # Z: bind a default handler to the "train" subcommand
    train_parser.set_defaults(command_handler=handle_train)

    test_parser = subparsers.add_parser("infer", help="Run inference on the specified dataset and split")
    test_parser.add_argument("--config", type=str, default=os.path.join("detection", "infer_config.yaml"))
    test_parser.set_defaults(command_handler=handle_test)

    return parser


def handle_train(args):
    """Z: receives a param "args", with attributes "config" and "resume"."""
    return train_from_config(args.config, resume_dir=args.resume)


def handle_test(args):
    """Z: receives a param "args", with attribute "config"."""
    return test_from_config(args.config)


def main(args=None):
    """Z: when args=None, parse_args reads parameters from terminal.
    Run "python main.py train"
    -> train_from_config("detection/train_config.yaml", resume_dir=None)

    Run "python main.py train --resume runs/<run_id>"
    -> train_from_config("detection/train_config.yaml", resume_dir="runs/<run_id>")

    Run "python main.py infer"
    -> test_from_config("detection/infer_config.yaml")
    """
    parser = build_parser()
    # Z: parse command line arguments
    # Z: receive args = ["train"] or args = ["infer"]
    # Z: if resume training, receive args = ["train", "--resume", "runs/<run_id>"]
    # Z: based on the rules registered previously, parse to get an object similar to:
    # Z: parsed_args.command = "train"
    # Z: parsed_args.config = "detection/train_config.yaml"
    # Z: parsed_args.resume = None, or "runs/<run_id>" if --resume is specified
    # Z: parsed_args.command_handler = handle_train
    parsed_args = parser.parse_args(args=args)
    # Z: call the appropriate handler: handle_train(parsed_args)
    # Z: -> train_from_config("detection/train_config.yaml", resume_dir=parsed_args.resume)
    return parsed_args.command_handler(parsed_args)


if __name__ == "__main__":
    main()
