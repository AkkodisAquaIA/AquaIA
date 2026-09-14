import argparse
import os
from detection.runner import train_from_config, infer_from_config


def build_parser():
    # Create the main argument parser
    parser = argparse.ArgumentParser(description="AquaIA entry point")
    # Create subparser system supporting subcommands
    # dest="command" stores the selected subcommand in parsed_args.command
    # required=True requires the user to select a subcommand
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create train subparser supporting "train" subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    # Add --config argument to train subparser, defaulting to detection/train_config.yaml
    train_parser.add_argument("--config", type=str, default=os.path.join("detection", "train_config.yaml"))
    # Add --resume argument to train subparser, allowing user to specify a run directory to resume training from
    # !Warning! Only for DINO
    train_parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR", help="Resume training from an existing run directory (e.g. runs/20250615_142200)")
    # Bind a default handler to the "train" subcommand
    train_parser.set_defaults(command_handler=handle_train)

    infer_parser = subparsers.add_parser("infer", help="Run inference on the specified dataset and split")
    infer_parser.add_argument("--config", type=str, default=os.path.join("detection", "infer_config.yaml"))
    infer_parser.set_defaults(command_handler=handle_infer)

    return parser


def handle_train(args):
    """Receives a param "args", with attributes "config" and "resume"."""
    return train_from_config(args.config, resume_dir=args.resume)


def handle_infer(args):
    """Receives a param "args", with attribute "config"."""
    return infer_from_config(args.config)


def main(args=None):
    """When args=None, parse_args reads parameters from terminal.
    Run "python main.py train"
    -> train_from_config("detection/train_config.yaml", resume_dir=None)

    Run "python main.py train --resume runs/<run_id>"
    -> train_from_config("detection/train_config.yaml", resume_dir="runs/<run_id>")

    Run "python main.py infer"
    -> infer_from_config("detection/infer_config.yaml")
    """
    parser = build_parser()
    # Parse command line arguments
    # Receive args = ["train"] or args = ["infer"]
    # If resume training, receive args = ["train", "--resume", "runs/<run_id>"]
    # Based on the rules registered previously, parse to get an object similar to:
    # parsed_args.command = "train"
    # parsed_args.config = "detection/train_config.yaml"
    # parsed_args.resume = None, or "runs/<run_id>" if --resume is specified
    # parsed_args.command_handler = handle_train
    parsed_args = parser.parse_args(args=args)
    # Call the appropriate handler: handle_train(parsed_args)
    # -> train_from_config("detection/train_config.yaml", resume_dir=parsed_args.resume)
    return parsed_args.command_handler(parsed_args)


if __name__ == "__main__":
    main()
