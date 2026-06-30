import argparse
import importlib

from trainer.base import Trainer
from trainer.config import add_config_arguments, parse_config


def load_task_module(task_name: str):
    return importlib.import_module(f"trainer.tasks.{task_name}")


def main() -> None:
    # Parse the task first so task-specific modules, including third-party imports,
    # are loaded only for the selected training target.
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument("--task", required=True)
    task_args, _ = task_parser.parse_known_args()
    task_module = load_task_module(task_args.task)

    parser = argparse.ArgumentParser(description="Trainer for pair latent transform models")
    parser.add_argument("--task", required=True)
    add_config_arguments(parser, task_module.Config)
    args = parser.parse_args()

    config = parse_config(args, task_module.Config)
    task = task_module.Task(config)

    trainer = Trainer(config=config, task=task, task_name=args.task)
    try:
        trainer.setup()
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
