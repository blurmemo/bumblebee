import os
import sys

import click
import subprocess

from pydantic.v1 import NoneStr

MODULE_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(MODULE_DIR, "scripts")


@click.group()
def cli():
    """Bumblebee: A flexible training and inference framework."""
    pass


@cli.command(help="Run model training.", context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--command",
    type=str,
    help="Allowed commands with `python(don't set), deepspeed, torchrun`",
    default=None,
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def train(command, extra_args):
    """
    Train command runs `bumblebee.scripts.train.py` using different launchers:
    Args:
        --command Optional [deepspeed, torchrun]
        --extra_args other arguments with `--name value...`

    Examples:
      bumblebee train --command deepspeed|torchrun .. --args args.py (
        --model model.py
        --dataloader dataloader.py
        --optimizer optimizer.py
        --scheduler scheduler.py
        --evaluate evaluate.py
        )

    """
    allowed_commands = ["deepspeed", "torchrun"]
    if command is None:
        command = sys.executable
    elif command not in allowed_commands:
        raise click.BadParameter(f"Invalid --command: {command}, Allowed: deepspeed, torchrun, or no-set for python")

    script_filename = "train.py"
    script_path = os.path.join(SCRIPTS_DIR, script_filename)
    # sys.executable == python exec
    cmd = [command, script_path] + list(extra_args)
    subprocess.run(cmd)



def main():
    cli()

if __name__ == "__main__":
    main()