import os
import sys

import click
import subprocess


MODULE_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(MODULE_DIR, "scripts")


@click.group()
def cli():
    """Bumblebee: A flexible training and inference framework."""
    pass


@cli.command()
@click.option(
    "--command",
    type=click.Choice(["deepspeed", "torchrun"]),
    help="Allowed commands with `python(don't set), deepspeed, torchrun`",
    default=sys.executable,
)
@click.argument("extra_args", nargs=-1)
def train(command, extra_args):
    script_filename = "train.py"
    script_path = os.path.join(SCRIPTS_DIR, script_filename)
    # sys.executable == python exec
    cmd = [command, script_path] + list(extra_args)
    subprocess.run(cmd)



def main():
    cli()

if __name__ == "__main__":
    main()