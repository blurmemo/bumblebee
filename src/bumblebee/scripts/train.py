import argparse
import logging

from bumblebee.utils import load_module
from bumblebee.trainer import Trainer



"""
# Full training
# `--args args.py` defines `args, (model, train_dataloader, eval_dataloader, optimizer, scheduler, evaluate) which is optional`
# `--model model.py etc. will overwrite --args args.py variants defined`
python bumblebee/scripts/train.py \
    --args args.py \
    --model model.py \
    --dataloader dataloader.py \
    --optimizer optimizer.py \
    --scheduler scheduler.py \
    --evaluate evaluate.py
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser(description="Bumblebee Trainer train.")
    parser.add_argument("--args", type=str, required=True, help="Path to a py file that defines args, (model, dataloader, optimizer, scheduler, evaluate).")
    parser.add_argument("--model", type=str, help="Path to a py file that defines `model=...`.")
    parser.add_argument("--dataloader", type=str, help="Path to a py file that defines `train_dataloader=..., eval_dataloader=...`.")
    parser.add_argument("--optimizer", type=str, help="Path to a py file that defines `optimizer=...`.")
    parser.add_argument("--scheduler", type=str, help="Path to a py file that defines `scheduler=...`.")
    parser.add_argument("--evaluate", type=str, help="Path to a py file that defines `evaluate=...` which is evaluation function.")
    parser.add_argument()
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args



def main():
    args_parsed, unknown_args_parsed = make_parser()
    arguments_path, model_path, dataloader_path, optimizer_path, scheduler_path, evaluate_path = args_parsed.args, args_parsed.model, args_parsed.dataloader, args_parsed.optimizer, args_parsed.scheduler, args_parsed.evaluate
    args_module = load_module(arguments_path)
    model_module = load_module(model_path) if model_path else args_module
    dataloader_module = load_module(dataloader_path) if dataloader_path else args_module
    optimizer_module = load_module(optimizer_path) if optimizer_path else args_module
    scheduler_module = load_module(scheduler_path) if scheduler_path else args_module
    evaluate_module = load_module(evaluate_path) if evaluate_path else args_module

    trainer_args = getattr(args_module, "args", None)
    model = getattr(model_module, "model", None)
    train_dataloader = getattr(dataloader_module, "train_dataloader", None)
    eval_dataloader = getattr(dataloader_module, "eval_dataloader", None)
    optimizer = getattr(optimizer_module, "optimizer", None)
    scheduler = getattr(scheduler_module, "scheduler", None)
    evaluate = getattr(evaluate_module, "evaluate", None)

    assert trainer_args and model and train_dataloader, logger.error(
        "Your `args, model and train_dataloader` are not defined in file."
        "Define as follows: args=`dict`, model=..., train_dataloader=... and so on"
    )

    trainer = Trainer(
        trainer_args,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        evaluate=evaluate,
    )

    logger.info("Starting training...")

    # Train!
    trainer.train()

if __name__ == '__main__':
    main()

