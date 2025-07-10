import importlib
from dataclasses import dataclass, asdict
from typing import Optional, Union, List

import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TracerArguments:
   pass


@dataclass
class WandbArguments(TracerArguments):
    project: str = None
    entity: Optional[str] = None
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None



class TracerRegistry:
    sdk = "wandb"

    def __init__(self, wandb_args: Optional[Union[TracerArguments, dict]] = None):
        self.wandb_args = wandb_args
        if self.wandb_args is None:
            self.wandb_args = {}


    def __call__(self, config_args = None, output_dir = None) -> "wandb.sdk.wandb_run.Run":
        if isinstance(self.wandb_args, TracerArguments):
            self.wandb_args = asdict(self.wandb_args)
        project = self.wandb_args.get("project", None)
        if project is None:
            logger.info(
                "`wandb` tracer need `project` argument, but not supported. Use default `wandb.project=bumblebee`.")
            self.wandb_args.update({"project": "bumblebee"})
        # import
        try:
            wandb = importlib.import_module("wandb")
            logger.info("You are using `wandb` as your tracer and importing `wandb library` successfully.")
        except ImportError:
            raise ImportError(
                "You are trying to use wandb which is not currently installed. "
                "Please install it using `pip install wandb`."
            )

        init_params_dict = self.wandb_args
        if output_dir is not None:
            init_params_dict["dir"] = output_dir

        run = wandb.init(**init_params_dict)
        if config_args is not None:
            run.config.update(config_args)
        return run




