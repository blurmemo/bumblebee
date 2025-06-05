from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Union

import logging
import torch

from ..utils.typology import SchedulerType



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LRSchedulerArguments:
    optimizer: torch.optim.Optimizer = None



@dataclass
class StepLRArguments(LRSchedulerArguments):
    step_size: int = 1
    gamma: float = 0.1




class LRSchedulerRegistry:
    framework = "bumblebee"


    def __init__(self, scheduler_type: Optional[Union[SchedulerType, str]] = None, scheduler_args: Optional[Union[LRSchedulerArguments, dict]] = None):
        self.scheduler_type = scheduler_type
        self.scheduler_args = scheduler_args
        self.validate_init_params()


    def validate_init_params(self):
        if self.scheduler_type is None:
            logger.info("`scheduler_type` is None. Default use `SchedulerType.STEP`.")
            self.scheduler_type = SchedulerType.STEP
        else:
            self.scheduler_type = SchedulerType(self.scheduler_type)

        if self.scheduler_args is None:
            logger.info("`scheduler_args` is None. Default use `lr_scheduler` init params.")
            self.scheduler_args = {}



    def __call__(self, optimizer: torch.optim.Optimizer = None, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        if optimizer is None:
            raise ValueError("`optimizer` cannot be None. Please pass `optimizer`.")
        if isinstance(self.scheduler_args, dict):
            self.scheduler_args.update({"optimizer": optimizer})
        elif isinstance(self.scheduler_args, LRSchedulerArguments):
            self.scheduler_args.optimizer = optimizer
        else:
            raise TypeError("Unsupported scheduler_args type.")
        callback = self.MAPPING[self.scheduler_type]
        return callback(self.scheduler_args)


    @cached_property
    def MAPPING(self):
        return {
            SchedulerType.STEP: self.steplr,
        }


    def steplr(self, scheduler_args):
        if isinstance(scheduler_args, dict):
            scheduler_args = StepLRArguments(**scheduler_args)
        return torch.optim.lr_scheduler.StepLR(
            scheduler_args.optimizer,
            step_size=scheduler_args.step_size,
            gamma=scheduler_args.gamma,
        )








