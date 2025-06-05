from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Union
import logging
import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from ..utils.typology import OptimType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class OptimArguments:
    params: ParamsT = None



@dataclass
class AdamArguments(OptimArguments):
    lr: Union[float, Tensor] = 1e-3
    weight_decay: float = 0.0
    betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999)
    eps: float = 1e-8



@dataclass
class AdamWArguments(AdamArguments):
    weight_decay: float = 1e-2



class OptimizerRegistry:
    framework = "bumblebee"


    def __init__(self, optim_type: Optional[Union[OptimType, str]] = None, optim_args: Optional[Union[OptimArguments, dict]] = None):
        self.optim_type = optim_type
        self.optim_args = optim_args
        self.validate_init_params()


    def validate_init_params(self):
        if self.optim_type is None:
            logger.info("`optim_type` is None. Default use `OptimType.ADAMW`.")
            self.optim_type = OptimType.ADAMW
        else:
            self.optim_type = OptimType(self.optim_type)

        if self.optim_args is None:
            logger.info("`optim_args` is None. Default use `optimizer` init params.")
            self.optim_args = {}



    def __call__(self, params: ParamsT = None) -> torch.optim.Optimizer:
        if params is None:
            raise ValueError("`params` is None but params (model parameters) are required.")
        if isinstance(self.optim_args, dict):
            self.optim_args.update({"params": params})
        elif isinstance(self.optim_args, OptimArguments):
            self.optim_args.params = params
        else:
            raise TypeError("Unsupported optim_args type.")

        callback = self.MAPPING[self.optim_type]

        return callback(self.optim_args)


    @cached_property
    def MAPPING(self):
        return {
            OptimType.ADAM: self.adam,
            OptimType.ADAMW: self.adamw,
        }


    def adam(self, optim_args):
        if isinstance(optim_args, dict):
            optim_args = AdamArguments(**optim_args)
        return torch.optim.Adam(
            optim_args.params,
            lr=optim_args.lr,
            weight_decay=optim_args.weight_decay,
            betas=optim_args.betas,
            eps=optim_args.eps,
        )


    def adamw(self, optim_args):
        if isinstance(optim_args, dict):
            optim_args = AdamWArguments(**optim_args)
        return torch.optim.AdamW(
            optim_args.params,
            lr=optim_args.lr,
            weight_decay=optim_args.weight_decay,
            betas=optim_args.betas,
            eps=optim_args.eps,
        )










