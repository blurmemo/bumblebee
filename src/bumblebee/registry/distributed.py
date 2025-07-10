from dataclasses import dataclass, field, fields
from enum import Enum
from functools import cached_property
from typing import Optional

import inspect
import logging
import torch
import deepspeed


from ..utils.typology import DistributedType
from ..utils.trainer_utils import DistributedState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedArguments:

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    @classmethod
    def from_dict(cls, **kwargs):
        cls_fields = {f.name for f in fields(cls)}
        cls_attr, extra_attr = {}, {}

        for k, v in kwargs.items():
            if k in cls_fields:
                cls_field = cls.__dataclass_fields__[k]
                if inspect.isfunction(cls_field.default_factory):
                    cls_v = cls_field.default_factory()
                else:
                    cls_v = cls_field.default

                if isinstance(cls_v, dict):
                    cls_v.update(v)
                else:
                    cls_v = v  # int, bool etc.

                cls_attr[k] = cls_v

            else:
                extra_attr[k] = v

        instance = cls(**cls_attr)

        for k, v in extra_attr.items():
            setattr(instance, k, v)

        return instance



@dataclass
class DeepSpeedArguments(DistributedArguments):
    max_train_batch_size: int = None
    train_micro_batch_size_per_gpu: int = 8
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    prescale_gradients: bool = False
    fp16: dict = field(default_factory=lambda: {
        "enabled": False,
        "loss_scale_window": 100,
    })
    bf16: dict = field(default_factory=lambda: {
        "enabled": False,
    })
    zero_optimization: dict = field(default_factory=lambda: {
        "stage": 0,
        "offload_param": {
            "device": "none"  # cpu
        },
        "offload_optimizer": {
            "device": "none"  # cpu
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 0,
        "memory_efficient_linear": False,
    })
    steps_per_print: int = 1e5
    wall_clock_breakdown: bool = False



class DistributedRegistry:
    framework = "bumblebee"


    def __init__(self, dist_state: Optional[DistributedState] = None):
        self.dist_state = dist_state
        self.dist_type = dist_state.dist_type
        self.dist_args = dist_state.dist_args  # dist_state.dist_args is dict type


    def __call__(
        self,
        model: torch.nn.Module = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_batch_size_per_device: int = -1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        model_dtype: torch.dtype = None,
        **kwargs
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        if model is None or optimizer is None:
            raise ValueError("`model` or `optimizer` must be required.")

        if train_batch_size_per_device < 1:
            raise ValueError("`train_batch_size_per_device` must be required and >=1.")

        if gradient_accumulation_steps < 1:
            raise ValueError(f"`gradient_accumulation_steps`={gradient_accumulation_steps} must be >=1.")

        if max_grad_norm < 0:
            raise ValueError(f"`max_grad_norm`={max_grad_norm} must be > 0.0")

        logger.warning(
            "Some parameters in `TrainArguments` will be prioritized over your `distributed_args`"
            "such as: [`train_batch_size_per_device`, `gradient_accumulation_steps`, `max_grad_norm`...]"
        )
        train_batch_size = self.dist_state.world_size * train_batch_size_per_device * gradient_accumulation_steps
        dist_args = self.dist_args

        dist_args.update({
            "train_micro_batch_size_per_gpu": train_batch_size_per_device,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": max_grad_norm,
            "fp16": {
                "enabled": model_dtype == torch.float16,
            },
            "bf16": {
                "enabled": model_dtype == torch.bfloat16,
            }
        })

        callback = self.MAPPING[self.dist_type]

        return callback(dist_args, model, optimizer, lr_scheduler)



    @cached_property
    def MAPPING(self):
        return {
            DistributedType.DEEPSPEED: self.deepspeed,
        }


    def deepspeed(
        self,
        dist_args: dict,
        model: torch.nn.Module = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        dist_args = DeepSpeedArguments.from_dict(**dist_args)
        wrappers = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=dist_args.to_dict(),
            dist_init_required=True,
        )
        model = wrappers[0]
        optimizer = wrappers[1]
        if lr_scheduler is not None: lr_scheduler = wrappers[-1]
        return model, optimizer, lr_scheduler




