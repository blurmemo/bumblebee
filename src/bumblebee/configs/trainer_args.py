import json
import os
import random
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Any
import warnings
import torch
import logging

from ..utils import (
    MixedPrecisionStrategy,
    IntervalStrategy,
    DistributedType,
    SchedulerType,
    PrecisionType,
    OptimType,
    CheckpointStrategy,
    SaveStrategy,
    DistributedState,
    TunerType,
)

from ..utils import (
    set_seed,
    read_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainArguments:
    framework = "pt"

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    _n_gpu: int = field(init=False, repr=False, default=-1)

    train: bool = field(
        default=False,
        metadata={"help": "Whether to train."}
    )

    eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the validation set."}
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions, checkpoints and logs will be written."
        },
    )

    # bf16: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Whether to use bf16 instead of 32-bit. Requires Ampere or higher NVIDIA"
    #             " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
    #         )
    #     },
    # )
    #
    # fp16: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use fp16 instead of 32-bit"},
    # )

    mixed_precision_dtype: Union[PrecisionType, str] = field(
        default=None,
        metadata={
            "help": "The precision to be used for mixed precision. no: full fp32.",
            "choices": ["fp16", "bf16"],
        }
    )

    mixed_precision_backend: Union[MixedPrecisionStrategy, str] = field(
        default=None,
        metadata={
            "help": "The backend to be used for mixed precision.",
            "choices": ["auto", "cuda_amp"],
        },
    )

    train_batch_size_per_device: int = field(
        default=1, metadata={"help": "Batch size per device core/CPU for training."}
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_epochs: int = field(default=0, metadata={"help": "Total number of training epochs to perform. Default calculate `max_steps`."})

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_epochs."},
    )

    ######### eval #########
    eval_batch_size_per_device: int = field(
        default=1, metadata={"help": "Batch size per device core/CPU for evaluation."}
    )

    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy (no, epoch, step) to use."},
    )

    eval_skip_first: bool = field(
        default=True,
        metadata={"help": "Whether to skip the first evaluation (step=0) when evaling."}
    )

    eval_delay: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " eval_strategy."
            )
        },
    )

    eval_intervals: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "eval interval with `eval_strategy`."
                "Default is `train_dataloader` length."
                "Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of X `eval_strategy`."
            )
        }
    )

    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Eval run total X steps."
                "Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total evaluation steps."
            )
        },
    )
    ######### eval #########

    distributed_type: Union[DistributedType, str] = field(
        default=None,
        metadata={
            "help": "distributed library mode.",
            "choices": ["deepspeed"],
        }
    )


    distributed_args: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with distributed. The value is either a "
                "json config file or an already loaded json file as `dict`."
            )
        },
    )

    lr: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})

    optim_type: Optional[Union[OptimType, str]] = field(
        default=None,
        metadata={"help": "The optimizer type to use."},
    )

    optim_args: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "Optional arguments except `params` which are `model parameters` to supply to optimizer."}
    )

    lr_scheduler_type: Optional[Union[SchedulerType, str]] = field(
        default=None,
        metadata={"help": "The scheduler type to use."},
    )

    lr_scheduler_args: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for LRScheduler expect `optimizer`."
            )
        },
    )

    tuner_type: Optional[Union[TunerType, str]] = field(
        default=None,
        metadata={"help": "The tuner type to use."},
    )

    tuner_args: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Extra parameters for the tuner such as {'rank': 4} for LoRA etc."
            )
        },
    )

    # current not support
    # warmup_delay: float = field(
    #     default=0.0,
    #     metadata={
    #         "help": (
    #             "Linear warmup over warmup_steps."
    #             "Should be an integer or a float in range `[0,1)`. "
    #             "If smaller than 1, will be interpreted as ratio of total training steps."
    #         )
    #     }
    # )

    checkpoint_strategy: Union[CheckpointStrategy, str] = field(
        default="auto",
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "auto: save model, optimizer, scheduler and rng state; model: save only model."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )

    # current not support, we suggest resume before Trainer.train()
    # resume_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    # )

    save_strategy: Union[SaveStrategy, str] = field(
        default="best",
        metadata={
            "help": (
                "The checkpoint save strategy to use."
                "no: don't save, best: save best"
            ),
            "choice": ["no", "best"],
        },
    )

    save_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`."
                " When `save_limit=1`, it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints which is `save_limit=0`."
            )
        },
    )

    log_dir: Optional[str] = field(default=None, metadata={"help": "log dir."})

    log_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={
            "help": (
                "The logging strategy to use."
                "no: don't log, auto: log every eval, epoch: every epoch and step: every X step."
            )
        },
    )

    log_intervals: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "log interval with `log_strategy`."
                "Default is `train_dataloader` length."
                "Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of X `log_strategy`."
            )
        }
    )

    wandb: Optional[bool] = field(
        default=False,
        metadata={
            "help": "`wandb` which is to trace experiments and log metrics such as `loss`, `perplexity`, etc..."
        }
    )

    wandb_args: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "`wandb_args` is used to init `wandb_run`.",
                "What you need to provide are initial parameters for `wandb`."
                "`wandb params(wandb_args key name)`: "
                "Must: project."
                "Optional: entity[str], job_type[str], tags[list[str]], group[str], notes[str], mode[str]."
            )
        }
    )


    def __post_init__(self):
        SEED_ATTEN = "***** You must set the random seed at the beginning of the code execution position! *****"
        for _ in range(3):
            logger.warning(SEED_ATTEN)

        # set random
        set_seed(self.seed)


        if self.output_dir is None:
            self.output_dir = os.path.join(Path.home().absolute(), "output")
        else:
            self.output_dir = str(Path(self.output_dir).absolute())
        self.output_dir = os.path.expanduser(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # default `train` = True
        if not self.train:
            logger.info("Your `train=False`, default enable `train=True`.")
            self.train = True


        if self.train_batch_size_per_device < 1:
            raise ValueError(f"`train_batch_size_per_device`: {self.train_batch_size_per_device} must be >= 1.")

        # if self.fp16 and self.bf16:
        #     raise ValueError("At most one of fp16 and bf16 can be True, but not both. Both are False mean use fp32.")

        if self.num_epochs < 1:
            raise ValueError(f"`num_epochs`: {self.num_epochs} is less than 1.")

        ########## typology ##########
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.log_strategy = IntervalStrategy(self.log_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)
        self.checkpoint_strategy = CheckpointStrategy(self.checkpoint_strategy)
        ########## typology ##########

        if self.eval is False and self.eval_strategy != IntervalStrategy.NO:
            self.eval = True
            logger.warning("`eval=False` but `eval_strategy` is set to enable `eval=True`.")

        if self.eval and self.eval_strategy == IntervalStrategy.NO:
            self.eval_strategy = IntervalStrategy.EPOCH
            logger.warning("`eval` enable but `eval_strategy` set to NO. Use default `eval_strategy: EPOCH` instead.")

        if self.eval and self.eval_batch_size_per_device < 1:
            raise ValueError(f"enable `eval` but `eval_batch_size_per_device`: {self.eval_batch_size_per_device} must be >= 1.")

        #vars = {"warmup_delay": self.warmup_delay}
        vars = {}
        if self.eval: vars.update({
            "eval_intervals": self.eval_intervals,
            "eval_delay": self.eval_delay,
            "eval_steps": self.eval_steps
        })
        if self.log_strategy != IntervalStrategy.NO: vars.update({
            "log_intervals": self.log_intervals,
        })

        for k, v in vars.items():
            if v is None:
                logger.info(f"`{k}` is None. Use default value.")
            elif v < 0:
                raise ValueError(f"`{k}`: {v} must be >= 0.")
            elif v > 1 and v != int(v):
                logger.warning(
                    f"`{k}`: {v} must be an integer."
                    f"`{k}` are reset to {int(v)}."
                )
                setattr(self, k, int(v))  # ensure self.attr will be updated

        if self.mixed_precision_backend is not None:
            self.mixed_precision_backend = MixedPrecisionStrategy(self.mixed_precision_backend)
            if self.mixed_precision_dtype is None:
                logger.info(f"`mixed precision backend` need `mixed_precision_dtype` but None. Default use `fp16`.")
                self.mixed_precision_dtype = PrecisionType.FP16
            else:
                self.mixed_precision_dtype = PrecisionType(self.mixed_precision_dtype)


        if self.optim_type is not None:
            self.optim_type = OptimType(self.optim_type)
            if self.optim_args is None:
                self.optim_args = {}
                logger.info("`optim_args` is None and it will use `bumblebee.registers.optimizer` default optim_args.")
            elif isinstance(self.optim_args, str):
                logger.info(f"`optim_args` are loaded from local json file and path is {self.optim_args}.")
                self.optim_args = read_json(self.optim_args)

        if self.lr_scheduler_type is not None:
            self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
            if self.lr_scheduler_args is None:
                self.lr_scheduler_args = {}
                logger.info("`lr_scheduler_args` is None and it will use `bumblebee.registers.lr_scheduler` default lr_scheduler_args.")
            elif isinstance(self.lr_scheduler_args, str):
                logger.info(f"`lr_scheduler_args` are loaded from local json file and path is {self.lr_scheduler_args}.")
                self.lr_scheduler_args = read_json(self.lr_scheduler_args)

        if self.tuner_type is not None:
            self.tuner_type = TunerType(self.tuner_type)
            if self.tuner_args is None:
                self.tuner_args = {}
                logger.info("`tuner_args` is None and it will use `bumblebee.registers.tuner` default tuner_args.")
            elif isinstance(self.tuner_args, str):
                logger.info(f"`tuner_args` are loaded from local json file and path is {self.tuner_args}.")
                self.tuner_args = read_json(self.tuner_args)

        if self.distributed_type is not None:
            self.distributed_type = DistributedType(self.distributed_type)
            if self.distributed_args is None:
                self.distributed_args = {}
                logger.info("`distributed_args` is None and it will use `bumblebee.registers.distributed` default distributed_args.")
            elif isinstance(self.distributed_args, str):
                logger.info(f"`distributed_args` are loaded from local json file and path is {self.distributed_args}.")
                self.distributed_args = read_json(self.distributed_args)

        if self.save_limit is None:
            self.save_limit = 1
            logger.info("`save_limit` is None and will be automatically set to 1. It will save the best one.")
        elif self.save_limit < 0:
            raise ValueError(f"`save_limit`: {self.save_limit} must be >= 0.")

        if self.log_strategy != IntervalStrategy.NO and self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, "logs")
            logger.info(f"`log_dir` is None and will be automatically set to `output_dir/logs/`: {self.log_dir}")

        # setup device
        if self.framework == "pt" and torch.cuda.is_available():
            self.device

        # if self.bf16 and self._n_gpu < 1:
        #     raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")

        if self.wandb:
            logger.info("`wandb` is used for tracing the process of training and evaluation.")
            if self.wandb_args is None:
                self.wandb_args = {}
                logger.info("`wandb_args` is None and it will use `bumblebee.utils.tracer` default wandb_args.")
            elif isinstance(self.wandb_args, str):
                logger.info(f"`wandb_args` are loaded from local json file and path is {self.wandb_args}.")
                self.wandb_args = read_json(self.wandb_args)




    def adapt(self, train_dataloader, eval_dataloader=None):
        """ You must call this function to adjust `TrainerArguments` after  initializing the data loader (if need) """
        if self.max_steps < 1:
            self.max_steps = self.num_epochs * len(train_dataloader)
            logger.info(f"`max_steps`: {self.max_steps} is automatically set to `num_epochs * train_dataloader length`: {self.num_epochs} * {len(train_dataloader)}.")


        INTERVAL_UNIT = {
            IntervalStrategy.EPOCH: len(train_dataloader), IntervalStrategy.STEP: 1,
        }

        if self.eval_strategy != IntervalStrategy.NO:
            interval_unit = INTERVAL_UNIT[self.eval_strategy]
            if self.eval_intervals is None:
                self.eval_intervals = len(train_dataloader)
            else:
                self.eval_intervals = int(self.eval_intervals * interval_unit)

            self.eval_delay = 0 if self.eval_delay is None else int(self.eval_delay * interval_unit)

            if self.eval_steps is None:
                self.eval_steps = len(eval_dataloader)
            elif self.eval_steps < 1:
                self.eval_steps = int(self.eval_steps * len(eval_dataloader))
            else:
                self.eval_steps = int(self.eval_steps)


        if self.log_strategy != IntervalStrategy.NO:
            interval_unit = INTERVAL_UNIT[self.log_strategy]
            if self.log_intervals is None:
                self.log_intervals = len(train_dataloader)
            else:
                self.log_intervals = int(interval_unit * self.log_intervals)

        # if self.warmup_delay is None:
        #     self.warmup_delay = 0
        # elif self.warmup_delay < 1:
        #     self.warmup_delay = int(len(train_dataloader) * self.warmup_delay)
        # else:
        #     self.warmup_delay = int(self.warmup_delay)


    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        per_device_batch_size = self.train_batch_size_per_device
        train_batch_size = per_device_batch_size * self.n_gpu * self.gradient_accumulation_steps
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        per_device_batch_size = self.eval_batch_size_per_device
        eval_batch_size = per_device_batch_size * self.n_gpu
        return eval_batch_size

    @property
    def n_gpu(self):
        return self._n_gpu

    @cached_property
    def _setup_devices(self) -> "torch.device":
        self.distributed_state = None
        if self.distributed_type is not None:
            # init distributed
            dist_state = DistributedState(dist_type=self.distributed_type, dist_args=self.distributed_args)
            self.distributed_state = dist_state
            device = dist_state.device
            self._n_gpu = dist_state.world_size
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._n_gpu = 1
        return device


    @property
    def device(self) -> "torch.device":
        return self._setup_devices  # this function will return current device id


    ##### mixed precision #####
    @property
    def amp(self) -> bool:
        return self.mixed_precision_backend == MixedPrecisionStrategy.AUTO or self.mixed_precision_backend == MixedPrecisionStrategy.CUDA_AMP

    @property
    def mixed_precision(self) -> "torch.dtype":
        mp = torch.float16  # default mixed precision
        if self.mixed_precision_dtype == PrecisionType.BF16:
            mp = torch.bfloat16
        return mp
    ##### mixed precision #####


    @cached_property
    def trace(self) -> bool:
        return self.wandb

    @cached_property
    def log(self) -> bool:
        return self.log_strategy != IntervalStrategy.NO

    @cached_property
    def save_checkpoint(self) -> bool:
        return self.save_strategy != SaveStrategy.NO

    @cached_property
    def save_model_only(self) -> bool:
        return self.checkpoint_strategy == CheckpointStrategy.MODEL

    @cached_property
    def tuner(self) -> bool:
        return self.tuner_type is not None


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
                d[k] = ", ".join([x.value for x in v])
        return d

    def to_json(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)


    def to_sanitized_dict(self) -> dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if torch.cuda.is_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}





