from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class MixedPrecisionStrategy(ExplicitEnum):
    AUTO = "auto"
    CUDA_AMP = "cuda_amp"


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    EPOCH = "epoch"
    STEP = "step"


class SaveStrategy(ExplicitEnum):
    NO = "no"
    BEST = "best"


class CheckpointStrategy(ExplicitEnum):
    AUTO = "auto"
    MODEL = "model"


class XProcessorUnit(ExplicitEnum):
    CPU = "cpu"
    GPU = "gpu"


class DistributedType(ExplicitEnum):
    DEEPSPEED = "deepspeed"


class DistributedBackend(ExplicitEnum):
    NCCL = "nccl"


class PrecisionType(ExplicitEnum):
    FP16 = "fp16"
    BF16 = "bf16"


class SchedulerType(ExplicitEnum):
    """
    Scheduler names for the parameter `lr_scheduler_type` in [`TrainingArguments`].
    By default, it uses "linear". Internally, this retrieves `get_linear_schedule_with_warmup` scheduler from [`Trainer`].
    Scheduler types:
       - "linear" = get_linear_schedule_with_warmup
       - "cosine" = get_cosine_schedule_with_warmup
       - "cosine_with_restarts" = get_cosine_with_hard_restarts_schedule_with_warmup
       - "polynomial" = get_polynomial_decay_schedule_with_warmup
       - "constant" =  get_constant_schedule
       - "constant_with_warmup" = get_constant_schedule_with_warmup
       - "inverse_sqrt" = get_inverse_sqrt_schedule
       - "reduce_lr_on_plateau" = get_reduce_on_plateau_schedule
       - "cosine_with_min_lr" = get_cosine_with_min_lr_schedule_with_warmup
       - "warmup_stable_decay" = get_wsd_schedule
    """
    # LINEAR = "linear"
    # COSINE = "cosine"
    # COSINE_WITH_RESTARTS = "cosine_with_restarts"
    # POLYNOMIAL = "polynomial"
    # CONSTANT = "constant"
    # CONSTANT_WITH_WARMUP = "constant_with_warmup"
    # INVERSE_SQRT = "inverse_sqrt"
    # REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    # COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    # WARMUP_STABLE_DECAY = "warmup_stable_decay"
    STEP = "step"


class OptimType(ExplicitEnum):
    ADAM = "adam"
    ADAMW = "adamw"



class TunerType(ExplicitEnum):
    LORA = "lora"

class LoRAType(ExplicitEnum):
    LINEAR = "linear"



