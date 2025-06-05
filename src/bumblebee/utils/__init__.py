import random
import torch
import numpy as np

from .import_utils import _LazyModule, load_module

from .typology import (
    ExplicitEnum,
    MixedPrecisionStrategy,
    IntervalStrategy,
    SaveStrategy,
    CheckpointStrategy,
    XProcessorUnit,
    DistributedType,
    PrecisionType,
    SchedulerType,
    OptimType,
    TunerType,
    LoRAType
)


from .io import (
    validate_file_path,
    read_json,
    write_json
)


from .trainer_utils import (
    TrainOutput,
    EvalOutput,
    ModelStageState,
    TrainerState,
    EvalerState,
    DistributedState
)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)

