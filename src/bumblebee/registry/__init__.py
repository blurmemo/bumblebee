from typing import TYPE_CHECKING
from ..utils import _LazyModule


_import_structure = {
    "dataloader": ["DataloaderRegistry", "DistributedDataLoaderRegistry", "DistributedSampler",],
    "distributed": ["DistributedRegistry", "DistributedArguments", "DeepSpeedArguments",],
    "lr_scheduler": ["LRSchedulerRegistry", "LRSchedulerArguments", "StepLRArguments",],
    "optimizer": ["OptimizerRegistry", "OptimArguments", "AdamArguments", "AdamWArguments",],
    "tuner": ["TunerRegistry", "TunerArguments", "LoRAArguments", "LinearLoRAArguments",],
    "tracer": ["TracerRegistry", "TracerArguments", "WandbArguments",],
    "snapshot": ["SnapshotRegistry",],
}

if TYPE_CHECKING:
    from .dataloader import DataloaderRegistry, DistributedDataLoaderRegistry, DistributedSampler
    from .distributed import DistributedRegistry, DistributedArguments, DeepSpeedArguments
    from .lr_scheduler import LRSchedulerRegistry, LRSchedulerArguments, StepLRArguments
    from .optimizer import OptimizerRegistry, OptimArguments, AdamArguments, AdamWArguments
    from .tuner import TunerRegistry, TunerArguments, LoRAArguments, LinearLoRAArguments
    from .tracer import TracerRegistry, TracerArguments, WandbArguments
    from .snapshot import SnapshotRegistry
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__   # type: ignore
    )