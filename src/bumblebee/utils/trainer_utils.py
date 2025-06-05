import logging

from contextlib import contextmanager

import numpy as np

import torch
import torch.distributed as dist

from collections.abc import Mapping

from typing import NamedTuple, Union, Optional

import deepspeed

from .typology import (
    XProcessorUnit,
    DistributedType,
    DistributedBackend
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# trainer storage from transformers `trainer_pt.utils`
# ---------------------------------------------------------------------------------------------
def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, : tensor2.shape[1]] = tensor2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    if not (isinstance(tensors, torch.Tensor) and isinstance(new_tensors, torch.Tensor)):
        assert type(tensors) is type(new_tensors), (
            f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
        )
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

def nested_numpy(tensors):
    "Numpy `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpy(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpy(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()


class TensorContainer:
    """
    Args:
        nested: (`bool`, *optional*, defaults to `False`):
            if `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.

    """
    def __init__(self, nested: bool = False, padding_index: int = -100):
        self.nested = nested
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def append(self, tensor) -> None:
        """Add tensors to the stored objects. If `nested=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensor if self.nested else [tensor]
        elif self.nested:
            self.tensors = nested_concat(self.tensors, tensor, padding_index=self.padding_index)
        else:
            self.tensors.append(tensor)

    def extend(self, tensors) -> None:
        """ tensors is list[Tensor]"""
        if self.tensors is None:
            self.tensors = tensors if self.nested else list(torch.unbind(tensors, dim=0))
        elif self.nested:
            self.tensors = nested_concat(self.tensors, tensors, padding_index=self.padding_index)
        else:
            self.tensors.extend(torch.unbind(tensors, dim=0))


    def numpy(self) -> list:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        if self.tensors is None:
            return self.arrays if self.arrays is not None else []

        new_arrays = nested_numpy(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.nested:
            self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
        else:
            self.arrays.extend(new_arrays)

        # reset device tensors after adding to cpu
        self.tensors = None
        return self.arrays


class TrainOutput(NamedTuple):
    pass


class EvalOutput(NamedTuple):
    """
    Args:
        step: total evaluation step which is not allways equal to dataloader length
        loss: evaluation average loss across samples and devices
        predictions: evaluation results for every batch samplers which are not decoded
        labels: every batch samplers grounding truth which are not decoded
        metrics: dictionary including loss_steps, perplexity_steps, etc.

    """
    step: Optional[int]
    loss: Optional[float]
    predictions: Optional[Union[np.ndarray, tuple[np.ndarray]]]
    labels: Optional[Union[np.ndarray, tuple[np.ndarray]]]
    metrics: Optional[dict[str, Union[float, tuple[np.ndarray]]]]
# ---------------------------------------------------------------------------------------------
# trainer storage from transformers `trainer_pt.utils`
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# trainer state
# ---------------------------------------------------------------------------------------------
class DistributedState:
    framework = "pt"


    _attrs = [
        "process_unit",
        "dist_type",
        "dist_args",
        "backend",
        "device",
        "group",
        "local_rank",
        "rank",
        "world_size",
        "is_main_process"
        "is_local_main_process"
    ]


    def __init__(self, dist_type: Optional[Union[str, DistributedType]] = None, dist_args: Optional[dict] = None):
        self.process_unit = XProcessorUnit.CPU
        self.dist_type = DistributedType(dist_type)
        self.dist_args = dist_args

        ##### distributed operation #####
        self.process_unit, self.backend = self._prepare_backend()
        self._group, self._local_rank, self._rank, self._world_size = self._init_process_group(self.dist_type, self.backend)
        self._device = self.set_device()
        ##### distributed operation #####



    def _prepare_backend(self) -> (Optional[Union[str, XProcessorUnit]], Optional[Union[str, DistributedBackend]]):
        if torch.cuda.is_available():
            pu = XProcessorUnit.GPU
            backend = DistributedBackend.NCCL
        else:
            raise ValueError("Your xPU is not supported.")
        return pu, backend


    def _init_process_group(self, dist_type: Union[DistributedType, str] = None, backend: Union[DistributedBackend, str] = None) -> tuple[int, int, int, int]:
        if backend is None:
            raise ValueError("`backend` is required by `_prepare_backend` but None.")

        if isinstance(backend, DistributedBackend):
            backend = backend.value

        if dist_type == DistributedType.DEEPSPEED:
            deepspeed.init_distributed()
        else:
            dist.init_process_group(backend)

        group, local_rank, rank, world_size = 0, 0, 0, 1
        if self.process_unit == XProcessorUnit.GPU:
            local_rank = dist.get_node_local_rank()
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        return group, local_rank, rank, world_size


    def set_device(self) -> "torch.device":
        if self.process_unit == XProcessorUnit.GPU:
            device = torch.device("cuda", self._local_rank)
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        else:
            raise TypeError("Your xPU is not supported.")
        return device


    def wait(self):
        dist.barrier()


    def destroy_group(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


    def _run_first(self, is_main: bool):
        if not is_main:
            self.wait()

        yield

        if is_main:
            self.wait()

    def __repr__(self) -> str:
        return (
            f"Distributed environment: {self.dist_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"World Size: {self.world_size}\n"
            f"Rank: {self.rank}\n"
            f"Local Rank: {self.local_rank}\n"
            f"Device: {self.device}\n"
        )

    @contextmanager
    def main_process_priority(self):
        yield from self._run_first(self.is_main_process)

    @contextmanager
    def local_main_process_priority(self):
        yield from self._run_first(self.is_local_main_process)


    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def group(self) -> int:
        return self._group

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size


    @property
    def device(self) -> torch.device:
        return self._device


    def __getattr__(self, name: str):
        # By this point we know that no attributes of `self` contain `name`,
        # so we just modify the error message
        if name in self._attrs:
            raise AttributeError(
                f"`DistributedState` object has no attribute `{name}`. "
            )
        # Raise a typical AttributeError
        raise AttributeError(f"'DistributedState' object has no attribute '{name}'")


class ModelStageState:
    """
    Args
        args: TrainArguments
        stage: True is train stage, False is eval stage
    """
    def __init__(self, args = None, stage: bool = True):
        self.args = args
        self._stage = stage
        ##### common state #####
        self.step = 0
        self.epoch = 0
        self.loss = 0.0
        self.step_loss = []
        self.epoch_loss = []
        ##### common state #####
        self._validate_params()
        self._prepare_state()


    def _validate_params(self):
        """ TrainerState and EvalerState `validate_params` function are different """
        raise NotImplementedError

    def _prepare_state(self):
        """ prepare unique and needed train or eval state """
        raise NotImplementedError


    def update_step(self, *args, **kwargs):
        """ update state every train or eval step """
        raise NotImplementedError


    def update_epoch(self, *args, **kwargs):
        """ update state every train or eval epoch """
        raise NotImplementedError



    @property
    def stage(self) -> str:
        return "train" if self._stage else "eval"



class TrainerState(ModelStageState):
    def __init__(self, args = None):
        super().__init__(args, stage=True)


    def _validate_params(self):
        logger.warning("`TrainerState` requires `TrainArguments`.")

    def _prepare_state(self):
        args = self.args
        # self.max_steps = args.max_steps
        # self.lr = args.lr

        self.max_steps_reached = False
        self.epoch_lr = []
        self.log_step = 0


    def update_step(self, loss):
        """ add every step loss to step_loss """
        self.step += 1
        self.step_loss.append(loss)


    def update_epoch(self, loss_epoch, epoch: int = 0, lr: float = 0.0):
        self.loss = loss_epoch
        self.epoch_loss.append(loss_epoch)
        self.epoch = epoch
        self.lr = lr
        # record
        self.epoch_lr.append(lr)

    @property
    def end(self):
        return self.max_steps_reached



class EvalerState(ModelStageState):

    def __init__(self, args = None):
        super().__init__(args, stage=False)

    def _validate_params(self):
        logger.warning("`EvalerState` requires `TrainArguments`.")

    def _prepare_state(self):
        args = self.args
        self.best_loss = float("inf")
        self.predictions = []
        self.labels = []
        self.metrics = {}

    def update_step(self):
        logger.warning("`EvalerState` does not support `update_step`. Please use `update` instead.")

    def update_epoch(self):
        logger.warning("`EvalerState` does not support `update_epoch`. Please use `update` instead.")

    def update(self, step, loss, predictions, labels, metrics):
        """
        Args:
            step: Optional[int] eval total step
            loss: Optional[float] eval average loss
            predictions: Optional[Union[np.ndarray, tuple[np.ndarray]]] model predictions without decoding
            labels: Optional[Union[np.ndarray, tuple[np.ndarray]]] model truth labels without decoding
            metrics: Optional[dict[str, Union[float, tuple[np.ndarray]]]] model metrics, including loss, perplexity, step_loss and step_perplexity.
        """
        self.step = step
        self.loss = loss
        self.predictions = predictions
        self.labels = labels
        self.metrics = metrics
        self.epoch_loss.append(loss)
        # update best_loss
        if self.best_loss > loss: self.best_loss = loss

    @property
    def model_save(self):
        # if update `best_loss`, it will replace by `loss`
        return self.best_loss == self.loss
# ---------------------------------------------------------------------------------------------
# trainer state
# ---------------------------------------------------------------------------------------------






