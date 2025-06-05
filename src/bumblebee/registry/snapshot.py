import logging
import os
import shutil
from enum import Enum
from functools import cached_property
from datetime import datetime
from typing import Callable

import torch
import torch.distributed as dist
import deepspeed
import transformers
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.distributed.fsdp import FullyShardedDataParallel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class SnapshotType(Enum):
    PT = "pt"
    DEEPSPEED = "deepspeed"
    HF = "hf"
    FSDP = "fsdp"




class SnapshotRegistry:
    def __init__(self, output_dir: str, limit: int):
        self.output_dir = output_dir
        self.limit = limit
        self._output_dirs = []
        self.validate_init_params()


    def validate_init_params(self):
        if self.output_dir is None:
            raise ValueError("`output_dir` is None but required for snapshot.")
        if self.limit <= 0:
            raise ValueError(f"`snapshot limit`: {self.limit} must be > 0.")


    def fetch_save_type(self, model):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            save_type = SnapshotType.DEEPSPEED
        elif isinstance(model, FullyShardedDataParallel):
            save_type = SnapshotType.FSDP
        elif isinstance(model, transformers.PreTrainedModel):
            save_type = SnapshotType.HF
        elif isinstance(model, torch.nn.Module):
            save_type = SnapshotType.PT
        else:
            raise NotImplementedError("The `save function` for your model type is not supported.")
        return save_type

    def unify_output_dir(self, output_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # broadcast
        if self.enable_dist:
            timestamps = [timestamp]
            dist.broadcast_object_list(timestamps, src=0)
            timestamp = timestamps[0]
        output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def limit_snapshot(self, output_dir: str):
        self._output_dirs.append(output_dir)
        output_dirs = self._output_dirs
        num_exported = len(self._output_dirs)  # start from 1
        if num_exported > self.limit:
            head_output_dir = output_dirs.pop(0)
            # only rank=0 delete dir in dist or one gpu
            if os.path.exists(head_output_dir) and self.rank == 0:
                shutil.rmtree(head_output_dir)
        return output_dirs[-1]


    def set_workflow(self, output_dir, save_type: SnapshotType, model, optimizer=None, scheduler=None) -> list[tuple[Callable, tuple]]:
        rank = self.rank  # default rank=0
        workflows = []
        # args for save_model and save_optimizer
        save_model_args = (output_dir, model)
        save_optim_args = (output_dir, optimizer)


        if save_type == SnapshotType.DEEPSPEED:
            # save_ds_model can save model, resume which has optimizer
            workflows.append((save_ds_model, (*save_model_args, rank)))
            # save optimizer with zero 0 which disable zero
            if rank == 0 and optimizer is not None and not model.zero_optimization():
                workflows.append((save_pt_optimizer, save_optim_args))
        elif save_type == SnapshotType.FSDP:
            pass
        elif save_type == SnapshotType.HF and rank == 0:
            workflows.append((save_hf_model, save_model_args))
            # save optimizer
            if optimizer is not None: workflows.append((save_pt_optimizer, save_optim_args))
        elif save_type == SnapshotType.PT and rank == 0:
            workflows.append((save_pt_model, save_model_args))
            # save optimizer
            if optimizer is not None: workflows.append((save_pt_optimizer, save_optim_args))
        else:
            raise ValueError("`set_workflow` may happen some errors when constructing snapshot.")

        # scheduler is controlled manually so we default use torch.save(..) to save scheduler
        if scheduler is not None and rank == 0:
            save_scheduler_args = (output_dir, scheduler)
            workflows.append((save_pt_scheduler, save_scheduler_args))

        return workflows


    def __call__(self, model, optimizer=None, scheduler=None):
        output_dir = self.unify_output_dir(self.output_dir)
        output_dir = self.limit_snapshot(output_dir)
        save_type = self.fetch_save_type(model)
        workflows = self.set_workflow(output_dir, save_type, model, optimizer, scheduler)
        for callback, args in workflows:
            callback(*args)



    @cached_property
    def enable_dist(self):
        return dist.is_available() and dist.is_initialized()

    @cached_property
    def rank(self):
        rank = 0
        if self.enable_dist:
            rank = dist.get_rank()
        return rank




def save_pt_model(output_dir: str, model: torch.nn.Module):
    logger.info("***** Save pt model *****")
    output_file = os.path.join(output_dir, f"model_weights.pth")
    torch.save(model.state_dict(), output_file)


def save_hf_model(output_dir: str, model: "transformers.PreTrainedModel"):
    logger.info("***** Save hf model *****")
    model.save_pretrained(output_dir)


def save_ds_model(output_dir: str, model: "deepspeed.DeepSpeedEngine", rank: int = 0):
    logger.info(f"***** Save ds model (rank:{rank}) *****")
    model_save = model.module if hasattr(model, "module") else model
    enable_zero = model.zero_optimization()
    state_dict = {}
    if enable_zero and model.zero_optimization_stage() == 3:
        ##### helper function #####
        def _fetch_zero3_param(params):
            return [
                p for p in params
                if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ]
        ##### helper function #####
        for k, v in model_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_fetch_zero3_param([v]), enabled=True):
                    v_p = v.data.cpu()  # this is a hack to get around the fact that we can't get the data from the param
            else:
                v_p = v.cpu()
            if rank == 0: # and "lora" not in k:
                state_dict[k] = v_p
    else:
        state_dict = model_save.state_dict()

    output_file = os.path.join(output_dir, "model_weights.bin")
    if rank == 0:
        torch.save(state_dict, output_file)
    del state_dict
    # save to resume
    resume_dir = os.path.join(output_dir, "resume")
    model.save_checkpoint(resume_dir)


def save_fsdp_model(output_dir: str, model: "FullyShardedDataParallel"):
    raise NotImplementedError


def save_pt_scheduler(output_dir: str, scheduler: "torch.optim.lr_scheduler.LRScheduler"):
    logger.info("***** Save scheduler *****")
    output_file = os.path.join(output_dir, f"scheduler.pth")
    torch.save(scheduler.state_dict(), output_file)


def save_pt_optimizer(output_dir: str, optimizer: "torch.optim.Optimizer"):
    logger.info("***** Save optimizer *****")
    output_file = os.path.join(output_dir, f"optimizer.pth")
    torch.save(optimizer.state_dict(), output_file)





