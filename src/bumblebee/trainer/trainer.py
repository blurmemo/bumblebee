import logging
import multiprocessing as mp
import os
from contextlib import nullcontext
from functools import cached_property
from typing import Union, Optional, Callable
from datetime import datetime

import math
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs import TrainArguments

from ..utils.io import write_json

from ..utils.trainer_utils import (
    EvalOutput,
    TrainerState,
    EvalerState
)

from ..utils.typology import DistributedType

from ..registry import (
    DataloaderRegistry,
    DistributedDataLoaderRegistry,
    OptimizerRegistry,
    LRSchedulerRegistry,
    TunerRegistry,
    DistributedRegistry,
    TracerRegistry,
    SnapshotRegistry
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Trainer:

    def __init__(
        self,
        args: Optional[Union[dict, TrainArguments]] = None,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        evaluate: Optional[Callable[[nn.Module, DataLoader,...], EvalOutput]] = None,
    ):
        ##### initialize parameters #####
        self.args = self.bind_args(args)  # TrainerArguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.autocast = nullcontext()
        self.scaler = None
        self.tracer = None
        self.distributed_state = self.args.distributed_state
        self.evaluate_loop = evaluate
        ##### validation parameters #####
        self.validate_init_params()
        ##### register components #####
        self.register_component()
        ##### setup snapshot and state after args having adapted #####
        self.state = TrainerState(self.args)
        self.eval_state = EvalerState(self.args) if self.args.eval else None
        self.snapshot = SnapshotRegistry(output_dir=self.args.output_dir, limit=self.args.save_limit)


    def bind_args(self, args):
        if not isinstance(args, TrainArguments):
            if isinstance(args, dict):
                args = TrainArguments(**args)
        else:
            raise TypeError("args must be a `TrainArguments` or `dict`")
        return args



    def validate_init_params(self):
        if self.model is None or self.args is None or self.train_dataloader is None:
            raise ValueError("`model`, `args` or `train_dataloader` is None which must be required.")
        if isinstance(self.args, dict):
            self.args = TrainArguments(**self.args)
        if self.args.eval and self.eval_dataloader is None:
            raise ValueError("`eval` enable but `eval_dataloader` is None which must be required.")

    def register_component(self):
        """
        dataloader: train and eval dataLoader (if eval) are needed to reset
        model: Interface to wrap model using tuner or other wrappers
        amp: automatic mixed-precision training
        optimizer: update model parameters
        scheduler: update learning rate
        distributed: enable distributed training or not
        evaluate: set evaluate function
        tracer: visualize the experimental process by wandb
        """
        self.register_dataloader()
        ##### `TrainArguments` adapt after register dataloader #####
        self.args.adapt(self.train_dataloader, eval_dataloader=self.eval_dataloader)

        self.register_model()

        if self.args.tuner:
            self.register_tuner()

        if self.args.amp:
            self.register_amp()

        if self.optimizer is None:
            self.register_optimizer()

        if self.lr_scheduler is None:
            self.register_scheduler()

        if self.distributed_state is not None:
            self.register_distributed()

        # enable gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.evaluate_loop is None:
            self.register_evaluate()

        if self.args.trace and (self.distributed_state is None or self.distributed_state.rank == 0):
            self.register_tracer()

    def register_dataloader(self):
        train_batch_size_per_device = self.args.train_batch_size_per_device
        eval_batch_size_per_device = self.args.eval_batch_size_per_device
        Loader = DataloaderRegistry
        optional_args = {}
        if self.distributed_state is not None:
            logger.info(
                "trainer `distributed_state` parameter is not None."
                "register distributed dataloader by `train/eval_dataloader` in `Bumblebee registers Dataloader Library`."
                "You can pass `Dataloader` with `DistributedSampler`, `Sampler` or `BatchSampler` because it will be converted uniformly."
            )
            Loader = DistributedDataLoaderRegistry  # dist enabled
            optional_args.update({
                "rank": self.distributed_state.rank,
                "world_size": self.distributed_state.world_size,
            })
        registry = Loader(self.train_dataloader, batch_size=train_batch_size_per_device, shuffle=True, **optional_args)
        self.train_dataloader = registry()
        if self.args.eval:
            registry = Loader(self.eval_dataloader, batch_size=eval_batch_size_per_device, shuffle=False, **optional_args)
            self.eval_dataloader = registry()


    def register_model(self):
        """ You can overwrite this method to wrap model, such as lora and so on """
        pass


    def register_tuner(self):
        logger.info(
            "trainer enable `tuner` to wrap model."
            f"tuner type is {self.args.tuner_type} in `Bumblebee registers Tuner Library`."
        )
        registry = TunerRegistry(tuner_type=self.args.tuner_type, tuner_args=self.args.tuner_args)
        self.model = registry(self.model)


    def register_amp(self):
        self.autocast = torch.autocast(device_type=self.args.device.type, dtype=self.args.mixed_precision)
        if self.args.mixed_precision is torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()

    def register_optimizer(self):
        """ register optimizer must be after registering model """
        logger.info(
            "trainer `optimizer` parameter is None."
            f"Default to use `TrainArguments' optim_type`: {self.args.optim_type} in `Bumblebee registers Optimizer Library`."
        )
        # Use bumblebee library optimizer
        registry = OptimizerRegistry(self.args.optim_type, self.args.optim_args)
        self.optimizer = registry(params=self.model.parameters())

    def register_scheduler(self):
        logger.info(
            "trainer `lr_scheduler` parameter is None."
            f"Default to use `TrainArguments' lr_scheduler_type`: {self.args.lr_scheduler_type} in `Bumblebee registers LRScheduler Library`."
        )
        # Use bumblebee library lr_scheduler
        registry = LRSchedulerRegistry(self.args.lr_scheduler_type, self.args.lr_scheduler_args)
        self.lr_scheduler = registry(optimizer=self.optimizer)

    def register_distributed(self):
        logger.info(
            "trainer trains model with distributed."
            f"Default to use `TrainArguments' distributed_state` in `Bumblebee registers Distributed Library`."
        )
        # Use bumblebee library distributed
        registry = DistributedRegistry(self.distributed_state)
        # default use fp32 to model
        self.model, self.optimizer, _ = registry(
            model=self.model,
            optimizer=self.optimizer,
            train_batch_size_per_device=self.args.train_batch_size_per_device,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            max_grad_norm=self.args.max_grad_norm,
            model_type=self.model_type,
        )

    def register_evaluate(self):
        self.evaluate_loop = self._evaluate_loop

    def register_tracer(self):
        registry = TracerRegistry(wandb_args=self.args.wandb_args)
        self.tracer = registry(config_args=self.args)


    def train(self):
        args = self.args
        model = self.model
        train_dataloader = self.train_dataloader
        eval_dataloader = self.eval_dataloader
        tracer = self.tracer
        # Train!
        logger.info("***** Train Running *****")
        logger.info(f"Trainable Parameters = {self.fetch_model_param(model, trainable=True):,}")
        logger.info(f"Batch Size Per Device = {args.train_batch_size_per_device:,}")
        logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(
            f"Total Batch Size (w. parallel, distributed & accumulation) = {args.train_batch_size:,} ({args.distributed_state.world_size} devices)")
        logger.info(f"Epochs = {args.num_epochs:,}")
        logger.info(f"Max Steps = {args.max_steps:,}")

        # wait for other ranks if enable dist
        self.wait()

        # enter train loop
        self._inner_train_loop(model, args, train_dataloader, eval_dataloader, tracer)

        self._trace_summary(tracer, train_state=self.state, eval_state=self.eval_state)

    def _inner_train_loop(
        self,
        model,
        args: "TrainArguments",
        train_dataloader: "torch.utils.data.DataLoader",
        eval_dataloader: "torch.utils.data.DataLoader",
        tracer
    ):
        # First Eval
        if args.eval and not args.eval_skip_first and args.eval_delay == 0:
            logger.info("***** First Eval Running *****")
            # first eval does not save model.
            self._evaluate()


        # total steps
        global_step = -1
        max_steps = args.max_steps
        grad_accum_steps = args.gradient_accumulation_steps
        num_epochs = args.num_epochs
        for epoch in range(num_epochs):
            logger.info(f"***** Epoch: {epoch + 1}/{num_epochs} *****")
            if self.state.end: break
            dataloader_epoch = train_dataloader
            steps_epoch = len(dataloader_epoch)
            loss_epoch = 0.0

            ##### progress bar #####
            pbar_total = steps_epoch // grad_accum_steps  # model update total steps
            pbar = tqdm(colour="blue", desc=f"Train Epoch: {epoch + 1}", total=pbar_total, dynamic_ncols=True)

            for _, batch in enumerate(dataloader_epoch):
                global_step += 1  # start with zero
                sync_step = (global_step + 1) % grad_accum_steps == 0 or (global_step + 1) == steps_epoch
                if global_step == max_steps:
                    self.state.max_steps_reached = True
                    break

                batch = self._to_device(batch, device=args.device)
                loss_step = self.train_step(model, inputs=batch)  # loss_step = loss.detach()

                loss_epoch = loss_epoch + loss_step.float()

                if self.enable_deepspeed:
                    model.step()
                    ##### pbar update #####
                    pbar.update(1)
                elif sync_step:
                    if self.scale_grad:
                        if self.clip_grad:
                            # unscale grad before clipping if using fp16 model weights
                            self.scaler.unscale_(self.optimizer)
                            # fsdp need `model.clip_grad_norm_(grad_threshold)`
                            # others need torch.nn.utils.clip_grad_norm_
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    ##### pbar update #####
                    pbar.update(1)
                else:
                    pass

                # update trainer_state every step by loss
                self.state.update_step(loss_step.float().item())
                # tracer log to wandb platform
                if tracer is not None:
                    self._trace_train_progress(tracer, loss_step.float(), epoch=epoch, step=global_step)
                # step+1: ensure current executed step starting with 1
                if args.log and (global_step+1) % args.log_intervals == 0:
                    # log trainer state
                    self._trainer_log(args.log_dir, self.state, rank=self.rank)
                ##### terminal pbar #####
                pbar.set_description(
                    f"Train Epoch: {epoch + 1}/{num_epochs}, "
                    f"Update Step: {pbar.n}/{pbar_total}, "
                    f"Global Step: {global_step}/{max_steps}(no accum gard {grad_accum_steps})"
                    f"(loss: {loss_step.float()})"
                )

                # whether to eval
                if args.eval and ((global_step+1) % args.eval_intervals == 0 and (global_step+1) > args.eval_delay or (global_step+1) == args.eval_delay):
                    # do eval
                    self._evaluate()
            pbar.close()

            # epoch

            # reduce loss_epoch across_devices if using distributed training and there are more than one device
            if self.enable_dist and self.args.n_gpu > 1:
                dist.all_reduce(loss_epoch, op=dist.ReduceOp.SUM)
                loss_epoch = loss_epoch / self.distributed_state.world_size
            effective_steps = (global_step - epoch * steps_epoch) if self.state.max_steps_reached else steps_epoch
            effective_steps = max(effective_steps, 0.001) # Avoid ZeroDivisionError
            loss_epoch = loss_epoch / effective_steps
            # update trainer_state every epoch by loss or others
            self.state.update_epoch(loss_epoch, epoch=epoch, lr=self.fetch_lr())

            # update lr every epoch manually
            self.lr_scheduler.step()


        # we need do the final evaluation
        if args.eval:
            logger.info("***** Final Eval Running *****")
            # final evaluation.
            self._evaluate()



    def train_step(self, model: nn.Module, inputs: dict) -> "torch.Tensor":
        model.train()
        with self.autocast:
            loss = self.compute_loss(model, inputs)
        del inputs

        ## whether empty gpu/tpu/.. cache

        ## apex only need scaler_loss.backward() but others need to `loss = loss / gradient_accumulation_steps` expect `deepspeed`
        ## current support apex, only support torch.cuda.amp or deepspeed.amp (need set)
        if self.enable_deepspeed:
            model.backward(loss)
        else:
            loss = loss / self.args.gradient_accumulation_steps
            if self.scale_grad:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss.detach()


    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


    def _evaluate(self):
        eval_loop = self.evaluate_loop
        output = eval_loop(self.model, self.eval_dataloader, eval_steps=self.args.eval_steps)
        # update `eval_state`
        self.eval_state.update(
            step=output.step,
            loss=output.loss,
            predictions=output.predictions,
            labels=output.labels,
            metrics=output.metrics
        )
        # tracer evaluation log to wandb platform
        if self.tracer is not None:
            self._trace_eval_progress(self.tracer, output.loss)
        # log evaler state
        if self.args.log:
            self._evaler_log(self.args.log_dir, self.eval_state, rank=self.rank)
        if self.args.save_checkpoint and self.eval_state.model_save:
            # save best model
            save_args = {"model": self.model}
            if not self.args.save_model_only:
                save_args.update({"optimizer": self.optimizer, "scheduler": self.lr_scheduler})
            self.save_checkpoint(**save_args)

        # wait for other ranks if enable dist
        self.wait()


    def _evaluate_loop(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> EvalOutput:
        rank, world_size = 0, 1
        enable_dist = dist.is_initialized()
        if enable_dist:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        device = model.device
        # default eval_steps(max_eval_steps)  is dataloader total length
        max_steps = kwargs.get("eval_steps", len(dataloader))
        model.eval()

        # prepare container to store
        predictions = []
        labels = []
        loss_steps = []
        # perplexity_steps = []
        # prepare container to store

        eval_loss = 0.0
        step = -1

        ##### progress bar #####
        pbar = tqdm(colour="green", desc=f"Eval Epoch(rank={rank})", total=len(dataloader), dynamic_ncols=True)
        ##### progress bar #####

        for _, batch in enumerate(dataloader):
            step += 1
            if step == max_steps: break

            for key in batch.keys():
                batch[key] = batch[key].to(device)

            with torch.no_grad():
                outputs = model(**batch)
                # We support dict ModelOutput type return with `loss` key.
                loss = outputs["loss"].detach()
                loss_steps.append(loss.float().item())
                # perplexity_steps.append(float(torch.exp(loss.float())))
                eval_loss += loss.float()

            # logits key may not return
            """
            logits_batch = outputs.get("logits", None)
            if logits_batch is not None:
                pred_batch = torch.argmax(logits_batch, dim=-1).detach().cpu().numpy()
                predictions.extend(pred_batch)
            # add labels
            labels_batch = batch["labels"].cpu().numpy()
            labels.extend(labels_batch)
            """

            # empty cuda cache in order to avoid `OOM`
            # del loss, # logits_batch
            # torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()

        if enable_dist and torch.cuda.device_count() > 1:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)


        effective_steps = max(step, 0.001)  # avoid ZeroDivide error
        eval_loss = eval_loss / effective_steps
        eval_loss = eval_loss / world_size

        metrics = {
            "step_loss": loss_steps,
            # "step_perplexity": perplexity_steps,
        }

        return EvalOutput(step=step, loss=eval_loss, predictions=predictions, labels=labels, metrics=metrics)


    def _to_device(self, batch: dict, device: torch.device) -> dict:
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        return batch


    def fetch_model_param(self, model, trainable: bool = False):
        # compute trainable or total model params
        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()
        return sum(numel(p) for p in model.parameters() if not trainable or p.requires_grad)


    def fetch_lr(self):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]

        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr


    def _trace_train_progress(self, tracer, loss, **kwargs):
        """
        Args:
            tracer: wandb.sdk.wandb_run.Run
            loss: float
            kwargs: dict including step, epoch.
        """
        stage = "train"
        logs = {"train/loss": loss, "train/perplexity": math.exp(loss)}
        for k, v in kwargs.items():
            logs[f"{stage}/{k}"] = v
        tracer.log(logs)

    def _trace_eval_progress(self, tracer, loss, **kwargs):
        """
        Args:
            tracer: wandb.sdk.wandb_run.Run
            loss: float
            kwargs: dict which is empty
        """
        stage = "eval"
        logs = {"eval/loss": loss, "eval/perplexity": math.exp(loss)}
        for k, v in kwargs.items():
            logs[f"{stage}/{k}"] = v
        tracer.log(logs)

    def _trace_summary(self, trace, train_state: TrainerState, eval_state: EvalerState, **kwargs):
        # metric
        metrics = {
            "average_train_loss": sum(train_state.epoch_loss) / len(train_state.epoch_loss),
            "average_eval_loss": sum(eval_state.epoch_loss) / len(eval_state.epoch_loss),
            "eval_best_loss": eval_state.best_loss,
        }

        for k, v in metrics.items():
            print(f"{k}:  {v}")
            trace.summary[k] = v


    def _trainer_log(self, output_dir, state: TrainerState, rank=0):
        """ log trainer state to disk """
        logs = {}
        step = state.step
        last_step = state.log_step  # start with 0
        train_step_loss = state.step_loss[last_step: step]
        lrs = state.epoch_lr
        train_epoch_loss = state.epoch_loss
        # set anchor trainer_state.log_step = step
        state.log_step = step
        logs.update({
            "step": [last_step, step],
            "loss": train_epoch_loss,
            "lr": lrs,
            "step_loss": train_step_loss,
            "information": (
                f"`step` indicate that the `step_loss` is from start step:{last_step} to end step:{step}."
                "`loss` is a list which includes every train epoch loss that have been executed."
                "`lr` is a list which includes every train epoch learning rate that have been executed."
                "`step_loss` is a list which includes every step loss that have been executed."
            )
        })
        output_file = os.path.join(output_dir, f"train_log_{rank}_{last_step}_{step}.json")
        mp.Process(target=write_json, args=(logs, output_file)).start()

    def _evaler_log(self, output_dir, state: EvalerState, rank=0):
        logs = {}
        step = state.step
        loss = state.loss
        epoch_loss = state.epoch_loss
        best_loss = state.best_loss
        logs.update({
            "step": [0, step],
            "loss": loss,
            "best_loss": best_loss,
            "past_loss": epoch_loss,
            "information": (
                "`step` is from 0 to eval stop step."
                "`loss` is the average among total sampler loss."
                "`best_loss` is the min loss among total evaluations."
                "`past_loss` is a list including every evaluation loss."
            )
        })
        logs.update(state.metrics)
        output_file = os.path.join(output_dir, f"eval_log_{rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        mp.Process(target=write_json, args=(logs, output_file)).start()


    def save_checkpoint(self, model, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,):
        """
        Snapshot model, optimizer, scheduler, trainer_state and son on
        You can overwrite this function
        """
        self.snapshot(model, optimizer=optimizer, scheduler=scheduler)


    @cached_property
    def model_type(self) -> "torch.dtype":
        return self.model.dtype

    @cached_property
    def clip_grad(self) -> bool:
        return self.args.max_grad_norm > 0.0

    @cached_property
    def scale_grad(self) -> bool:
        return self.model_type == torch.float16

    @cached_property
    def enable_deepspeed(self) -> bool:
        return self.args.distributed_type == DistributedType.DEEPSPEED

    @cached_property
    def enable_dist(self) -> bool:
        return self.args.distributed_type in (DistributedType.DEEPSPEED, ) or (dist.is_available() and dist.is_initialized())

    @property
    def rank(self) -> int:
        return self.distributed_state.rank if self.enable_dist else 0

    def wait(self):
        if self.distributed_state is not None:
            self.distributed_state.wait()





