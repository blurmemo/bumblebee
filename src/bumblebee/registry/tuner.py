import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Union, List
import logging
from torch import nn

from ..utils.typology import TunerType, LoRAType, DistributedType
from ..tuners.lora import Linear
from ..tuners.tuner_utils import recursive_setattr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class TunerArguments:
    fan_in_fan_out: bool = False


@dataclass
class LoRAArguments(TunerArguments):
    layer: str = "linear"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    merge_weights: bool = True


@dataclass
class LinearLoRAArguments(LoRAArguments):
    target_names: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])





class TunerRegistry:
    framework = "bumblebee"

    def __init__(self, tuner_type: Optional[Union[TunerType, str]] = None, tuner_args: Optional[Union[TunerArguments, dict]] = None):
        self.tuner_type = tuner_type
        self.tuner_args = tuner_args
        self.validate_init_params()

    def validate_init_params(self):
        if self.tuner_type is not None:
            self.tuner_type = TunerType(self.tuner_type)

        if self.tuner_args is None:
            logger.info("`tuner_args` is None. Default use `Tuner` init params.")
            self.tuner_args = {}


    def __call__(self, model):
        callback = self.MAPPING.get(self.tuner_type, None)

        return model if callback is None else callback(model, self.tuner_args)



    @cached_property
    def MAPPING(self):
        return {
            TunerType.LORA: self.lora
        }


    def lora(self, model, tuner_args):
        logger.warning("If you enable `deepspeed zero3`, set `LoRA.merge_weights=False` to avoid RuntimeError.")

        if isinstance(tuner_args, dict):
            layer_name = tuner_args.get("layer", None)
        elif isinstance(tuner_args, LoRAArguments):
            layer_name = tuner_args.layer
        else:
            raise TypeError

        if layer_name is None:
            logger.warning(f"`layer={layer_name}`, no suitable layer to use LoRA.")
            return model

        layer_name = LoRAType(layer_name)

        if layer_name == LoRAType.LINEAR:
            ARGS = LinearLoRAArguments

        if isinstance(tuner_args, dict):
            tuner_args = ARGS(**tuner_args)

        if not isinstance(tuner_args, ARGS):
            raise TypeError(
                f"`tuner_args type` is not satisfied with `layer={layer_name}`."
                f"Please use `dict` or `{ARGS.__name__}`"
            )

        for name, module in model.named_modules():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, nn.Linear) and any(re.search(tn, name) for tn in tuner_args.target_names):
                tmp = Linear(
                    module.in_features,
                    module.out_features,
                    r=tuner_args.r,
                    lora_alpha=tuner_args.lora_alpha,
                    lora_dropout=tuner_args.lora_dropout,
                    fan_in_fan_out=tuner_args.fan_in_fan_out,
                    merge_weights=tuner_args.merge_weights,
                ).to(module.weight.device).to(module.weight.dtype)
                tmp.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    tmp.bias.data.copy_(module.bias.data)
                recursive_setattr(model, name, tmp)


        # This enable input require grads function to make gradient checkpointing work for lora-only optimization for Huggingface
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model













