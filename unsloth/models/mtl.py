from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
import functools
from math import sqrt as math_sqrt
import warnings, subprocess, re, inspect, psutil, os, math
from typing import Optional, Tuple, List, Union
from ..tokenizer_utils import *
__version__ = "2025.3.19"
from torch.nn.functional import scaled_dot_product_attention
from unsloth_zoo.utils import Version, _get_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import sdpa_attention_forward, flash_attention_forward
from peft import PeftConfig, PeftModel
from cut_cross_entropy import linear_cross_entropy

from transformers import __version__ as transformers_version
transformers_version = Version(transformers_version)
IS_ATTENTION_REFACTOR = transformers_version > Version("4.47.1")

import os, contextlib, sys
from huggingface_hub.utils import get_token

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from transformers.cache_utils import Cache
from transformers.utils import LossKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA   = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2  = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32 = transformers_version  > Version("4.45.0")
SUPPORTS_GRANITE = transformers_version >= Version("4.46.0")
from huggingface_hub import HfFileSystem
from .loader_utils import get_model_name
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__
torch_nn_functional_softmax = torch.nn.functional.softmax

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)

HAS_CUT_CROSS_ENTROPY = False

KV_CACHE_INCREMENT = 512 # KV Cache update size

LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

BlockDiagonalCausalMask = None

from unsloth_zoo.tokenizer_utils import (
    patch_tokenizer as _patch_tokenizer,
)

torch_square = torch.square
torch_mean   = torch.mean
torch_mv     = torch.mv
torch_matmul = torch.matmul
torch_mm     = torch.mm

def patch_tokenizer(model, tokenizer):
    model, tokenizer = _patch_tokenizer(model, tokenizer)
    if model is not None:
        model.config.update({"unsloth_version" : __version__})
    return model, tokenizer
pass

def fused_linear_cross_entropy(
    hidden_states      : torch.Tensor,
    lm_weight          : torch.Tensor,
    labels             : torch.Tensor,
    num_items_in_batch : int = None,
    ignore_index       : int = -100,
    reduction          : str = "mean",
    logit_softcapping  : float = 0,
    accuracy_threshold : str = "auto",
):
    # All Unsloth Zoo code licensed under LGPLv3
    reduction = "sum" if num_items_in_batch is not None else "mean"
    if logit_softcapping == 0: logit_softcapping = None
    loss = linear_cross_entropy(
        hidden_states.to(lm_weight.dtype),
        lm_weight,
        targets      = labels,
        ignore_index = ignore_index,
        softcap      = logit_softcapping,
        reduction    = reduction,
        shift        = True,
        filter_eps   = accuracy_threshold,
    )
    if num_items_in_batch is not None: loss = loss / num_items_in_batch
    return loss
pass

class FastMTLLlamaModel(AutoModelForCausalLM):
    @staticmethod
    def from_pretrained(
            model_name                 = "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length             = 4096,
            dtype                      = None,
            load_in_4bit               = False,
            token                      = None,
            device_map                 = "mps",
            rope_scaling               = None, # Qwen2 does not support RoPE scaling
            fix_tokenizer              = True,
            model_patcher              = None,
            tokenizer_name             = None,
            trust_remote_code          = False,
            revision                   = None,
            use_exact_model_name       = False,
            *args, **kwargs,
        ):

        if token is None: token = get_token()
        if model_patcher is None: model_patcher = FastMTLLlamaModel
        model_patcher.pre_patch()
        assert (dtype is None or dtype == torch.float16 or dtype == torch.bfloat16)

        # First check if it's a normal model via AutoConfig
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()

        autoconfig_error = None
        peft_error = None
        try:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_model = True
        except Exception as error:
            autoconfig_error = str(error)
            is_model = False
        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_peft = True
        except Exception as error:
            peft_error = str(error)
            is_peft = False
        pass

        both_exist = (is_model and is_peft) and not SUPPORTS_LLAMA32

        # New transformers need to check manually.
        if SUPPORTS_LLAMA32:
            # Check if folder exists locally
            if os.path.isdir(model_name):
                exist_adapter_config = os.path.exists(os.path.join(model_name, "adapter_config.json"))
                exist_config         = os.path.exists(os.path.join(model_name, "config.json"))
                both_exist = exist_adapter_config and exist_config
            else:
                # Because HfFileSystem assumes linux paths, we need to set the path with forward slashes, even on Windows.
                files = HfFileSystem(token = token).glob(f"{model_name}/*.json")
                files = (os.path.split(x)[-1] for x in files)
                if sum(x == "adapter_config.json" or x == "config.json" for x in files) >= 2:
                    both_exist = True
                pass
            pass
        pass

        # Error out if both LoRA and normal model config exists.
        if both_exist:
            raise RuntimeError(
                "Unsloth: Your repo has a LoRA adapter and a base model.\n"\
                "You have 2 files `config.json` and `adapter_config.json`.\n"\
                "We must only allow one config file.\n"\
                "Please separate the LoRA and base models to 2 repos."
            )
        elif not is_model and not is_peft:
            error = autoconfig_error or peft_error
            # Old transformers version
            if "rope_scaling" in error.lower() and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support new RoPE scaling methods.\n"\
                    f"This includes Llama 3.1. The minimum required version is 4.43.2\n"\
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            raise RuntimeError(combined_error)
        pass

        # Get base model for PEFT:
        if is_peft:
            # Check base model again for PEFT
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(model_name, load_in_4bit)
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
            )
        pass

        if not was_disabled: enable_progress_bars()

        model_max_seq_length = model_config.max_position_embeddings
        max_position_embeddings = max(max_seq_length, model_max_seq_length)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache    = False,
            torch_dtype  = torch.float16,  # torch.float32,
            load_in_8bit = False,
            load_in_4bit = False,
            **kwargs,
        )

        # # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        tokenizer = load_correct_tokenizer(
            tokenizer_name    = tokenizer_name,
            model_max_length  = max_position_embeddings,
            padding_side      = "right",
            token             = token,
            trust_remote_code = trust_remote_code,
            fix_tokenizer     = fix_tokenizer,
        )

        # For transformers > 4.47.1, we need to add rotary_emb to all attention layers
        if IS_ATTENTION_REFACTOR or hasattr(model.model, "rotary_emb"):
            rotary_emb = LlamaRotaryEmbedding(config=model_config)
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = rotary_emb
        pass

        return model, tokenizer
    pass
pass

def FastMTLQwen2Attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

def LLamaForCausalLM_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)

        loss = None
        if labels is not None:
            # loss = self.loss_function(
            #     logits=logits,
            #     labels=labels,
            #     vocab_size=self.config.vocab_size, **kwargs
            # )
            n_items = kwargs.get("num_items_in_batch", None) or kwargs.get("n_items", None)
            loss = fused_linear_cross_entropy(
                hidden_states      = hidden_states,
                lm_weight          = self.lm_head,
                labels             = labels,
                num_items_in_batch = n_items,
                logit_softcapping  = logit_softcapping,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastMTLQwen2Model(FastMTLLlamaModel):
    @staticmethod
    def pre_patch():
        Qwen2Attention      .forward = FastMTLQwen2Attention_forward
        Qwen2ForCausalLM    .forward = LLamaForCausalLM_forward
        import transformers.models.qwen2.modeling_qwen2
        transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding = LlamaRotaryEmbedding
    pass


    @staticmethod
    def from_pretrained(
        model_name        = "Qwen/Qwen2-7B",
        max_seq_length    = 4096,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None,
        fix_tokenizer     = True,
        model_patcher     = None,
        tokenizer_name    = None,
        trust_remote_code = False,
        **kwargs,
    ):
        return FastMTLLlamaModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastMTLQwen2Model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
    pass

