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
from peft import PeftConfig, PeftModel

import os, contextlib, sys
from huggingface_hub.utils import get_token
from transformers import __version__ as transformers_version
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
transformers_version = Version(transformers_version)
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

def patch_tokenizer(model, tokenizer):
    model, tokenizer = _patch_tokenizer(model, tokenizer)
    if model is not None:
        model.config.update({"unsloth_version" : __version__})
    return model, tokenizer
pass

torch_square = torch.square
torch_mean   = torch.mean
torch_mv     = torch.mv
torch_matmul = torch.matmul
torch_mm     = torch.mm

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def fast_rms_layernorm_inference(self, X, XX = None, XX2 = None, variance = None):
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim = True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out = XX2), -1, keepdim = True, out = variance)
    pass
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None: X = XX.to(old_dtype)
    else: X.copy_(XX)

    X *= self.weight
    return X
pass

torch_nn_functional_silu = torch.nn.functional.silu
def fast_swiglu_inference(self, X, temp_gate = None, temp_up = None):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    # mlp_size = self.config.intermediate_size
    # temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda:0")

    gate = fast_linear_forward(self.gate_proj, X, out = temp_gate)
    up   = fast_linear_forward(self.  up_proj, X, out = temp_up)
    gate = torch_nn_functional_silu(gate, inplace = True)
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])
    return down
pass

def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = getattr(proj, "base_layer", proj) # (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, getattr(W, "quant_state", None), None, None, None, base_layer.bias
    pass

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None: adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    return (
        W,
        getattr(W, "quant_state", None),
        proj.lora_A [adapter].weight,
        proj.lora_B [adapter].weight,
        proj.scaling[adapter],
        base_layer.bias,
    )
pass

def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass
    out = torch_matmul(X, W, out = out)
    if W_quant is not None: del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass

    return out.view(batch, seq_len, -1) if reshape else out
pass

def fast_linear_forward(proj, X, temp_lora = None, out = None):

    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1: return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch_matmul(X, W.t(), out = out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)
        out = torch_matmul(X, W, out = out)
    pass

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        pass

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch_mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch_mm(X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora)
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        pass
        out = out.view(bsz, 1, out_dim)
    pass

    if bias is not None: out += bias

    return out
pass

def patch_linear_scaling(
    model_name = "gemma2",
    rope_module = None,
    scaled_rope_module = None,
    attention_module = None,
):
    assert(rope_module is not None and scaled_rope_module is not None)
    assert(attention_module is not None)

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = \
        f"import torch.nn as nn\n"\
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"\
        f"from {model_filepath} import logger, "\
        f"{model_name.title()}Attention, {model_name.title()}Config"

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        # Most likely already patched!
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """
    fix_rope_function = fix_rope_function.format(
        rope_function        = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
    )
    rotary_emb = re.findall(
        r"self\.rotary\_emb \= .+?\)", function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0:
        return None, exec_code + "\n\n" + function

    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function
pass

class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1]//2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        # RH_Q *= sin
        # Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q
    pass

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1]//2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        # RH_dY *= sin
        # dY += RH_dY
        return dY, None, None, None
    pass
pass


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    return Q, K
pass

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaMTLAttention_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask:         Optional[BlockDiagonalCausalMask] = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   bool = False,
    use_cache:           bool = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)

        if position_ids is None:
            # Useful for LongRoPE
            cos, sin = rotary_emb.get_cached(kv_seq_len)
        else:
            cos, sin = rotary_emb(V, seq_len = kv_seq_len)

    # Q, K = (
    #     fast_rope_embedding(Q, K, cos, sin)
    #     if position_ids is None
    #     else inplace_rope_embedding(Q, K, cos, sin, position_ids)
    # )

    Q, K = inplace_rope_embedding(Q, K, cos, sin)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
    pass
    # Must be contiguous or else results are False!
    # https://github.com/pytorch/pytorch/issues/112577
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    # Needs (batch_size, n_heads, seq_len, head_dim)
    # is_casual and attention_mask must not be both set!
    A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
    # Go back to (batch_size, seq_len, n_heads, head_dim)
    A = A.transpose(1, 2).contiguous()

    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass

# Solves https://github.com/unslothai/unsloth/issues/168
# Static KV Cache was introduced in 4.38.0, causing training to be much slower.
# Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
# https://github.com/huggingface/transformers/pull/27931
# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = getattr(config, "head_dim", None)
            if dim is None: dim = int((config.hidden_size // config.num_attention_heads))
            device = "cuda"
            max_position_embeddings = config.max_position_embeddings
        pass

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=self.current_rope_size, device=device, dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype = x.dtype),
            self.sin_cached[:seq_len].to(dtype = x.dtype),
        )
    pass

    def get_cached(self, seq_len = None):
        return self.cos_cached, self.sin_cached
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        self._set_cos_sin_cache(self.current_rope_size, device = "cuda", dtype = x.dtype)
    pass
pass

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim = dim, max_position_embeddings = max_position_embeddings, base = base, device = device, config = config)
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.current_rope_size, device="cpu", dtype=torch.int64).float()
        t = t / self.scaling_factor

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
    pass
pass

def fast_rms_layernorm(layernorm, X : torch.Tensor, gemma : bool = False):
    W : torch.Tensor = layernorm.weight
    eps : float = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
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

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:       torch.Tensor,
    causal_mask          = None,
    attention_mask:      Optional[torch.Tensor] = None,
    position_ids:        Optional[torch.LongTensor] = None,
    past_key_value:      Optional[Tuple[torch.Tensor]] = None,
    output_attentions:   Optional[bool] = False,
    use_cache:           Optional[bool] = False,
    padding_mask:        Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if use_cache and hasattr(self, "_flag_for_generation"):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states       = hidden_states,
            causal_mask         = causal_mask,
            attention_mask      = attention_mask,
            position_ids        = position_ids,
            past_key_value      = past_key_value,
            output_attentions   = output_attentions,
            use_cache           = use_cache,
            padding_mask        = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass

def CausalLM_fast_forward(fast_forward_inference):
    def _CausalLM_fast_forward(
        self,
        input_ids: torch.LongTensor = None,
        causal_mask: Optional[BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = 0,
        logits_to_keep: Optional[int] = 0,
        *args, **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None:
            outputs = fast_forward_inference(
                self,
                input_ids,
                past_key_values,
                position_ids = position_ids,
                attention_mask = attention_mask,
            )
        else:
            causal_mask = None

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            self.model._has_no_labels = labels is None
            outputs = self.model(
                input_ids = input_ids,
                causal_mask = causal_mask,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
            )
        pass
        hidden_states = outputs[0]

        bsz, q_len, hd = hidden_states.shape
        lm_head = self.lm_head.weight
        lm_head_device = lm_head.device

        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
        logit_scaling     = getattr(self.config, "logit_scale", 0)
        dtype = lm_head.dtype
        num_logits_to_keep = max(num_logits_to_keep, logits_to_keep)

        # Move items to same device as lm_head
        hidden_states = hidden_states.to(lm_head_device)
        if labels is not None: labels = labels.to(lm_head_device)

        # Output last hidden states without logits if asked
        if self.training and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            if num_logits_to_keep != 0:
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]
            return CausalLMOutputWithPast(
                loss = None,
                logits = hidden_states,
                past_key_values = outputs.past_key_values,
                hidden_states = outputs.hidden_states,
                attentions=  outputs.attentions,
            )
        pass

        if bsz == 1 and q_len == 1:
            logits = torch.mv(lm_head, hidden_states.ravel().to(dtype))
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif num_logits_to_keep != 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :].to(dtype))
        else:
            RETURN_LOGITS = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            # < 1024 Normal Unsloth uses less VRAM!
            if bsz*q_len <= 1024: RETURN_LOGITS = True

            if not RETURN_LOGITS and HAS_CUT_CROSS_ENTROPY and labels is not None:

                n_items = kwargs.get("num_items_in_batch", None) or kwargs.get("n_items", None)
                loss = fused_linear_cross_entropy(
                    hidden_states      = hidden_states,
                    lm_weight          = lm_head,
                    labels             = labels,
                    num_items_in_batch = n_items,
                    logit_softcapping  = logit_softcapping,
                )
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output

                output = CausalLMOutputWithPast(
                    loss=loss,
                    logits=EMPTY_LOGITS,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
                return output
            pass
            logits = self.lm_head(hidden_states.to(dtype))
        pass

        logits = logits.to(_get_dtype(self.config.torch_dtype))
        loss = None
        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
        logit_scaling     = getattr(self.config, "logit_scale", 0)
        if self.config.model_type == "granite":
            # granite uses logit_scaling as key and they divide by the scale unlike cohere
            # notice that for granite, logits_scale is 16 and for cohere it is 0.125 (aka 1/8) in their respective configs
            # granite: https://github.com/huggingface/transformers/blob/4d1d0f29a493098e6bc6b904b82e29cb331827f5/src/transformers/models/granite/modeling_granite.py#L1103
            # cohere: https://github.com/huggingface/transformers/blob/4d1d0f29a493098e6bc6b904b82e29cb331827f5/src/transformers/models/cohere/modeling_cohere.py#L1176
            logit_scaling = 1 / getattr(self.config, "logits_scaling", 1)

        if labels is not None:
            shift_logits = logits
            # if not hasattr(self, "extra_ignored_labels"):
            #     # Fixes https://github.com/unslothai/unsloth/issues/10
            #     self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda:0")
            # pass
            shift_labels = torch.empty_like(labels)
            shift_labels[..., :-1] = labels[..., 1:]
            shift_labels[..., -1] = -100
            # shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
            # loss = fast_cross_entropy_loss(
            #     logits = shift_logits,
            #     labels = shift_labels,
            #     logit_softcapping = logit_softcapping,
            #     logit_scaling     = logit_scaling,
            #     n_items           = kwargs.get("num_items_in_batch", None) or kwargs.get("n_items", None),
            # )

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index = -100,
                reduction = "mean",
            )
        else:
            if logit_scaling != 0:
                if logits.requires_grad:
                    logits = logit_scaling * logits
                else:
                    logits *= logit_scaling
                pass
            pass
            if logit_softcapping != 0:
                if logits.requires_grad:
                    logits = (1.0 / logit_softcapping) * logits
                    logits = torch.tanh(logits)
                    logits = logit_softcapping * logits
                else:
                    logits *= (1.0 / logit_softcapping)
                    torch.tanh(logits, out = logits)
                    logits *= logit_softcapping
                pass
            pass
        pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions=  outputs.attentions,
        )
    pass
    return _CausalLM_fast_forward
pass

# Fix new HF's inference code
def _fast_prepare_inputs_for_generation(self, input_ids, **kwargs,):
    past_key_values = kwargs.get("past_key_values", None)
    if past_key_values is not None:
        # Check for uninitialized DynamicCache
        if len(past_key_values) == 0:
            past_key_values = None
            kwargs["past_key_values"] = None
        else:
            input_ids = input_ids[:,[-1]]
            kwargs["attention_mask"] = kwargs["attention_mask"][:,[-1]]
    if "cache_position" in kwargs:
        kwargs["position_ids"] = kwargs["cache_position"]
    return { "input_ids" : input_ids, **kwargs, }
pass

torch_matmul = torch.matmul
def LlamaAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L406
        Fast inference using KV cache.
        QK^T can be computed in 4 chunks

        [Q, q] @ [K, k].T where q, k are the new tokens.
        [QK^T, Qk^T]
        [qK^T, qk^T]

        Since the attention mask wipes Qk^T, we just get
        [QK^T,    0]
        [qK^T, qk^T]

        Since softmax is row-wise, we get
        softmax([QK^T,    0])
        softmax([qK^T, qk^T])

        We then multiply by   [V]
                              [v]
        softmax([QK^T,    0]) [softmax(QK^T)V] *
        softmax([qK^T, qk^T]) [softmax([qK^T, qk^T]) @ [V, v]]

        But notice * [softmax(QK^T)V] is just the last attention.
        We just need to compute the last final row.

        This means we can pass in a row of Q, but we need to
        remember K and V, which are called the KV cache.
    """
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    K1, V1 = past_key_value
    dtype = Xn.dtype

    n_heads    = self.config.num_attention_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim   = self.head_dim
    # assert(n_kv_heads * n_groups == n_heads)

    hidden_size = self.config.hidden_size
    attention_size = n_heads*head_dim
    seq_len = K1.shape[-2]
    kv_seq_len = seq_len + 1

    # Prefill phase
    # if not hasattr(self, "paged_attention"):
    device = hidden_states.device
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = device)
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = device)
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)

        # Mistral Nemo 12b has weird dimensions
        if attention_size != hidden_size:
            self.temp_O = torch.empty((1, bsz, hidden_size), dtype = dtype, device = device)
        else:
            self.temp_O = self.temp_QA[1][:,:,:hidden_size]
        pass

        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = device)
        self.scalar = 1.0 / math_sqrt(self.head_dim)
        self.half_head_dim = head_dim // 2
    elif kv_seq_len >= self.paged_attention.shape[0]:
        self.paged_attention.resize_((self.paged_attention.shape[0]+KV_CACHE_INCREMENT, 2, bsz, n_kv_heads, head_dim))
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1]+KV_CACHE_INCREMENT))
    pass

    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])
    Kn = fast_linear_forward(self.k_proj, Xn, out = self.temp_KV[0])
    Vn = fast_linear_forward(self.v_proj, Xn, out = self.temp_KV[1])
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)

    # cos, sin = self.rotary_emb(Vn, seq_len = kv_seq_len)
    # Qn, Kn = inplace_rope_embedding(Qn, Kn, cos, sin, position_ids)

    # Need to do it prior 2 steps before hitting full on short KV cache
    # or else error
    self.rotary_emb.extend_rope_embedding(Vn, seq_len + 2)
    cos, sin = self.rotary_emb.get_cached(kv_seq_len)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    h = self.half_head_dim

    RH_Q = self.RH_Q
    RH_Q[:,:,:,:h] = Qn[:,:,:,h:]
    RH_Q[:,:,:,h:] = Qn[:,:,:,:h]
    RH_Q[:,:,:,:h].neg_() # torch.neg(RH_Q[:,:,:,:h], out = RH_Q[:,:,:,:h])
    Qn *= cos
    Qn.addcmul_(RH_Q, sin)

    RH_K = RH_Q[:,:n_kv_heads,:,:] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda:0")
    RH_K[:,:,:,:h] = Kn[:,:,:,h:]
    RH_K[:,:,:,h:] = Kn[:,:,:,:h]
    RH_K[:,:,:,:h].neg_() #torch.neg(RH_K[:,:,:,:h], out = RH_K[:,:,:,:h])
    Kn *= cos
    Kn.addcmul_(RH_K, sin)

    # New KV cache
    # Kn = torch.cat([K1, Kn], dim = 2)
    # Vn = torch.cat([V1, Vn], dim = 2)
    self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)
    self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)
    Kn = self.paged_attention_K[:kv_seq_len].permute(1, 2, 0, 3)
    Vn = self.paged_attention_V[:kv_seq_len].permute(1, 2, 0, 3)

    # Handle sliding windows
    sliding_window = getattr(self.config, "sliding_window", None)
    if sliding_window is not None and kv_seq_len > sliding_window:
        # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L193
        slicing_tokens = 1 - sliding_window
        Knn = Kn[:, :, slicing_tokens:, :]#.contiguous()
        Vnn = Vn[:, :, slicing_tokens:, :]#.contiguous()
    else:
        Knn, Vnn = Kn, Vn
    pass

    # Grouped query attention
    _, _, cached_len, _ = Knn.shape
    if bsz == 1 or not SDPA_HAS_GQA and n_groups != 1:
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Knn = Knn.reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, head_dim)
    pass
    # else:
    #     Knn, Vnn = Knn, Vnn
    # pass

    # Attention
    if bsz == 1:
        Qn *= self.scalar # See https://github.com/ggerganov/llama.cpp/issues/7805#issuecomment-2153349963
        # It seems like doing (Q * scalar) @ K is better than (Q @ K) * scalar to stop overflows
        A = torch_matmul(Qn, Knn.transpose(2, 3), out = self.attention[:,:,:,:cached_len])
        # if attention_mask is not None: A += attention_mask # Must add attention_mask for batched
        A[:] = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
        A = torch_matmul(A, Vnn, out = Qn)
    else:
        if SDPA_HAS_GQA:
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False, enable_gqa = True)
        else:
            A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False)
    pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    input_ids = input_ids[:,:self.max_seq_length]
    bsz, q_len = input_ids.shape
    hd = self.config.hidden_size
    mlp_size = self.config.intermediate_size

    X = self.model.embed_tokens(input_ids)
    X = X.to(_get_dtype(self.config.torch_dtype))
    bsz, q_len, hd = X.shape
    assert(q_len == 1)
    # Get saved buffers to reduce memory movement
    residual = torch.empty((bsz, q_len, hd), dtype = torch.float32, device = "cuda:0")
    _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = "cuda:0")
    XX, XX2 = _XX[0], _XX[1]
    variance = torch.empty((bsz, q_len, 1), dtype = torch.float32, device = "cuda:0")
    temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda:0")
    temp_gate, temp_up = temp_mlp[0], temp_mlp[1]

    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            X,
            seq_len,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
    else:
        attention_mask = None
    pass

    next_decoder_cache = []

    for idx, decoder_layer in enumerate(self.model.layers):
        residual.copy_(X) # residual = X
        X = fast_rms_layernorm_inference(
            decoder_layer.input_layernorm,
            X,
            XX = XX,
            XX2 = XX2,
            variance = variance,
        )
        X, present_key_value = LlamaAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = X,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = attention_mask,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
        )
        X += residual

        residual.copy_(X) # residual = X
        X = fast_rms_layernorm_inference(
            decoder_layer.post_attention_layernorm,
            X,
            XX = XX,
            XX2 = XX2,
            variance = variance,
        )
        X = fast_swiglu_inference(
            decoder_layer.mlp,
            X,
            temp_gate = temp_gate,
            temp_up = temp_up,
        )
        X += residual

        next_decoder_cache.append(present_key_value)
    pass
    X = fast_rms_layernorm_inference(
        self.model.norm,
        X,
        XX = XX,
        XX2 = XX2,
        variance = variance,
    )

    return BaseModelOutputWithPast(
        last_hidden_state = X,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass

def fix_prepare_inputs_for_generation(module):
    # Fix prepare_inputs_for_generation
    if hasattr(module, "prepare_inputs_for_generation"):
        module.prepare_inputs_for_generation = _fast_prepare_inputs_for_generation
    pass
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

        model, tokenizer = patch_tokenizer(model, tokenizer)
        return model, tokenizer
    pass
pass

class FastMTLQwen2Model(FastMTLLlamaModel):
    @staticmethod
    def pre_patch():
        pass
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

