"""
2026.5.1
2026.5.2
4.57.6
0.15.2
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.grpo_trainer import (Any, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, Dataset, GRPOConfig, GRPOTrainer, GenerationConfig, IterableDataset, Optional, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, RepeatRandomSampler, RewardFunc, Sampler, SyncRefModelCallback, Trainer, TrainerCallback, Union, apply_chat_template, broadcast_object_list, create_reference_model, defaultdict, gather, gather_object, generate_model_card, get_comet_experiment_url, is_conversational, is_deepspeed_zero3_enabled, is_peft_model, is_wandb_available, maybe_apply_chat_template, nn, os, pad, prepare_deepspeed, selective_log_softmax, set_seed, textwrap, torch, transformers, unwrap_model_for_generation, version, warnings, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, Dataset, GRPOConfig, GenerationConfig, IterableDataset, Optional, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, RewardFunc, SyncRefModelCallback, Trainer, TrainerCallback, Union, create_reference_model, defaultdict, is_deepspeed_zero3_enabled, is_peft_model, os, pad, prepare_deepspeed, set_seed, torch, transformers, warnings, os, selective_log_softmax, torch, transformers, Any, Union, apply_chat_template, broadcast_object_list, gather, gather_object, is_conversational, maybe_apply_chat_template, nn, os, pad, torch, unwrap_model_for_generation, GRPOTrainer, Trainer, gather, os, pad, torch)


import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Get gradient checkpointing setting from training arguments
        use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {
            "epilogue_fusion"   : True,
            "max_autotune"      : False,
            "shape_padding"     : True,
            "trace.enabled"     : False,
            "triton.enable_persistent_tma_matmul": torch.cuda.get_device_capability()[0] >= 9,
            "cuda.cutlass_epilogue_fusion_enabled": torch.cuda.get_device_capability()[0] >= 9,
            "cuda.cutlass_tma_only": torch.cuda.get_device_capability()[0] >= 9,
            "cuda.compile_opt_level"              : "-O2",
            "cuda.enable_cuda_lto"                : True,
        }

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index, temperature: float = 1.0):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens
    """
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts

def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Aligns a log probability tensor with a given attention mask.
    """

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Create destination row indices
    # Shape: [batch_size, logprob_seq_len]
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # --- 4. Filter out-of-bounds indices and perform assignment ---
    # Create a mask to identify only the indices that are within the bounds
    # of the target tensor's sequence length.
    valid_mask = dest_indices < mask_seq_len

    # Use this mask to select only the valid row indices, column indices,
    # and the corresponding values from the logprob tensor.
    # This flattens the selected elements into 1D tensors.
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]

    # Place the valid values into their correct positions in the padded tensor
    # using a single, efficient advanced indexing operation.
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # For XPU: estimate free memory from total - reserved
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        return 4, final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value
def _unsloth_get_final_logit_softcapping(config):
    """Return final_logit_softcapping for a model config, falling back to the
    nested text sub-config for composite models. Handles both:
      - Gemma-4-style configs where the attribute lives on ``config.text_config``
      - T5Gemma-style composite configs where the text sub-config is only
        reachable via ``config.get_text_config()``
    Returns 0 if unset, matching the previous behaviour.
    """
    softcap = getattr(config, "final_logit_softcapping", None)
    if softcap is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            get_text_config = getattr(config, "get_text_config", None)
            if callable(get_text_config):
                try:
                    text_cfg = get_text_config()
                except (TypeError, ValueError):
                    text_cfg = None
        if text_cfg is not None and text_cfg is not config:
            softcap = getattr(text_cfg, "final_logit_softcapping", None)
    return 0 if softcap is None else softcap

def _unsloth_get_mm_token_id(processing_class, attr_name, token):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    token_id = getattr(processing_class, attr_name, None)
    if token_id is None:
        token_id = getattr(tokenizer, attr_name, None)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if token_id is None and convert_tokens_to_ids is not None:
        token_id = convert_tokens_to_ids(token)

    if type(token_id) is int and token_id >= 0:
        if token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    return None

def _unsloth_fix_mm_token_type_ids(
    processing_class, input_ids, mm_token_type_ids = None, completion_ids = None
):
    image_token_id = _unsloth_get_mm_token_id(
        processing_class, "image_token_id", "<|image_pad|>"
    )
    video_token_id = _unsloth_get_mm_token_id(
        processing_class, "video_token_id", "<|video_pad|>"
    )

    if image_token_id is not None or video_token_id is not None:
        rebuilt = input_ids.new_zeros(input_ids.shape)
        if image_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == image_token_id, 1)
        if video_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == video_token_id, 2)
        return rebuilt

    if (
        mm_token_type_ids is not None
        and completion_ids is not None
        and mm_token_type_ids.shape[0] == input_ids.shape[0]
        and mm_token_type_ids.shape[1] + completion_ids.shape[1] == input_ids.shape[1]
    ):
        return torch.cat(
            [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
            dim = 1,
        )
    return mm_token_type_ids

def grpo_compute_loss(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Set defaults for optional arguments
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    input_ids = input_ids.unsqueeze(-1)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            old_per_token_logps=old,
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None:
            #must filter out extra prompt tokens in begining after making input_ids left padded
            importance_sampling_ratio = torch.exp((old * mask) - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=vllm_importance_sampling_cap
            )
    pass

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    # loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL
    # Note that this is a low variance low bias estimator for the KL divergence as used in GRPO paper
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        # set kl_i to a tensor of zeros with the correct shape
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)
    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
        #breakpoint()
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        if get_sapo_token_loss is None:
            raise Exception(f"sapo is only available in TRL 0.26.0+")
        loss_i = torch.empty_like(coef_1)
        positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
        #since we have n_chunks some tensors may error if they dont have elements in them
        if coef_1[positive_advantages_mask].numel() != 0:
            loss_i[positive_advantages_mask] = get_sapo_token_loss(
                coef_1[positive_advantages_mask], sapo_temperature_pos
            )
        if coef_1[~positive_advantages_mask].numel() != 0:
            loss_i[~positive_advantages_mask] = get_sapo_token_loss(
                coef_1[~positive_advantages_mask], sapo_temperature_neg
            )
        loss_i = -loss_i * advantages
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if use_vllm and sampling_per_token_logps is not None:
        loss_i = loss_i * importance_sampling_ratio
        #delta for metric
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["cispo", "dapo"]:
        normalizer = num_items_in_batch/ num_processes
        loss = (loss_i * mask).sum() / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return completion_length, x.mean()
            else:
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward
                mean_kl = mean_kl_per_reward.mean()
                return completion_length, mean_kl
    completion_length, mean_kl = masked_batch_mean(kl_i)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask

class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under AGPL3
    @staticmethod
    def forward(ctx, _new_logps, _old_logps, _ref_logps, _sampling_per_token_logps, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, _mask  = grpo_compute_loss(
                ref_logps,
                new_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio, coef_1)
        pass

        device =_new_logps.device
        grad_inputs = torch.empty_like(_new_logps)
        accumulated_loss              = torch.zeros(1, device = device)[0]
        accumulated_completion_length = torch.zeros(1, device = device)[0]
        accumulated_mean_kl           = torch.zeros(1, device = device)[0]
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        accumulated_coef_1            = []

        def accumulate_chunk(
            new_logps_j,
            old_logps_j,
            ref_logps_j,
            sampling_per_token_logps_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio, chunk_coef_1)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_delta            .append(chunk_delta)
            accumulated_flat_is_ratio    .append(chunk_flat_is_ratio)
            accumulated_coef_1           .append(chunk_coef_1)
            grad_inputs_j[:] = chunk_grad_input
        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_logps  = torch.chunk(_new_logps, chunks = n_chunks, dim = 0)
        if _old_logps is not None:
            old_logps  = torch.chunk(_old_logps, chunks = n_chunks, dim = 0)
        else:
            old_logps = [None] * n_chunks
        if _ref_logps is not None:
            ref_logps  = torch.chunk(_ref_logps, chunks = n_chunks, dim = 0)
        else:
            ref_logps = [None] * n_chunks
        if _sampling_per_token_logps is not None:
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        # mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)

        for (grad_inputs_j, new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, ) in\
            zip(grad_inputs_chunks, new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages):

            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long

            # mark_dynamic(new_hidden_states_j)
            # mark_dynamic(ref_hidden_states_j)
            # if old_hidden_states_j is not None:
            #     mark_dynamic(old_hidden_states_j)
            # mark_dynamic(input_ids_j)
            # mark_dynamic(mask_j)
            accumulate_chunk(
                new_logps_j,
                old_logps_j,
                ref_logps_j,
                sampling_per_token_logps_j,
                input_ids_j,
                mask_j,
                advantages_j,
                scaling,
                grad_inputs_j,
            )
        pass

        grad_inputs                  .div_(n_chunks)
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)

        if _sampling_per_token_logps is not None:
            accumulated_delta = torch.cat(accumulated_delta, dim=0)
            accumulated_flat_is_ratio = torch.cat(accumulated_flat_is_ratio, dim=0)
        else:
            accumulated_delta = None
            accumulated_flat_is_ratio = None
        accumulated_coef_1  = torch.cat(accumulated_coef_1, dim=0)
        ctx.save_for_backward(grad_inputs)
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio,
            accumulated_coef_1
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio, dcoef_1):
        (grad_input,) = ctx.saved_tensors
        return (grad_input, None, None, None, None, None, None, None, None, None, None, None)
    pass

def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_logps,
    ref_logps,
    n_chunks = -1,
    **kwargs,
):
    # All Unsloth Zoo code licensed under AGPL3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get('pixel_values',None)
    image_grid_thw = kwargs.get('image_grid_thw',None)
    pixel_attention_mask = kwargs.get('pixel_attention_mask',None)
    image_sizes = kwargs.get('image_sizes',None)
    # Transformers 5.x requires token_type_ids/mm_token_type_ids for some vision models
    token_type_ids = kwargs.get('token_type_ids',None)
    mm_token_type_ids = kwargs.get('mm_token_type_ids',None)
    if mm_token_type_ids is not None or image_grid_thw is not None:
        mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
            trainer.processing_class, input_ids, mm_token_type_ids
        )
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None) if getattr(trainer, "vllm_importance_sampling_correction", False) else None
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    prev_max_left_pad    = kwargs.get("max_left_pad", 0) #Always get max_left_pad for when training LLMs, enabled by deafult.

    #Delete this from kwargs so less issues
    _ = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = trainer.vllm_importance_sampling_cap if sampling_per_token_logps is not None else None
    kwargs["get_sapo_token_loss"] = trainer.get_sapo_token_loss if hasattr(trainer, "get_sapo_token_loss") else None
    kwargs["sapo_temperature_pos"] = trainer.args.sapo_temperature_pos if hasattr(trainer.args, "sapo_temperature_pos") else None
    kwargs["sapo_temperature_neg"] = trainer.args.sapo_temperature_neg if hasattr(trainer.args, "sapo_temperature_neg") else None
    kwargs["get_off_policy_mask"] = trainer.get_off_policy_mask if hasattr(trainer, "get_off_policy_mask") else None
    kwargs["off_policy_mask_threshold"] = trainer.args.off_policy_mask_threshold  if hasattr(trainer.args, "off_policy_mask_threshold") else None
    kwargs["use_vllm"] = trainer.use_vllm
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    if not hasattr(trainer, '_autocast_dtype'):
        trainer._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight
    dtype_bytes = 16 if trainer._autocast_dtype in [torch.float16, torch.bfloat16] else 32

    total_rows = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_dim = lm_head.shape[1]
    vocab_dim = lm_head.shape[0]

    if trainer.args.unsloth_grpo_mini_batch is None:
        if not hasattr(trainer, "_has_autotuned"):
            trainer._has_autotuned = True
            B, multiplier = autotune_batch_and_chunks(
                total_rows, seq_len, hidden_dim, vocab_dim, dtype_bytes, trainer.args.unsloth_logit_chunk_multiplier
            )
            trainer.args.unsloth_grpo_mini_batch = max(1, total_rows//B)
            trainer.args.unsloth_logit_chunk_multiplier = multiplier
            B = trainer.args.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
        elif trainer._step % trainer.current_gradient_accumulation_steps == 0:
            B = trainer.args.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
            del trainer._has_autotuned
            del trainer.args.unsloth_grpo_mini_batch
            del trainer.args.unsloth_logit_chunk_multiplier
        else:
            B = trainer.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
    else:
        if trainer.args.unsloth_grpo_mini_batch > total_rows:
            B = total_rows
        else:
            B = trainer.args.unsloth_grpo_mini_batch

        if trainer.args.unsloth_logit_chunk_multiplier is None:
            multiplier = max(4, seq_len // 4096)
        else:
            multiplier = trainer.args.unsloth_logit_chunk_multiplier

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

        # Determine max_left_pad from precomputed logprobs shape for consistency
        if old_logps is not None:
            max_left_pad = old_logps.shape[1] - logits_to_keep
        elif ref_logps is not None:
            max_left_pad = ref_logps.shape[1] - logits_to_keep
        else:
            max_left_pad = torch.max(left_pad_tokens_per_prompt).item()

        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]

        completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)

        if trainer.use_vllm and sampling_per_token_logps is not None and getattr(trainer, "vllm_importance_sampling_correction", False):
            sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
        else:
            sampling_per_token_logps = None
        attention_mask =  input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else:
        completion_input_ids = input_ids[:, -logits_to_keep:]

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass

    all_logprobs_list = []

    attention_mask_chunks = torch.chunk(attention_mask, chunks=B, dim=0)
    completion_ids_chunks = torch.chunk(completion_input_ids, chunks=B, dim=0)

    def chunk_optional(tensor, chunks):
        if tensor is None:
            return [None] * chunks
        return torch.chunk(tensor, chunks=chunks, dim=0)

    import math
    total_samples = input_ids.shape[0]
    batch_size = math.ceil(total_samples / B)

    input_ids_chunks = []
    attention_mask_chunks = []
    pixel_values_chunks = []
    image_grid_thw_chunks = []
    pixel_attention_mask_chunks = []

    current_pixel_idx = 0
    #TRL 0.23.0 batching logic
    for start in range(0, total_samples, batch_size):
        end = start + batch_size

        input_ids_chunks.append(input_ids[start:end])
        attention_mask_chunks.append(attention_mask[start:end])

        if image_grid_thw is not None and pixel_values is not None:

            grid_slice = image_grid_thw[start:end]
            image_grid_thw_chunks.append(grid_slice)
            batch_pixel_count = grid_slice.prod(dim=-1).sum().item()

            start_pixel_idx = current_pixel_idx
            end_pixel_idx = current_pixel_idx + batch_pixel_count

            pixel_values_chunks.append(pixel_values[start_pixel_idx:end_pixel_idx])

            if pixel_attention_mask is not None:
                pixel_attention_mask_chunks.append(
                    pixel_attention_mask[start_pixel_idx:end_pixel_idx]
                )
            else:
                pixel_attention_mask_chunks.append(None)

            current_pixel_idx = end_pixel_idx

        else:
            pixel_values_chunks.append(None)
            image_grid_thw_chunks.append(None)
            pixel_attention_mask_chunks.append(None)

    if image_sizes is not None and not isinstance(image_sizes, torch.Tensor):
        image_sizes_chunks = [[size] for size in image_sizes]
    else:
        image_sizes_chunks = chunk_optional(image_sizes, B)

    # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models
    token_type_ids_chunks = chunk_optional(token_type_ids, B)
    mm_token_type_ids_chunks = chunk_optional(mm_token_type_ids, B)

    zipped_inputs = zip(
        input_ids_chunks,
        attention_mask_chunks,
        pixel_values_chunks,
        image_grid_thw_chunks,
        pixel_attention_mask_chunks,
        image_sizes_chunks,
        token_type_ids_chunks,
        mm_token_type_ids_chunks,
        completion_ids_chunks
    )

    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)

    def to_device(tensor, device, non_blocking=True):
        if tensor is None: return None
        return tensor.to(device, non_blocking=non_blocking)

    class Unsloth_Offloaded_Log_Softmax(torch.autograd.Function):
        """
        Manual Gradient Checkpointing/CPU Offloading for Log Softmax.
        """
        @staticmethod
        def forward(ctx, hidden_states, lm_head, index, chunks,
                    logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature):
            #Only the activations are needed so if we keep entire computational graph, keeps unnecessary memory on CPU so we detach it
            ctx.saved_hidden_states = hidden_states.detach().contiguous().to("cpu", non_blocking=True)
            ctx.device = hidden_states.device
            ctx.dtype = hidden_states.dtype

            ctx.lm_head = lm_head
            ctx.lm_head_requires_grad = lm_head.requires_grad
            ctx.index = index
            ctx.args = (chunks, logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature)

            with torch.no_grad():
                output = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, index, *ctx.args
                )

            return output

        @staticmethod
        def backward(ctx, grad_output):
            hidden_states = to_device(ctx.saved_hidden_states, ctx.device)
            hidden_states = hidden_states.to(ctx.dtype)
            hidden_states.requires_grad_(True)

            lm_head = ctx.lm_head
            # #Possibly redundant lines
            # if ctx.lm_head_requires_grad:
            #     hidden_states.requires_grad_(True)
            # else:
            #     lm_head = lm_head.detach()

            index = ctx.index

            with torch.enable_grad():
                output = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, index, *ctx.args
                )

            torch.autograd.backward(output, grad_output)

            return (
                hidden_states.grad,
                lm_head.grad if ctx.lm_head_requires_grad else None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                            logit_scale_multiply=0.0, logit_scale_divide=0.0,
                            logit_softcapping=0.0, temperature=1, batch_size=8):
        if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:
            #We save a gigabyte or speed with the normal path under these specific conditions
            return chunked_hidden_states_selective_log_softmax(
                hidden_states,
                lm_head,
                index,
                chunks,
                logit_scale_multiply,
                logit_scale_divide,
                logit_softcapping,
                temperature
            )
        else:
            return Unsloth_Offloaded_Log_Softmax.apply(
                hidden_states, lm_head, index, chunks,
                logit_scale_multiply, logit_scale_divide,
                logit_softcapping, temperature
            )
    for (
        input_ids_chunk,
        attention_mask_chunk,
        pixel_values_chunk,
        image_grid_thw_chunk,
        pixel_attention_mask_chunk,
        image_sizes_chunk,
        token_type_ids_chunk,
        mm_token_type_ids_chunk,
        completion_ids
    ) in zipped_inputs:
            _extra_vision_kwargs = {}
            if token_type_ids_chunk is not None:
                _extra_vision_kwargs["token_type_ids"] = token_type_ids_chunk
            if mm_token_type_ids_chunk is not None:
                _extra_vision_kwargs["mm_token_type_ids"] = mm_token_type_ids_chunk
            with autocaster:
                if pixel_values is None:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        pixel_values = pixel_values_chunk,
                        image_grid_thw = image_grid_thw_chunk,
                        pixel_attention_mask = pixel_attention_mask_chunk,
                        image_sizes = image_sizes_chunk,
                        **_extra_vision_kwargs,
                    ).logits

                    new_hidden_states_chunk = new_hidden_states_chunk[:, -(logits_to_keep + max_left_pad + 1): , :]
                    new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                    logprobs_chunk = efficient_log_softmax(
                        new_hidden_states_chunk,
                        lm_head,
                        completion_ids,
                        chunks=input_ids_chunk.shape[0]*multiplier,
                        logit_scale_multiply=logit_scale_multiply,
                        logit_scale_divide=logit_scale_divide,
                        logit_softcapping=logit_softcapping,
                        temperature=temperature,
                        batch_size = B
                    )
                else:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        pixel_values = pixel_values_chunk,
                        image_grid_thw = image_grid_thw_chunk,
                        pixel_attention_mask = pixel_attention_mask_chunk,
                        image_sizes = image_sizes_chunk,
                        logits_to_keep = logits_to_keep + 1,
                        **_extra_vision_kwargs,
                    ).logits

                    new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                    # Guard: check if model returned hidden states or logits
                    if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
                        logprobs_chunk = efficient_log_softmax(
                            new_hidden_states_chunk,
                            lm_head,
                            completion_ids,
                            chunks=input_ids_chunk.shape[0]*multiplier,
                            logit_scale_multiply=logit_scale_multiply,
                            logit_scale_divide=logit_scale_divide,
                            logit_softcapping=logit_softcapping,
                            temperature=temperature,
                            batch_size = B
                        )
                    else:
                        # Model returned logits directly - scaling/softcapping already applied by model forward
                        logprobs_chunk = chunked_selective_log_softmax(new_hidden_states_chunk, completion_ids, temperature)
                #This is needed to avoid race conditions with GPT OSS offload_embbed=True
                #However, it seems that this line does not slow down or disrupt models.
                device_synchronize()
            all_logprobs_list.append(logprobs_chunk)

    new_logprobs = torch.cat(all_logprobs_list, dim=0)

    with autocaster:
        loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1 = UnslothEfficientGRPO.apply(
            new_logprobs,
            old_logps,
            ref_logps,
            sampling_per_token_logps,
            lm_head,
            completion_input_ids,
            completion_mask,
            advantages,
            trainer.beta,
            trainer.accelerator.scaler,
            1,
            kwargs
        )

    # Must force not returning hidden states but logits otherwise gibberish
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, completion_mask
    # Old non efficient code path
    new_logits = torch.matmul(new_hidden_states, lm_head.t())
    new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    old_logits = torch.matmul(old_hidden_states, lm_head.t())
    old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    loss, completion_length, mean_kl = grpo_compute_loss(
        old_logits,
        new_logits,
        completion_input_ids,
        completion_mask,
        trainer.beta,
        advantages,
    )
    return loss, completion_length, mean_kl
    pass

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)
def grpo_compute_loss_slow(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Set defaults for optional arguments
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    input_ids = input_ids.unsqueeze(-1)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            old_per_token_logps=old,
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None:
            #must filter out extra prompt tokens in begining after making input_ids left padded
            importance_sampling_ratio = torch.exp((old * mask) - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=vllm_importance_sampling_cap
            )
    pass

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    # loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL
    # Note that this is a low variance low bias estimator for the KL divergence as used in GRPO paper
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        # set kl_i to a tensor of zeros with the correct shape
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)
    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
        #breakpoint()
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        if get_sapo_token_loss is None:
            raise Exception(f"sapo is only available in TRL 0.26.0+")
        loss_i = torch.empty_like(coef_1)
        positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
        #since we have n_chunks some tensors may error if they dont have elements in them
        if coef_1[positive_advantages_mask].numel() != 0:
            loss_i[positive_advantages_mask] = get_sapo_token_loss(
                coef_1[positive_advantages_mask], sapo_temperature_pos
            )
        if coef_1[~positive_advantages_mask].numel() != 0:
            loss_i[~positive_advantages_mask] = get_sapo_token_loss(
                coef_1[~positive_advantages_mask], sapo_temperature_neg
            )
        loss_i = -loss_i * advantages
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if use_vllm and sampling_per_token_logps is not None:
        loss_i = loss_i * importance_sampling_ratio
        #delta for metric
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["cispo", "dapo"]:
        normalizer = num_items_in_batch/ num_processes
        loss = (loss_i * mask).sum() / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return completion_length, x.mean()
            else:
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward
                mean_kl = mean_kl_per_reward.mean()
                return completion_length, mean_kl
    completion_length, mean_kl = masked_batch_mean(kl_i)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask

def grpo_update_SamplingParams(SamplingParams, generation_kwargs, vllm_sampling_params = None):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    # Filter generation_kwargs
    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        for key in good_sampling_params_keys:
            if hasattr(vllm_sampling_params, key):
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key is not None and (type(overwrited_key) in (list, tuple,) and len(overwrited_key) != 0):
                    generation_kwargs[key] = overwrited_key
    return generation_kwargs

def _get_inference_mode_context_manager(model: torch.nn.Module):
    """
    If the state dict was quantized using torchao, we will run into
    the following error when calling ops like aten.t() in inference mode.
    This is a bug in PyTorch that affects all tensor subclasses.

        Cannot set version_counter for inference tensor

    For now, we work around this issue by using `torch.no_grad()` in this case.
    See https://github.com/pytorch/pytorch/issues/164872 for more details.
    Otherwise, just return `torch.inference_mode()`.
    """
    torchao_config = getattr(model, "torchao_config", None)
    if torchao_config is not None and torchao_config.qat_scheme is None:
        return torch.no_grad()
    else:
        return torch.inference_mode()

def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams

    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
@dataclass
class UnslothGRPOConfig(GRPOConfig):
    """
    
Configuration class for the [`GRPOTrainer`].

Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
[`~transformers.TrainingArguments`] documentation.

Using [`~transformers.HfArgumentParser`] we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

Parameters:
    > Parameters that control the model and reference model

    model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
        Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
        argument of the [`GRPOTrainer`] is provided as a string.

    > Parameters that control the data preprocessing

    remove_unused_columns (`bool`, *optional*, defaults to `False`):
        Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
        requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
    max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
        Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
    num_generations (`int` or `None`, *optional*, defaults to `8`):
        Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
        must be divisible by this value.
    temperature (`float`, *optional*, defaults to `0.9`):
        Temperature for sampling. The higher the temperature, the more random the completions.
    max_completion_length (`int` or `None`, *optional*, defaults to `256`):
        Maximum length of the generated completion.
    ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
        This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
        improving generation speed. However, disabling this option allows training models that exceed the VRAM
        capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
        with vLLM generation.

    > Parameters that control generation acceleration powered by vLLM

    use_vllm (`bool`, *optional*, defaults to `False`):
        Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
        training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
    vllm_device (`str`, *optional*, defaults to `"auto"`):
        Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
        automatically select the next available GPU after the last one used for training. This assumes that
        training has not already occupied all available GPUs. If only one device is available, the device will be
        shared between both training and vLLM.
    vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
        Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
        device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
        improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
        during initialization.
    vllm_dtype (`str`, *optional*, defaults to `"auto"`):
        Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
        based on the model configuration. Find the supported values in the vLLM documentation.
    vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
        If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
        `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
        context size, which might be much larger than the KV cache, leading to inefficiencies.

    > Parameters that control the training

    learning_rate (`float`, *optional*, defaults to `1e-6`):
        Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
        [`~transformers.TrainingArguments`].
    beta (`float`, *optional*, defaults to `0.04`):
        KL coefficient.
    reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
        Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
        weighted equally with weight `1.0`.
    sync_ref_model (`bool`, *optional*, defaults to `False`):
        Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
        the `ref_model_mixup_alpha` parameter. This synchronization originites from the
        [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
    ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
        α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
        between the current policy and the previous reference policy during updates. The reference policy is
        updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
        must set `sync_ref_model=True`.
    ref_model_sync_steps (`int`, *optional*, defaults to `64`):
        τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
        frequently the current policy is synchronized with the reference policy. To use this parameter, you must
        set `sync_ref_model=True`.

    > Parameters that control the logging

    log_completions (`bool`, *optional*, defaults to `False`):
        Whether to log the completions during training.

    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {'help': 'Multiplier for chunked logit computations.'},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'},
    )
    
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        lr_scheduler_kwargs = None,
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = False,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = None,
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        parallelism_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = 'none',
        project = 'huggingface',
        trackio_space_id = 'trackio',
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        hub_revision = None,
        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = False,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        liger_kernel_config = None,
        eval_use_gather_object = False,
        average_tokens_across_devices = True,
        model_init_kwargs = None,
        max_prompt_length = 512,
        num_generations = 8,
        temperature = 0.9,
        max_completion_length = 256,
        ds3_gather_for_generation = True,
        use_vllm = False,
        vllm_device = 'auto',
        vllm_gpu_memory_utilization = 0.9,
        vllm_dtype = 'auto',
        vllm_max_model_len = None,
        beta = 0.001,
        reward_weights = None,
        sync_ref_model = False,
        ref_model_mixup_alpha = 0.9,
        ref_model_sync_steps = 64,
        log_completions = False,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        
        **kwargs,
    ):
        if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if num_train_epochs is None:
            num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        if (per_device_train_batch_size // num_generations) * num_generations != per_device_train_batch_size:
            print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\nWe will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))
            per_device_train_batch_size = num_generations
        
        if temperature <= 0:
            raise ValueError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')
        elif temperature >= 10:
            raise ValueError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')
        
        if use_vllm and (top_k is None or top_k == 0): top_k = -1
        div = per_device_train_batch_size // num_generations
        if div * num_generations != per_device_train_batch_size:
            print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\nWe will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))
            per_device_train_batch_size = num_generations
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            parallelism_config = parallelism_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            project = project,
            trackio_space_id = trackio_space_id,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            model_init_kwargs = model_init_kwargs,
            max_prompt_length = max_prompt_length,
            num_generations = num_generations,
            temperature = temperature,
            max_completion_length = max_completion_length,
            ds3_gather_for_generation = ds3_gather_for_generation,
            use_vllm = use_vllm,
            vllm_device = vllm_device,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_dtype = vllm_dtype,
            vllm_max_model_len = vllm_max_model_len,
            beta = beta,
            reward_weights = reward_weights,
            sync_ref_model = sync_ref_model,
            ref_model_mixup_alpha = ref_model_mixup_alpha,
            ref_model_sync_steps = ref_model_sync_steps,
            log_completions = log_completions,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        

pass

class _UnslothGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled [not supported]
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if False:
            model = model

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta

        # The trainer estimates the number of FLOPs [floating-point operations] using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            self.llm = model.vllm_engine; self._last_loaded_step = 0; self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    **getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {}),
                )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, compute_efficient = False
    ):
        if True:  # os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return None  # Unsloth efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = (
                torch.float16
                if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                else torch.bfloat16
            )
            if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
                self._autocast_dtype = torch.float16

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(device_type = DEVICE_TYPE, dtype = self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                logits_to_keep = logits_to_keep + 1,
            ).logits
            # logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            return logits
            # input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            # logits = logits[:, -logits_to_keep:]
            # return logits
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            # logits = logits / self.temperature
            # logps = selective_log_softmax(logits, input_ids)

            # row_indices, col_indices = torch.where(logps < -20)

            # # Method 1: Check if tensors have elements
            # if len(row_indices) > 0 and len(col_indices) > 0:
            #     breakpoint()  # Breakpoint triggered here
            #     print("Found high values!")
            # return  logps #  compute logprobs for the input tokens

    def _move_model_to_vllm(self, *args, **kwargs):
        return None

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        
        _chat_template_ = getattr(self.processing_class, "chat_template", None)
        if _chat_template_ is None: _chat_template_ = ""
        _supported_keys_ = set(("prompt", "chosen", "rejected", "completion", "messages", "label"))
        _batch_chat_kwargs_ = getattr(self, "_unsloth_batch_chat_kwargs", None)

        prompts_text = []
        for _idx_, _example_ in enumerate(inputs):
            _tokenizer_kwargs_ = {}
            if type(_example_) is not dict:
                _example_ = {"prompt": _example_}
            _left_keys_ = _example_.keys() - _supported_keys_
            for k in _left_keys_:
                if k in _chat_template_:
                    v = _example_[k]
                    if type(v) is str:
                        _tokenizer_kwargs_[k] = v
            if _batch_chat_kwargs_ is not None and _idx_ < len(_batch_chat_kwargs_):
                for _bk_, _bv_ in _batch_chat_kwargs_[_idx_].items():
                    if _bk_ not in _tokenizer_kwargs_:
                        _tokenizer_kwargs_[_bk_] = _bv_
            _x_ = maybe_apply_chat_template(_example_, self.processing_class, **_tokenizer_kwargs_)["prompt"]
            prompts_text.append(_x_)

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False, lora_request = self.model.load_lora('grpo_trainer_lora_model', load_tensors = True))
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode(), torch.amp.autocast(device_type = 'cuda', dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) if not torch.is_autocast_enabled('cuda') else nullcontext())if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode(), torch.amp.autocast(device_type = 'cuda', dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) if not torch.is_autocast_enabled('cuda') else nullcontext())if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(
        self, model, inputs, return_outputs = False, num_items_in_batch = None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        pixel_values, image_grid_thw = (
            inputs.get("pixel_values", None),
            inputs.get("image_grid_thw", None),
        )
        pixel_attention_mask, image_sizes = (
            inputs.get("pixel_attention_mask", None),
            inputs.get("image_sizes", None),
        )
        # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models
        token_type_ids = inputs.get("token_type_ids", None)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        current_gradient_accumulation_steps = self.current_gradient_accumulation_steps
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1)
        if mm_token_type_ids is not None or image_grid_thw is not None:
            mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                input_ids,
                mm_token_type_ids,
                completion_ids = completion_ids,
            )
        # attention_mask = None
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = (
            lambda model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size = None,
            compute_entropy = False,
            compute_efficient = False: self._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, compute_efficient
            )
            if hasattr(self, "_get_per_token_logps")
            else self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                compute_entropy,
                compute_efficient,
            )[0]
        )  # logps

        per_token_logps = get_logps_func(
            model, input_ids, attention_mask, logits_to_keep, compute_efficient = True
        )
        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        # if self.beta != 0.0:
        #     with torch.inference_mode(), model.disable_adapter():
        #         ref_per_token_logps = per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)
        # else:
        #     ref_per_token_logps = None
        ref_logps = inputs.get("ref_per_token_logps", None)
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        old_logps = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        # Get logit softcapping and logit scale
        logit_softcapping = _unsloth_get_final_logit_softcapping(model.config)  # Gemma
        logit_scale_multiply = getattr(model.config, "logit_scale", 0)  # Cohere
        if logit_scale_multiply is None:
            logit_scale_multiply = 0
        logit_scale_divide = getattr(model.config, "logits_scaling", 0)  # Granite
        if logit_scale_divide is None:
            logit_scale_divide = 0

        max_left_pad = inputs.get("max_left_pad", 0)
        if per_token_logps is not None:
            (
                loss,
                completion_length,
                mean_kl,
                delta,
                flat_is_ratio,
                coef_1,
                completion_mask,
            ) = grpo_compute_loss_slow(
                ref_logps,
                per_token_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                loss_type = self.args.loss_type,
                importance_sampling_level = self.importance_sampling_level,
                epsilon_low = self.epsilon_low,
                epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = self.args.delta,
                temperature = self.args.temperature,
                max_left_pad = max_left_pad,
                logit_softcapping = logit_softcapping,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
                num_items_in_batch = num_items_in_batch,
                current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                num_processes = num_processes,
            )
        else:
            if hasattr(self.args, "loss_type"):
                (
                    loss,
                    completion_length,
                    mean_kl,
                    delta,
                    flat_is_ratio,
                    coef_1,
                    completion_mask,
                ) = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    importance_sampling_level = self.importance_sampling_level,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = self.args.delta,
                    temperature = self.args.temperature,
                    max_left_pad = max_left_pad,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                    num_items_in_batch = num_items_in_batch,
                    current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                    num_processes = num_processes,
                    sampling_per_token_logps = sampling_per_token_logps,
                    token_type_ids = token_type_ids,
                    mm_token_type_ids = mm_token_type_ids,
                )
            else:
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
                loss, completion_length, mean_kl, coef_1, completion_mask = (
                    grpo_accumulated_loss(
                        trainer = self,
                        input_ids = _input_ids,
                        logits_to_keep = logits_to_keep,
                        completion_mask = completion_mask,
                        advantages = advantages,
                        old_logps = old_logps,
                        ref_logps = ref_logps,
                        n_chunks = self.args.unsloth_num_chunks,
                        temperature = self.args.temperature,
                        logit_softcapping = logit_softcapping,
                        logit_scale_multiply = logit_scale_multiply,
                        logit_scale_divide = logit_scale_divide,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                        mm_token_type_ids = mm_token_type_ids,
                    )
                )
        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        if (
            self.use_vllm
            and delta is not None
            and getattr(self, "vllm_importance_sampling_correction", False)
        ):
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                self.accelerator.gather(min_importance_sampling_ratio)
                .nan_to_num(nan = float("inf"))
                .min()
                .item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                self.accelerator.gather(max_importance_sampling_ratio)
                .nan_to_num(nan = float("-inf"))
                .max()
                .item()
            )

        completion_token_count = completion_mask.sum().clamp(min = 1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(
                gathered_low_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/low_min"].append(
                nanmin(gathered_low_clip).item()
            )
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(
                gathered_high_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/high_max"].append(
                nanmax(gathered_high_clip).item()
            )
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                gathered_clip_ratio.nanmean().item()
            )
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(
                gathered_cispo_clip_ratio.nanmean().item()
            )

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
class UnslothGRPOTrainer(_UnslothGRPOTrainer):
    """
    
Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

Example:

```python
from datasets import load_dataset
from trl import GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    train_dataset=dataset,
)

trainer.train()
```

Args:
    model (`Union[str, PreTrainedModel]`):
        Model to be trained. Can be either:

        - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
          a path to a *directory* containing model weights saved using
          [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
          loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
          in `args.model_init_kwargs`.
        - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
    reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
        Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
        functions with the prompts and completions and sum the rewards. Can be either:

        - A single reward function, such as:
            - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
            path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
            using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
            keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
            - A custom reward function: The function is provided with the prompts and the generated completions,
              plus any additional columns in the dataset. It should return a list of rewards. For more details, see
              [Using a custom reward function](#using-a-custom-reward-function).
        - A list of reward functions, where each item can independently be any of the above types. Mixing different
        types within the list (e.g., a string model ID and a custom reward function) is allowed.
    args ([`GRPOConfig`], *optional*, defaults to `None`):
        Configuration for this trainer. If `None`, a default configuration is used.
    train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
        Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
        ignored. The format of the samples can be either:

        - [Standard](dataset_formats#standard): Each sample contains plain text.
        - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
          and content).
    eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
        Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
    processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
        Processing class used to process the data. The padding side must be set to "left". If `None`, the
        processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
    reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
        Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

        - A single processing class: Used when `reward_funcs` contains only one reward function.
        - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
        If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
        `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
        For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
        the corresponding entries in `reward_processing_classes` are ignored.
    callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
        List of callbacks to customize the training loop. Will add those to the list of default callbacks
        detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

        If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
        method.
    optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
        A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
        model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
    peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
        PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

    """
    def __init__(
        self,
        model,
        reward_funcs,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_classes = None,
        callbacks = None,
        peft_config = None,
        **kwargs
    ):
        if args is None: args = UnslothGRPOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'
        if not full_finetuning and (os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'):
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().weight.dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            # Forced float32 training
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            # Mixed precision training
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'fp16' if float16 else 'bf16'
            # args.mixed_precision is a new argument which needs to be set now
        elif mixed_precision_dtype == 'bfloat16':
            # Both False since bfloat16 full finetuning doesn't do any autocasting.
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if model is not None:
            _warnings_issued = getattr(model, 'warnings_issued', None)
            if _warnings_issued is None:
                model.warnings_issued = {}
            elif not isinstance(_warnings_issued, dict):
                try:
                    model.warnings_issued = dict(_warnings_issued)
                except Exception:
                    model.warnings_issued = {}
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        other_metrics = []
        if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]
        else: _reward_funcs = reward_funcs
        for reward_func in _reward_funcs:
            try:
                reward_func_name = reward_func.__name__
                if False:
                    other_metrics.append(f'rewards/{reward_func_name}/mean')
                if False:
                    other_metrics.append(f'rewards/{reward_func_name}/std')
                if True:
                    other_metrics.append(f'rewards/{reward_func_name}')
            except: pass
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('grpo_trainer', other_metrics)
        
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals() and hasattr(model, "for_training"):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        super().__init__(
            model = model,
            reward_funcs = reward_funcs,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_classes = reward_processing_classes,
            callbacks = callbacks,
            peft_config = peft_config,**kwargs)
        if "model" in locals() and hasattr(model, "for_inference"):
            model.for_inference()
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        if hasattr(self, 'train'):
            self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)
        pass
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):
            _vllm_tok = self.llm.get_tokenizer()
            _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:
                _vllm_tok.chat_template = _pc.chat_template
        pass
        
pass
