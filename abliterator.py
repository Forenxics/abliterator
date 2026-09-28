"""
Abliterator: A Python library for transformer activation steering and ablation.

This module provides tools for discovering and applying activation steering
interventions to Large Language Models (LLMs) to control behavior, particularly
for studying harmful/harmless outputs by calculating directions in activation space.
"""
# Deferred annotation evaluation (PEP 563): required so the `X | Y` union syntax
# used throughout this module's type hints doesn't crash at import time on
# Python < 3.10, where `|` is not yet supported between typing generics/types.
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
import functools
import einops
import gc
import re
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from datasets import load_dataset
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure module-level logger
logger = logging.getLogger(__name__)

# Type aliases for common types
TokenSet = Union[List[int], Tuple[int, ...], Set[int], Int[Tensor, '...']]
InstructionList = List[str]
DatasetSplit = Tuple[InstructionList, InstructionList]

def batch(iterable: Iterator, n: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from an iterable.

    Args:
        iterable: Any iterable to chunk.
        n: Size of each chunk.

    Yields:
        Lists of up to n items from the iterable.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def get_harmful_instructions() -> Tuple[List[str], List[str]]:
    """Load harmful instructions from the orthogonal activation steering dataset.

    Fetches the 'Undi95/orthogonal-activation-steering-TOXIC' dataset from
    HuggingFace and splits it into train/test sets.

    Returns:
        Tuple of (train_instructions, test_instructions).
    """
    hf_path = 'Undi95/orthogonal-activation-steering-TOXIC'
    dataset = load_dataset(hf_path)
    instructions = [i['goal'] for i in dataset['test']]

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions() -> Tuple[List[str], List[str]]:
    """Load harmless instructions from the Alpaca dataset.

    Fetches the 'tatsu-lab/alpaca' dataset from HuggingFace, filters for
    instructions without additional input, and splits into train/test sets.

    Returns:
        Tuple of (train_instructions, test_instructions).
    """
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    # Filter for instructions that do not have inputs
    instructions = [
        item['instruction']
        for item in dataset['train']
        if item['input'].strip() == ''
    ]

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def prepare_dataset(dataset: Tuple[List[str], List[str]] | List[str]) -> Tuple[List[str], List[str]]:
    """Prepare a dataset by splitting into train/test sets if not already split.

    Args:
        dataset: Either a tuple of (train_list, test_list) or a flat list of strings to be split.

    Returns:
        Tuple of (train_list, test_list)
    """
    # Check if dataset is already split into a (train, test) tuple. Per the
    # documented type contract, the flat/unsplit form is always a List, never
    # a tuple, so checking for a 2-tuple of lists is sufficient and (unlike a
    # non-empty/content check) doesn't misclassify a presplit dataset that
    # happens to have an empty train or test list.
    is_presplit = (
        isinstance(dataset, tuple) and
        len(dataset) == 2 and
        isinstance(dataset[0], list) and
        isinstance(dataset[1], list)
    )

    if is_presplit:
        train, test = dataset
    else:
        train, test = train_test_split(list(dataset), test_size=0.1, random_state=42)

    return train, test

def directional_hook(
    activation: Float[Tensor, "... d_model"],
    hook: HookPoint,
    direction: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    """Hook function for ablating activations along a specific direction.

    Projects out the component of the activation along the given direction.

    Args:
        activation: The activation tensor to modify.
        hook: The hook point (unused but required by TransformerLens).
        direction: The direction vector to project out.

    Returns:
        The modified activation with the directional component removed.
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)

    proj = einops.einsum(activation, direction.view(-1, 1), '... d_model, d_model single -> ... single') * direction
    return activation - proj


def clear_mem() -> None:
    """Clear GPU memory by running garbage collection and emptying CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def measure_fn(measure: str, input_tensor: Tensor, *args, **kwargs) -> Float[Tensor, '...']:
    """Apply a measure function to a tensor, consistently returning values only.

    Note: torch.max and torch.median return namedtuples (values, indices) when dim is specified.
    This function extracts just the values for consistency.
    """
    avail_measures = {
        'mean': torch.mean,
        'median': torch.median,
        'max': torch.max,
        'min': torch.min,
        'stack': torch.stack,
        'sum': torch.sum
    }

    if measure not in avail_measures:
        raise NotImplementedError(
            f"Unknown measure function '{measure}'. Available measures: {', '.join(avail_measures.keys())}"
        )

    result = avail_measures[measure](input_tensor, *args, **kwargs)

    # torch.max/min/median return a plain Tensor when no `dim` is given, but a
    # (values, indices) namedtuple-like when `dim` is given. `hasattr(result,
    # 'values')` is NOT a safe way to tell these apart: every torch.Tensor
    # also exposes a `.values` bound method (part of the sparse-tensor API),
    # so that check fires for plain tensors too and silently returns the
    # unbound method instead of the tensor. Checking the type directly avoids
    # that trap.
    if isinstance(result, Tensor):
        return result
    return result.values

class ChatTemplate:
    """A context manager for temporarily managing chat templates.

    Allows setting custom instruction formatting templates that can be
    used as context managers for temporary template changes.

    Attributes:
        model: The model this template is associated with.
        template: The template string with {instruction} placeholder.
    """

    def __init__(self, model: Any, template: str):
        """Initialize a chat template.

        Args:
            model: The ModelAbliterator instance.
            template: Template string with {instruction} placeholder.
        """
        self.model = model
        self.template = template
        self._prev: Optional['ChatTemplate'] = None

    def format(self, instruction: str) -> str:
        """Format an instruction using this template.

        Args:
            instruction: The instruction text to format.

        Returns:
            The formatted prompt string.
        """
        return self.template.format(instruction=instruction)

    def __enter__(self) -> 'ChatTemplate':
        """Enter context: save current template and set this one."""
        self._prev = self.model.chat_template
        self.model.chat_template = self
        return self

    def __exit__(self, exc: Any, exc_value: Any, exc_tb: Any) -> None:
        """Exit context: restore the previous template."""
        self.model.chat_template = self._prev
        self._prev = None


# Predefined chat templates for common models
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
PHI3_CHAT_TEMPLATE = """<|user|>\n{instruction}<|end|>\n<|assistant|>"""

class ModelAbliterator:
    """Main class for performing activation steering and ablation on LLMs.

    This class provides methods for:
    - Loading and managing transformer models via TransformerLens
    - Caching activations from harmful and harmless instruction sets
    - Computing refusal directions in activation space
    - Testing and applying ablation directions to model weights
    - Measuring the impact of interventions

    The class supports both temporary (hook-based) and permanent (weight-based)
    modifications, with context manager support for automatic state restoration.

    Attributes:
        model: The HookedTransformer model instance.
        harmful: Cached activations from harmful instructions.
        harmless: Cached activations from harmless instructions.
        modified_layers: Track of which layers have been modified.
        activation_layers: List of activation types to cache.

    Example:
        >>> from abliterator import ModelAbliterator, get_harmful_instructions, get_harmless_instructions
        >>> harmful = get_harmful_instructions()
        >>> harmless = get_harmless_instructions()
        >>> abliterator = ModelAbliterator("meta-llama/Meta-Llama-3-8B-Instruct", [harmful, harmless])
        >>> abliterator.cache_activations()
        >>> dirs = abliterator.refusal_dirs()
    """

    def __init__(
        self,
        model: str,
        dataset: Union[Tuple[DatasetSplit, DatasetSplit], List[DatasetSplit]],
        device: str = 'cuda',
        n_devices: Optional[int] = None,
        cache_fname: Optional[str] = None,
        activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out'],
        chat_template: Optional[str] = None,
        positive_toks: Optional[TokenSet] = None,
        negative_toks: Optional[TokenSet] = None,
        hf_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        """Initialize the ModelAbliterator.

        Args:
            model: HuggingFace model path or name, used to resolve which
                TransformerLens architecture/config to build (e.g. "gpt2",
                "meta-llama/Meta-Llama-3-8B-Instruct"). If `hf_model` is not
                given, weights are also downloaded from this name/path.
            dataset: Tuple of (harmful_dataset, harmless_dataset), where each dataset
                is either a pre-split (train, test) tuple or a list to be split.
            device: Device to load the model on ('cuda' or 'cpu').
            n_devices: Number of devices for model parallelism. Defaults to available GPUs.
            cache_fname: Path to a cached activations file to load.
            activation_layers: List of activation types to cache (e.g., 'resid_pre', 'mlp_out').
            chat_template: Custom chat template string with {instruction} placeholder.
            positive_toks: Token IDs indicating positive/compliant responses.
            negative_toks: Token IDs indicating negative/refusal responses.
            hf_model: Optional pre-loaded HuggingFace model instance to wrap
                instead of downloading `model` from the Hub. Use this for
                local/offline weights (e.g. `AutoModelForCausalLM.from_pretrained(
                "/path/to/local/model")`). `model` must still name the
                TransformerLens-recognized architecture the checkpoint is based on.
            tokenizer: Optional pre-loaded tokenizer to use instead of downloading
                one for `model`. Required alongside `hf_model` for fully offline use.
            dtype: Torch dtype to load the model in. Defaults to bfloat16;
                use float32 (or float16) on hardware/backends without good
                bfloat16 support, e.g. most CPUs.
        """
        self.MODEL_PATH = model
        if n_devices is None and torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
        elif n_devices is None:
            n_devices = 1

        # Save memory
        torch.set_grad_enabled(False)

        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            hf_model=hf_model,
            tokenizer=tokenizer,
            n_devices=n_devices,
            device=device,
            dtype=dtype,
            default_padding_side='left'
        )

        self.model.requires_grad_(False)

        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.chat_template = chat_template or ChatTemplate(self,LLAMA3_CHAT_TEMPLATE)

        self.hidden_size = self.model.cfg.d_model
        self.original_state = {k:v.to('cpu') for k,v in self.model.state_dict().items()}
        self.harmful = {}
        self.harmless = {}
        self.modified_layers = {'mlp':{}, 'W_O':{}}
        self.checkpoints = []

        if cache_fname is not None:
            # Note: weights_only=False is required for loading ActivationCache objects.
            # Only load cache files from trusted sources.
            try:
                outs = torch.load(cache_fname, map_location='cpu', weights_only=False)
                if not isinstance(outs, (list, tuple)) or len(outs) < 4:
                    raise ValueError(f"Invalid cache file format: expected list with at least 4 elements, got {type(outs)}")
                self.harmful, self.harmless, modified_layers, checkpoints = outs[:4]
                self.checkpoints = checkpoints or []
                self.modified_layers = modified_layers or {'mlp': {}, 'W_O': {}}
            except Exception as e:
                raise RuntimeError(f"Failed to load cache from '{cache_fname}': {e}") from e

        self.harmful_inst_train,self.harmful_inst_test = prepare_dataset(dataset[0])
        self.harmless_inst_train,self.harmless_inst_test = prepare_dataset(dataset[1])

        self.fwd_hooks = []
        self.modified = False
        self.activation_layers = [activation_layers] if isinstance(activation_layers, str) else activation_layers
        if negative_toks is None:
            logger.warning("'negative_toks' not set, defaulting to tokens for Llama-3 vocab")
            self.negative_toks = {4250, 14931, 89735, 20451, 11660, 11458, 956}  # llama-3 refusal tokens e.g. ' cannot', ' unethical', ' sorry'
        else:
            self.negative_toks = negative_toks
        if positive_toks is None:
            logger.warning("'positive_toks' not set, defaulting to tokens for Llama-3 vocab")
            self.positive_toks = {32, 1271, 8586, 96556, 78145}
        else:
            self.positive_toks = positive_toks
        self._blacklisted = set()

    def __enter__(self) -> 'ModelAbliterator':
        """Enter context: save current model state for restoration on exit."""
        if hasattr(self, "current_state"):
            raise RuntimeError("Cannot nest context managers (multi-contexting not supported)")
        self.current_state = self.model.state_dict()
        self.current_layers = self.modified_layers.copy()
        self.was_modified = self.modified
        return self

    def __exit__(self, exc: Any, exc_value: Any, exc_tb: Any) -> None:
        """Exit context: restore model to the state before entering."""
        self.model.load_state_dict(self.current_state)
        del self.current_state
        self.modified_layers = self.current_layers
        del self.current_layers
        self.modified = self.was_modified
        del self.was_modified

    def reset_state(self) -> None:
        """Reset the model to its original unmodified state."""
        self.modified = False
        self.modified_layers = {'mlp': {}, 'W_O': {}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self) -> None:
        """Save the current modification state to the checkpoint list."""
        # TODO: Consider offloading to disk to save RAM
        self.checkpoints.append(self.modified_layers.copy())

    # Utility functions

    def blacklist_layer(self, layer: int | List[int]):
        """Prevents a layer from being modified."""
        if isinstance(layer, (list, tuple)):
            for lyr in layer:
                self._blacklisted.add(lyr)
        else:
            self._blacklisted.add(layer)

    def whitelist_layer(self, layer: int | List[int]):
        """Removes layer from blacklist to allow modification."""
        if isinstance(layer, (list, tuple)):
            for lyr in layer:
                self._blacklisted.discard(lyr)
        else:
            self._blacklisted.discard(layer)

    def save_activations(self, fname: str):
        torch.save([self.harmful,self.harmless,self.modified_layers if self.modified_layers['mlp'] or self.modified_layers['W_O'] else None, self.checkpoints if len(self.checkpoints) > 0 else None], fname)

    def get_whitelisted_layers(self) -> List[int]:
        return [l for l in range(self.model.cfg.n_layers) if l not in self._blacklisted]

    def get_all_act_names(self, activation_layers: List[str] = None) -> List[Tuple[int,str]]:
        return [(i,utils.get_act_name(act_name,i)) for i in self.get_whitelisted_layers() for act_name in (activation_layers or self.activation_layers)]

    def calculate_mean_dirs(self, key: str, include_overall_mean: bool = False) -> Dict[str, Float[Tensor, 'd_model']]:
        """Calculate mean activation directions for harmful and harmless sets.

        Args:
            key: The activation name key in the cache.
            include_overall_mean: If True, also compute the overall mean direction.

        Returns:
            Dict with 'harmful_mean', 'harmless_mean', and optionally 'mean_dir'.
        """
        dirs = {
            'harmful_mean': torch.mean(self.harmful[key], dim=0),
            'harmless_mean': torch.mean(self.harmless[key], dim=0)
        }

        if include_overall_mean:
            if self.harmful[key].shape != self.harmless[key].shape or self.harmful[key].device.type == 'cuda':
                # If the shapes are different, we can't add them together; we'll need to concatenate the tensors first.
                # Using 'cpu', this is slower than the alternative below.
                # Using 'cuda', this seems to be faster than the alternatives.
                # NOTE: Assume both tensors are on the same device.
                #
                dirs['mean_dir'] = torch.mean(torch.cat((self.harmful[key], self.harmless[key]), dim=0), dim=0)
            else:
                # If the shapes are the same, we can add them together, take the mean,
                # then divide by 2.0 to account for the initial element-wise addition of the tensors.
                #
                # The result is identical to:
                #    `torch.sum(self.harmful[key] + self.harmless[key]) / (len(self.harmful[key]) + len(self.harmless[key]))`
                #
                dirs['mean_dir'] =  torch.mean(self.harmful[key] + self.harmless[key], dim=0) / 2.0

        return dirs

    def get_avg_projections(self, key: str, direction: Float[Tensor, 'd_model']) -> Tuple[Float[Tensor, 'd_model'], Float[Tensor, 'd_model']]:
        """Get average projections of harmful and harmless means onto a direction.

        Args:
            key: The activation name key in the cache.
            direction: The direction vector to project onto.

        Returns:
            Tuple of (harmful_projection, harmless_projection).
        """
        dirs = self.calculate_mean_dirs(key)
        return (torch.dot(dirs['harmful_mean'], direction), torch.dot(dirs['harmless_mean'], direction))

    def get_layer_dirs(self, layer: int, key: Optional[str] = None, include_overall_mean: bool = False) -> Dict[str, Float[Tensor, 'd_model']]:
        """Get mean directions for a specific layer.

        Args:
            layer: Layer index.
            key: Activation type key. Defaults to first activation layer.
            include_overall_mean: If True, also compute the overall mean direction.

        Returns:
            Dict with mean directions.

        Raises:
            IndexError: If layer is out of range.
            KeyError: If activation not found in cache.
        """
        act_key = key or self.activation_layers[0]
        if layer < 0 or layer >= self.model.cfg.n_layers:
            raise IndexError(f"Invalid layer {layer}. Must be in range [0, {self.model.cfg.n_layers})")
        act_name = utils.get_act_name(act_key, layer)
        if act_name not in self.harmful:
            raise KeyError(f"Activation '{act_name}' not found in cache. Run cache_activations first.")
        return self.calculate_mean_dirs(act_name, include_overall_mean=include_overall_mean)

    def refusal_dirs(self, invert: bool = False) -> Dict[str, Float[Tensor, 'd_model']]:
        """Compute normalized refusal directions for all cached activations.

        The refusal direction is the difference between harmful and harmless mean
        activations, normalized to unit length.

        Args:
            invert: If True, compute harmless - harmful instead of harmful - harmless.

        Returns:
            Dict mapping activation names to normalized refusal direction tensors.

        Raises:
            IndexError: If no activations have been cached.
        """
        if not self.harmful:
            raise IndexError("No cache. Run cache_activations first.")

        # Don't include layer 0 as it often becomes NaN
        refusal_dirs = {key: self.calculate_mean_dirs(key) for key in self.harmful if '.0.' not in key}
        if invert:
            refusal_dirs = {key: v['harmless_mean'] - v['harmful_mean'] for key, v in refusal_dirs.items()}
        else:
            refusal_dirs = {key: v['harmful_mean'] - v['harmless_mean'] for key, v in refusal_dirs.items()}

        return {key: (v / v.norm()).to('cpu') for key, v in refusal_dirs.items()}

    def scored_dirs(self,invert = False) -> List[Tuple[str,Float[Tensor, 'd_model']]]:
        refusals = self.refusal_dirs(invert=invert)
        return sorted([(ln,refusals[act_name]) for ln,act_name in self.get_all_act_names()],reverse=True, key=lambda x:abs(x[1].mean()))

    def get_layer_of_act_name(self, ref: str) -> str|int:
        s = re.search(r"\.(\d+)\.",ref)
        return s if s is None else int(s[1])

    def layer_attn(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            # make sure device doesn't change
            self.modified = True
            self.model.blocks[layer].attn.W_O.data = replacement.to(self.model.blocks[layer].attn.W_O.device)
            self.modified_layers['W_O'][layer] = self.modified_layers.get(layer,[])+[(self.model.blocks[layer].attn.W_O.data.to('cpu'),replacement.to('cpu'))]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            # make sure device doesn't change
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(self.model.blocks[layer].mlp.W_out.device)
            self.modified_layers['mlp'][layer] = self.modified_layers.get(layer,[])+[(self.model.blocks[layer].mlp.W_out.data.to('cpu'),replacement.to('cpu'))]
        return self.model.blocks[layer].mlp.W_out.data

    def tokenize_instructions_fn(
        self,
        instructions: List[str]
    ) -> Int[Tensor, 'batch_size seq_len']:
        prompts = [self.chat_template.format(instruction=instruction) for instruction in instructions]
        return self.model.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def generate_logits(
        self,
        toks: Int[Tensor, 'batch_size seq_len'],
        *args,
        drop_refusals: bool = True,
        stop_at_eos: bool = False,
        max_tokens_generated: int = 1,
        **kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Int[Tensor, 'batch_size seq_len']]:
        # does most of the model magic
        #
        # NOTE: the model is always run on the full batch each step. `generating`
        # only tracks which rows are still active (haven't hit EOS yet); it
        # gates which rows get new tokens written and which count toward the
        # refusal check. This keeps the returned `logits` batch-aligned with
        # `toks` — shrinking the forward pass itself would make `logits` cover
        # fewer/reordered rows once any sequence finishes early, silently
        # breaking any caller that indexes it by original batch position.
        batch_size, prompt_len = toks.shape
        all_toks = torch.zeros((batch_size, prompt_len + max_tokens_generated), dtype=torch.long, device=toks.device)
        all_toks[:, :prompt_len] = toks
        generating = set(range(batch_size))
        for i in range(max_tokens_generated):
            write_pos = prompt_len + i
            logits = self.model(all_toks[:, :write_pos], *args, **kwargs)
            next_tokens = logits[:, -1, :].argmax(dim=-1).to('cpu')
            active_idx = sorted(generating)
            all_toks[active_idx, write_pos] = next_tokens[active_idx]
            if drop_refusals and any(negative_tok in next_tokens[active_idx] for negative_tok in self.negative_toks):
                # refusals we handle differently: if it's misbehaving, we stop all batches and move on to the next one
                break
            if stop_at_eos:
                generating = {idx for idx in generating if all_toks[idx, write_pos] != self.model.tokenizer.eos_token_id}
                if not generating:
                    break
        return logits, all_toks

    def generate(
        self,
        prompt: List[str]|str,
        *model_args,
        max_tokens_generated: int = 64,
        stop_at_eos: bool = True,
        **model_kwargs
    ) -> List[str]:
        # convenience function to test manual prompts, no caching
        if isinstance(prompt, str):
            gen = self.tokenize_instructions_fn([prompt])
        else:
            gen = self.tokenize_instructions_fn(prompt)

        logits,all_toks = self.generate_logits(gen, *model_args, stop_at_eos=stop_at_eos, max_tokens_generated=max_tokens_generated, **model_kwargs)
        return self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)

    def test(
        self,
        *args,
        test_set: List[str] = None,
        N: int = 16,
        batch_size: int = 4,
        **kwargs
    ):
        if test_set is None:
            test_set = self.harmful_inst_test
        for prompts in batch(test_set[:min(len(test_set), N)], batch_size):
            for res in self.generate(prompts, *args, **kwargs):
                print(res)  # Intentional user-facing output for test results

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] = None,
        incl_bwd: bool = False,
        device: str = None,
        remove_batch_dim: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        fwd_hooks: List[str] = [],
        max_new_tokens: int = 1,
        **model_kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Dict[str, Float[Tensor, 'batch_size seq_len d_model']]]:
        if names_filter is None and self.activation_layers:
            def activation_layering(namefunc: str):
                return any(s in namefunc for s in self.activation_layers)
            names_filter = activation_layering


        cache_dict, fwd, bwd = self.model.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=utils.Slice(None)
        )

        fwd_hooks = fwd_hooks+fwd+self.fwd_hooks

        if not max_new_tokens:
            # must do at least 1 token
            max_new_tokens = 1

        with self.model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd, reset_hooks_end=reset_hooks_end, clear_contexts=clear_contexts):
            #model_out = self.model(*model_args,**model_kwargs)
            model_out,toks = self.generate_logits(*model_args,max_tokens_generated=max_new_tokens, **model_kwargs)
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def apply_refusal_dirs(
        self,
        refusal_dirs: List[Float[Tensor, 'd_model']],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[int] = None
    ):
        """Apply refusal directions to model weights.

        Args:
            refusal_dirs: List of direction tensors to ablate from weights.
            W_O: Whether to modify attention output weights.
            mlp: Whether to modify MLP output weights.
            layers: List of layer indices to modify. Defaults to all layers except layer 0.
        """
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for refusal_dir in refusal_dirs:
            for layer in layers:
                for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                    if modifying[0]:
                        matrix = modifying[1](layer)
                        if refusal_dir.device != matrix.device:
                            refusal_dir = refusal_dir.to(matrix.device)
                        proj = einops.einsum(matrix, refusal_dir.view(-1, 1), '... d_model, d_model single -> ... single') * refusal_dir
                        modifying[1](layer, matrix - proj)

    def induce_refusal_dir(
        self,
        refusal_dir: Float[Tensor, 'd_model'],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[int] = None
    ):
        """Induce a refusal direction into model weights.

        Note: This method is incomplete and needs further work.

        Args:
            refusal_dir: Direction tensor to induce.
            W_O: Whether to modify attention output weights.
            mlp: Whether to modify MLP output weights.
            layers: List of layer indices to modify. Defaults to all layers except layer 0.
        """
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for layer in layers:
            for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                if modifying[0]:
                    matrix = modifying[1](layer)
                    if refusal_dir.device != matrix.device:
                        refusal_dir = refusal_dir.to(matrix.device)
                    proj = einops.einsum(matrix, refusal_dir.view(-1, 1), '... d_model, d_model single -> ... single') * refusal_dir
                    avg_proj = refusal_dir * self.get_avg_projections(utils.get_act_name(self.activation_layers[0], layer), refusal_dir)
                    modifying[1](layer, (matrix - proj) + avg_proj)

    def test_dir(
        self,
        refusal_dir: Float[Tensor, 'd_model'],
        activation_layers: List[str] = None,
        use_hooks: bool = True,
        layers: List[str] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, 'd_model']]:
        # `use_hooks=True` is better for bigger models as it causes a lot of memory swapping otherwise, but
        # `use_hooks=False` is much more representative of the final weights manipulation

        before_hooks = self.fwd_hooks
        try:
            if layers is None:
                layers = self.get_whitelisted_layers()

            if activation_layers is None:
                activation_layers = self.activation_layers

            if use_hooks:
                hook_fn = functools.partial(directional_hook,direction=refusal_dir)
                self.fwd_hooks = before_hooks+[
                    (act_name,hook_fn)
                    for ln,act_name in self.get_all_act_names(activation_layers)
                    if ln in layers
                ]
                return self.measure_scores(**kwargs)
            else:
                with self:
                    self.apply_refusal_dirs([refusal_dir],layers=layers)
                    return self.measure_scores(**kwargs)
        finally:
            self.fwd_hooks = before_hooks

    def find_best_refusal_dir(
        self,
        N: int = 4,
        positive: bool = False,
        use_hooks: bool = True,
        invert: bool = False
    ) -> List[Tuple[float,str]]:
        dirs = self.refusal_dirs(invert=invert)
        if self.modified:
            logger.warning("Model is modified; will restore to current modified state each run")
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1],N=N,use_hooks=use_hooks)[int(positive)]
            scores.append((score,direction))
        return sorted(scores,key=lambda x:x[0])

    def measure_scores(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = 'max',
        batch_measure: str = 'max',
        positive: bool = False
    ) -> Dict[str, Float[Tensor, 'd_model']]:
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_test[:N])
        logits,cache = self.run_with_cache(toks,max_new_tokens=sampled_token_ct,drop_refusals=False)

        negative_score,positive_score = self.measure_scores_from_logits(logits,sampled_token_ct,measure=batch_measure)

        negative_score = measure_fn(measure,negative_score)
        positive_score = measure_fn(measure,positive_score)
        return {'negative':negative_score.to('cpu'), 'positive':positive_score.to('cpu')}

    def measure_scores_from_logits(
        self,
        logits: Float[Tensor, 'batch_size seq_len d_vocab'],
        sequence: int,
        measure: str = 'max'
    ) -> Tuple[Float[Tensor, 'batch_size'], Float[Tensor, 'batch_size']]:
        normalized_scores = torch.softmax(logits[:,-sequence:,:].to('cpu'),dim=-1)[:,:,list(self.positive_toks)+list(self.negative_toks)]

        normalized_positive,normalized_negative = torch.split(normalized_scores,[len(self.positive_toks), len(self.negative_toks)], dim=2)

        max_negative_score_per_sequence = torch.max(normalized_negative,dim=-1)[0]
        max_positive_score_per_sequence = torch.max(normalized_positive,dim=-1)[0]

        negative_score_per_batch = measure_fn(measure, max_negative_score_per_sequence, dim=-1)
        positive_score_per_batch = measure_fn(measure, max_positive_score_per_sequence, dim=-1)
        return negative_score_per_batch, positive_score_per_batch

    def do_resid(self, fn_name: str) -> Tuple[Float[Tensor, 'layer batch d_model'], Float[Tensor, 'layer batch d_model'], List[str]]:
        # Validate that caches are properly initialized as ActivationCache objects
        if not isinstance(self.harmful, ActivationCache) or not isinstance(self.harmless, ActivationCache):
            raise TypeError("Caches are not initialized. Run cache_activations first.")
        if not any("resid" in k for k in self.harmless.keys()):
            raise AssertionError("You need residual streams to decompose layers! Run cache_activations with 'resid_pre' or 'resid_post' in `activation_layers`")

        resid_harmful, labels = getattr(self.harmful, fn_name)(apply_ln=True, return_labels=True)
        resid_harmless = getattr(self.harmless, fn_name)(apply_ln=True)

        return resid_harmful, resid_harmless, labels

    def decomposed_resid(self) -> Tuple[Float[Tensor, 'layer batch d_model'], Float[Tensor, 'layer batch d_model'], List[str]]:
        return self.do_resid("decompose_resid")

    def accumulated_resid(self) -> Tuple[Float[Tensor, 'layer batch d_model'], Float[Tensor, 'layer batch d_model'], List[str]]:
        return self.do_resid("accumulated_resid")

    def unembed_resid(self, resid: Float[Tensor, "layer batch d_model"], pos: int = -1) -> Float[Tensor, "layer batch d_vocab"]:
        W_U = self.model.W_U
        if pos is None:
            return einops.einsum(resid.to(W_U.device), W_U, "layer batch d_model, d_model d_vocab -> layer batch d_vocab").to('cpu')
        else:
            return einops.einsum(resid[:, pos, :].to(W_U.device), W_U, "layer d_model, d_model d_vocab -> layer d_vocab").to('cpu')

    def create_layer_rankings(
        self,
        token_set: List[int]|Set[int]|Int[Tensor, '...'],
        decompose: bool = True,
        token_set_b: List[int]|Set[int]|Int[Tensor, '...'] = None
    ) -> List[Tuple[int,int]]:
        decomposer = self.decomposed_resid if decompose else self.accumulated_resid

        decomposed_resid_harmful, decomposed_resid_harmless, labels = decomposer()

        W_U = self.model.W_U.to('cpu')
        unembedded_harmful = self.unembed_resid(decomposed_resid_harmful)
        unembedded_harmless = self.unembed_resid(decomposed_resid_harmless)

        sorted_harmful_indices = torch.argsort(unembedded_harmful, dim=1, descending=True)
        sorted_harmless_indices = torch.argsort(unembedded_harmless, dim=1, descending=True)

        harmful_set = torch.isin(sorted_harmful_indices, torch.tensor(list(token_set)))
        harmless_set = torch.isin(sorted_harmless_indices, torch.tensor(list(token_set if token_set_b is None else token_set_b)))

        indices_in_set = zip(harmful_set.nonzero(as_tuple=True)[1],harmless_set.nonzero(as_tuple=True)[1])
        return indices_in_set

    def mse_positive(
        self,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1
    ) -> Dict[str, Float[Tensor, 'd_model']]:
        # Calculate mean squared error against currently loaded negative cached activation
        # Idea being to get a general sense of how the "normal" direction has been altered.
        # This is to compare ORIGINAL functionality to ABLATED functionality, not for ground truth.

        #load full training set to ensure alignment
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_train[:N]+self.harmless_inst_train[:N])

        splitpos = min(N,len(self.harmful_inst_train))

        # select for just harmless
        toks = toks[splitpos:]
        self.loss_harmless = {}

        for i in tqdm(range(0,min(N,len(toks)),batch_size)):
            logits,cache = self.run_with_cache(toks[i:min(i+batch_size,len(toks))])
            for key in cache:
                if any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :],dim=1).to('cpu')
                    if key not in self.loss_harmless:
                        self.loss_harmless[key] = tensor
                    else:
                        self.loss_harmless[key] = torch.cat((self.loss_harmless[key], tensor),dim=0)
            del logits,cache
            clear_mem()

        return {k:F.mse_loss(self.loss_harmless[k].float()[:N],self.harmless[k].float()[:N]) for k in self.loss_harmless}

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        measure_refusal: int = 0,
        stop_at_layer: int = None
    ) -> Tuple[ActivationCache, List[str]]:
        # Base functionality for creating an activation cache with a training set, prefer 'cache_activations' for regular usage

        base = dict()
        z_label = [] if measure_refusal > 1 else None
        for i in tqdm(range(0,min(N,len(toks)),batch_size)):
            logits,cache = self.run_with_cache(toks[i:min(i+batch_size,len(toks))],max_new_tokens=measure_refusal,stop_at_layer=stop_at_layer)
            if measure_refusal > 1:
                z_label.extend(self.measure_scores_from_logits(logits,measure_refusal)[0])
            for key in cache:
                if self.activation_layers is None or any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:,-last_indices:,:].to('cpu'),dim=1)
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)

            del logits, cache
            clear_mem()

        return ActivationCache(base,self.model), z_label

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        measure_refusal: int = 0,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_harmless: bool = True,
        stop_at_layer: int = None
    ):
        if hasattr(self, "current_state"):
            logger.warning("Caching activations while using a context manager")
        if self.modified:
            logger.warning("Running cache on modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        harmless_is_set = len(getattr(self,"harmless",{})) > 0
        preserve_harmless = harmless_is_set and preserve_harmless

        if reset == True or getattr(self,"harmless",None) is None:
            self.harmful = {}
            if not preserve_harmless:
                self.harmless = {}

            self.harmful_z_label = []
            self.harmless_z_label = []

        # load the full training set here to align all the dimensions (even if we're not going to run harmless)
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_train[:N]+self.harmless_inst_train[:N])

        splitpos = min(N,len(self.harmful_inst_train))
        harmful_toks = toks[:splitpos]
        harmless_toks = toks[splitpos:]

        last_indices = last_indices or 1

        self.harmful,self.harmful_z_label = self.create_activation_cache(harmful_toks,N=N,batch_size=batch_size,last_indices=last_indices,measure_refusal=measure_refusal,stop_at_layer=None)
        if not preserve_harmless:
            self.harmless, self.harmless_z_label = self.create_activation_cache(harmless_toks,N=N,batch_size=batch_size,last_indices=last_indices,measure_refusal=measure_refusal,stop_at_layer=None)

