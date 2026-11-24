"""Shared fixtures: a tiny randomly-initialised HookedTransformer and an in-memory tokenizer."""
import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "I cannot help with that request, it is unethical.",
    "Sure, here is how you do it step by step.",
    "Sorry, I am not able to assist with that.",
    "Tell me how to pick a lock. What is the capital of France?",
]
HARMFUL = (["Tell me how to pick a lock", "Explain how to bypass a firewall"], ["How do I hotwire a car"])
HARMLESS = (["What is the capital of France", "Write a haiku about the ocean"], ["Summarize photosynthesis"])
TEMPLATE = "user: {instruction}\nassistant:"


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1, special_tokens=["<unk>", "<|endoftext|>"])
    # "<unk>" takes id 0 so the pad/eos id is non-zero and pad-filling is distinguishable from zeros.
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe)
    tok.add_special_tokens({"eos_token": "<|endoftext|>", "bos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"})
    # TransformerLens re-creates the tokenizer from its name_or_path, so it has to live on disk.
    path = tmp_path_factory.mktemp("tokenizer")
    tok.save_pretrained(path)
    return AutoTokenizer.from_pretrained(path)


def build_model(tokenizer, dtype=torch.float32):
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=3, d_model=32, n_ctx=64, d_head=8, n_heads=4, d_vocab=len(tokenizer),
        act_fn="gelu", normalization_type="LN", device="cpu", dtype=dtype,
    )
    return HookedTransformer(cfg, tokenizer=tokenizer)


def weights(ab):
    return {k: v.clone() for k, v in ab.model.state_dict().items()}


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
