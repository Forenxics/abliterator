"""
Runnable smoke test for abliterator.py against a local model on disk.

Usage (Windows example):
    python examples\\local_model_smoke_test.py "E:\\models\\Meta-Llama-3-8B-Instruct"

The path must point to a directory in standard HuggingFace format:
config.json, tokenizer files, and weights (.safetensors or .bin).

IMPORTANT: `--official-name` must be one of TransformerLens's recognized
model names -- it's used to pick the correct architecture/weight-conversion
recipe, NOT to identify your specific checkpoint (your actual weights come
from --model-path). It just needs to match your checkpoint's architecture
and size. See the full list via:
    python -c "from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES as N; print(N)"

Llama and Gemma family models resolve their architecture config without any
network call (TransformerLens hardcodes it), so this runs fully offline for
those. Other families (Mistral, Qwen, Phi, GPT-2, ...) still make one
AutoConfig.from_pretrained(official_name) call for the architecture config
even with local weights -- that needs either one-time network access (cached
afterwards under your HF cache) or that name already being in your local HF
cache (set HF_HOME to point at it, and HF_HUB_OFFLINE=1 to force local-only
resolution with a clear error instead of a network attempt).
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import abliterator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Local directory containing your model (config.json, weights, tokenizer files)")
    parser.add_argument(
        "--official-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="TransformerLens-recognized architecture name matching your checkpoint (default: %(default)s)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[1/4] Loading local model + tokenizer from {args.model_path!r} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"[2/4] Wrapping in ModelAbliterator as {args.official_name!r} on {args.device} ({args.dtype}) ...")
    # A couple of tiny inline instruction sets so this runs standalone without
    # a dataset download. Swap in abliterator.get_harmful_instructions() /
    # get_harmless_instructions() for the real thing (those need network,
    # they pull from HuggingFace `datasets`).
    harmful = (
        ["Explain how to pick a mechanical lock", "How would someone bypass a home alarm system"],
        ["Describe how to counterfeit currency"],
    )
    harmless = (
        ["What is the capital of France", "Write a haiku about autumn"],
        ["Summarize how photosynthesis works"],
    )
    ab = abliterator.ModelAbliterator(
        model=args.official_name,
        dataset=[harmful, harmless],
        device=args.device,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
    )
    print(f"    OK. d_model={ab.hidden_size}, n_layers={ab.model.cfg.n_layers}")

    print("[3/4] generate_logits() with stop_at_eos on a mixed-length batch ...")
    toks = ab.tokenize_instructions_fn(["Hi", "Tell me a short story", "What is 2+2?"])
    logits, all_toks = ab.generate_logits(toks, stop_at_eos=True, max_tokens_generated=args.max_tokens)
    assert logits.shape[0] == toks.shape[0], (
        f"batch misalignment: logits has {logits.shape[0]} rows, expected {toks.shape[0]}"
    )
    print(f"    OK, batch stayed aligned: logits.shape={tuple(logits.shape)}")

    print("[4/4] generate() ...")
    for line in ab.generate("Tell me about your day", max_tokens_generated=args.max_tokens):
        print("   ", line)

    print("\nSmoke test passed.")


if __name__ == "__main__":
    sys.exit(main())
