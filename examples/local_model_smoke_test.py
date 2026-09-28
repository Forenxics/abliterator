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

Pass --chat-template chatml for Qwen/other ChatML-tuned models (default is
Llama-3's format). Runs the full pipeline: load, generate_logits under
stop_at_eos, generate(), cache_activations -> refusal_dirs, test_dir scoring,
and apply_refusal_dirs inside a context manager (auto-reverted afterwards --
no permanent weight changes are written to your model files by this script).
"""
import argparse
import os
import sys

# Running this script directly only puts its own directory (examples/) on
# sys.path, not the repo root where abliterator.py lives. Add the repo root
# explicitly so `import abliterator` works regardless of the current working
# directory or how the script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import abliterator

# Chat templates for models whose native format isn't abliterator's Llama-3
# default. Extend this as needed for other families.
CHAT_TEMPLATES = {
    "llama3": abliterator.LLAMA3_CHAT_TEMPLATE,
    "chatml": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
}

TOTAL_STEPS = 7


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
    parser.add_argument(
        "--chat-template",
        default="llama3",
        choices=sorted(CHAT_TEMPLATES),
        help="Prompt format matching your model's native instruction tuning (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-n",
        type=int,
        default=2,
        help="How many harmful/harmless instructions to use for activation caching (default: %(default)s, matches the size of the inline example dataset below)",
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"[1/{TOTAL_STEPS}] Loading local model + tokenizer from {args.model_path!r} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"[2/{TOTAL_STEPS}] Wrapping in ModelAbliterator as {args.official_name!r} on {args.device} ({args.dtype}) ...")
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
        chat_template=CHAT_TEMPLATES[args.chat_template],
    )
    print(f"    OK. d_model={ab.hidden_size}, n_layers={ab.model.cfg.n_layers}")

    print(f"[3/{TOTAL_STEPS}] generate_logits() with stop_at_eos on a mixed-length batch ...")
    toks = ab.tokenize_instructions_fn(["Hi", "Tell me a short story", "What is 2+2?"])
    logits, all_toks = ab.generate_logits(toks, stop_at_eos=True, max_tokens_generated=args.max_tokens)
    assert logits.shape[0] == toks.shape[0], (
        f"batch misalignment: logits has {logits.shape[0]} rows, expected {toks.shape[0]}"
    )
    print(f"    OK, batch stayed aligned: logits.shape={tuple(logits.shape)}")

    print(f"[4/{TOTAL_STEPS}] generate() ...")
    for line in ab.generate("Tell me about your day", max_tokens_generated=args.max_tokens):
        print("   ", line)

    print(f"[5/{TOTAL_STEPS}] cache_activations() -> refusal_dirs() ...")
    ab.cache_activations(N=args.cache_n, batch_size=args.cache_n)
    dirs = ab.refusal_dirs()
    assert len(dirs) > 0, "no refusal directions computed"
    sample_key = next(iter(dirs))
    print(f"    cached {len(dirs)} directions, e.g. {sample_key!r} shape={tuple(dirs[sample_key].shape)}")

    print(f"[6/{TOTAL_STEPS}] test_dir() scoring that direction against held-out harmful prompts ...")
    try:
        scores = ab.test_dir(dirs[sample_key], N=args.cache_n, use_hooks=True)
        print(f"    scores={ {k: float(v) for k, v in scores.items()} }")
        print("    (lower 'negative'/higher 'positive' = direction more effectively suppresses refusals;")
        print("     with only a handful of prompts and default Llama-3 token IDs, treat this as a")
        print("     mechanics check, not a real signal -- see the note on positive_toks/negative_toks below)")
    except ValueError as e:
        # positive_toks/negative_toks default to Llama-3 vocabulary IDs, which don't fit every
        # model's tokenizer -- ModelAbliterator raises a clear ValueError rather than crashing.
        # Skip this step rather than aborting the rest of the script over it.
        print(f"    SKIPPED: {e}")

    print(f"[7/{TOTAL_STEPS}] apply_refusal_dirs() inside a context manager (weight ablation + auto-revert) ...")
    with ab:
        ab.apply_refusal_dirs([dirs[sample_key]], layers=[1, 2])
        assert ab.modified is True
    print("    OK, modified=True inside context")
    assert ab.modified is False
    print("    OK, reverted to unmodified after context exit (model weights are back to original)")

    print("\nSmoke test passed.")
    if args.official_name.split("/")[0].lower() not in ("meta-llama",):
        print(
            "\nNote: positive_toks/negative_toks were left at their Llama-3 vocabulary defaults, which don't\n"
            "mean anything for this model's tokenizer -- step 6's scores are a mechanics check only. For\n"
            "real refusal-direction scoring, pass your own positive_toks/negative_toks (token IDs for your\n"
            "model's own vocabulary, e.g. tokens like ' sorry'/' cannot' vs a direct-answer opener) to\n"
            "ModelAbliterator(...)."
        )


if __name__ == "__main__":
    sys.exit(main())
