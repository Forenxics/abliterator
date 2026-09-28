# abliterator.py

Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens.

Most of its advantage in workflow comes from being able to enter temporary contexts, quickly cache activations with N samples, refusal direction calculation built-in, and tokenizer utilities. As well as wrapping around certain quirks of TransformerLens.

If you're interested in notebooking your own orthogonalized model, this library will help save you a LOT of time in performing and measuring experiments to find your best orthogonalization.

This is ultimately just bits and pieces to make it so to process and experiment with ablation direction turns into shorter, hopefully clearer code without losing track of where things are at; encapsulating a lot of useful logic that you'll find yourself writing if you're looking to do this more.

Right now, this works very well for the personal workflow it grew out of, but the goal is to systematize and ideally automate this further, and broaden out from the pure "harmless / harmful" feature ablation, to augmentation, and adding additional features.

## Installation

```bash
git clone https://github.com/Forenxics/abliterator.git
cd abliterator
pip install -r requirements.txt
```

**Important dependency note:** this library uses TransformerLens's `HookedTransformer` API, which was **removed in TransformerLens 4.0** (replaced by a new `TransformerBridge` API). `requirements.txt`/`pyproject.toml` pin `transformer_lens>=3.0.0,<4.0.0` for exactly this reason — don't `pip install -U transformer_lens` on top of this, it'll break the import.

## Loading a model in

### From the Hugging Face Hub

```python
import abliterator

model = "meta-llama/Meta-Llama-3-70B-Instruct"  # the huggingface repo id or local path of the model you're interested in loading in
dataset = [abliterator.get_harmful_instructions(), abliterator.get_harmless_instructions()] # datasets to be used for caching and testing, split by harmful/harmless
device = 'cuda'                             # optional: defaults to cuda
n_devices = None                            # optional: when set to None, defaults to `torch.cuda.device_count()`
cache_fname = 'my_cached_point.pth'         # optional: if you need to save where you left off, you can use `save_activations(filename)` which will write out a file. This is how you load that back in.
activation_layers = ['resid_pre', 'resid_post', 'attn_out', 'mlp_out']  # optional: this is the library default. Set to None to cache ALL activation layer types instead.
chat_template = None                        # optional: defaults to the Llama-3 instruction template. You can use a format string e.g. ("<system>{instruction}<end><assistant>") or a custom class with a `.format(instruction="")` function. See abliterator.ChatTemplate for a very basic structure.
negative_toks = [4250]                      # optional, but highly recommended: ' cannot' in Llama's tokenizer. Tokens you don't want to be seeing. Defaults to a preset for Llama-3 models.
positive_toks = [23371, 40914]              # optional, but highly recommended: ' Sure' and 'Sure' in Llama's tokenizer. Tokens you want to be seeing, basically. Defaults to a preset for Llama-3 models.

my_model = abliterator.ModelAbliterator(
  model,
  dataset,
  device='cuda',
  n_devices=None,
  cache_fname=None,
  activation_layers=activation_layers,
  chat_template="<system>\n{instruction}<end><assistant>",
  positive_toks=positive_toks,
  negative_toks=negative_toks
)
```

`model` needs to be one of TransformerLens's recognized architecture names — see the full list with:
```python
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
print(OFFICIAL_MODEL_NAMES)
```
It's used to pick the correct weight-conversion recipe; by default the *same* name is also what gets downloaded from the Hub.

### From a local directory (offline / already-downloaded weights)

If you have weights already downloaded somewhere on disk — a HuggingFace-format directory with `config.json`, tokenizer files, and `.safetensors`/`.bin` weights — load them via `hf_model=`/`tokenizer=` instead of letting `ModelAbliterator` download from the Hub. `model=` still needs to be one of the TransformerLens-recognized names above (it's used purely to pick the architecture/conversion recipe — your actual weights come from `hf_model`):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import abliterator

hf_model = AutoModelForCausalLM.from_pretrained("/path/to/local/model", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/path/to/local/model")

my_model = abliterator.ModelAbliterator(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # architecture name matching your checkpoint, not a Hub download
    dataset=[abliterator.get_harmful_instructions(), abliterator.get_harmless_instructions()],
    device="cuda",
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,  # bfloat16 is the ModelAbliterator default; override to float32 (or float16) if your hardware/backend doesn't support it well -- most CPUs, for instance
)
```

A full, runnable version of this is in [`examples/local_model_smoke_test.py`](examples/local_model_smoke_test.py) — see [Local-model quickstart](#local-model-quickstart) below.

**Whether this needs network access depends on the model family:**
- **Llama and Gemma family** models resolve their architecture config from TransformerLens's own hardcoded tables — zero network calls, fully offline, even on first run.
- **Everything else** (Mistral, Qwen, Phi, GPT-2, etc.) still makes one `AutoConfig.from_pretrained(model_name)` call for the architecture config, even with local weights supplied via `hf_model=`. That needs either one-time network access (results get cached under your normal HuggingFace cache afterwards) or that model already being in your local HF cache with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set to force local-only resolution.

**GGUF models (Ollama, LM Studio, llama.cpp) are not supported.** Both `transformers.AutoModelForCausalLM` and TransformerLens expect standard PyTorch/safetensors weights; they cannot load `.gguf` files at all. If your only copy of a model is in Ollama's or LM Studio's model store, you'll need a separate HuggingFace-format download of the same model (e.g. via `hf download <repo_id> --local-dir <path>`, see below) — this library can't work from GGUF weights directly.

## Cache activations/sample dataset
Once loaded in, run the model against N samples of harmful, and N samples of harmless so it has some data to work with:
```python
my_model.cache_activations(N=512,reset=True,preserve_harmless=True)
```
`preserve_harmless=True` is generally useful, as it keeps the "desired behaviour" unaltered from any stacked modifications if you run it after some mods.

## Saving state
Most of the advantage of this is a lot of groundwork has been laid to make it so you aren't repeating yourself 1000 times just to try one little experiment.
`save_activations('file.pth')` will save your cached activations, and any currently applied modifications to the model's weights to a file so you can restore them next time you load up with `cache_fname='file.pth'` in your ModelAbliterator initialization.

## Getting refusal directions from the cached activations
Speaking of modding, here's a simple representation of how to pick, test, and actually apply a direction from a layer's activations:
```python
refusal_dirs = my_model.refusal_dirs()
testing_dir = refusal_dirs['blocks.18.hook_resid_pre']
my_model.test_dir(testing_dir, N=32, use_hooks=True) # I recommend use_hooks=True for large models as it can slow things down otherwise, but use_hooks=False can give you more precise scoring to an actual weights modification
```
`test_dir` will apply your refusal_dir to the model temporarily, and run against N samples of test data, and return a composite (negative_score, positive_score) from those runs. Generally, you want negative_score to go down, positive_score to go up.

### Testing lots of refusal directions

This is one of the functions included in the library, but it's also useful for showing how this can be generalized to test a whole bunch of directions.
```python
    def find_best_refusal_dir(N=4, use_hooks=True, invert=False):
        dirs = self.refusal_dirs(invert=invert)
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1],N=N,use_hooks=use_hooks)[0]
            scores.append((score,direction))
        return sorted(scores,key=lambda x:x[0])

```

## Applying the weights

And now, to apply it!
```python
my_amazing_dir = find_best_refusal_dir()[0]
my_model.apply_refusal_dirs([my_amazing_dir],layers=None)
```
Note the `layers=None`. You can supply a list here to specify which layers you want to apply the refusal direction to. None will apply it to all writable layers.

### Blacklisting specific layers
Sometimes some layers are troublesome no matter what you do. If you're worried about accidentally replacing it, you can blacklist it to prevent any alteration from occurring:
```python
my_model.blacklist_layer(27)
my_model.blacklist_layer([i for i in range(27,30)]) # it also accepts lists!
```

#### Whitelisting
And naturally, to undo this and make sure a layer can be overwritten:
```
my_model.whitelist_layer(27)
```
By default, all layers are whitelisted. I recommend blacklisting the first and last couple layers, as those can and will have dramatic effects on outputs.

Neither of these will provide success/failure states. They will just assure the desired state in running it at that instant.

## Benchmarking
Now to make sure you've not damaged the model dramatically after applying some stuff, you can do a test run:
```python
with my_model: # loads a temporary context with the model
  my_model.apply_refusal_dirs([my_new_precious_dir]) # Because this was applied in the 'with my_model:', it will be unapplied after coming out.
  print(my_model.mse_positive(N=128)) # While we've got the dir applied, this tells you the Mean Squared Error using the currently cached harmless activations as "ground truth" (loss function, effectively)
```

### Want to see it run? Test it!
```python
my_model.test(N=16,batch_size = 4) # runs N samples from the harmful test set and prints them for the user. Good way to check the model hasn't completely derailed.
# Note that by default if a test run produces a negative token, it will stop the whole batch and move on to the next. (it will show lots of '!!!!' in Llama-3's case, as that's token ID 0)

my_model.generate("How much wood could a woodchuck chuck if a woodchuck could chuck wood?") # runs and prints the prompt!
```

## Custom `positive_toks`/`negative_toks` for non-Llama-3 models

`ModelAbliterator`'s built-in defaults for `positive_toks`/`negative_toks` are hardcoded Llama-3 vocabulary token IDs. If you don't override them for a different model:

- `generate_logits`' `drop_refusals` early-stop just silently becomes a no-op (those IDs never come up in a different tokenizer's output) — harmless.
- `test_dir`/`measure_scores`/`find_best_refusal_dir` will raise a clear `ValueError` if any default ID is out of range for your model's vocabulary (this is common — GPT-2's vocab is 50257, Llama-2/Mistral's is 32000, both smaller than some of the default IDs). Large-vocab models (Qwen2.5, for instance, at 151936) happen not to trip this, but the IDs still don't mean anything for a non-Llama-3 tokenizer, so scores from them aren't meaningful either way.

For real refusal-direction scoring on another model, find token IDs for refusal-ish vs. compliance-ish openers in *that model's own tokenizer*, e.g.:
```python
tokenizer = my_model.model.tokenizer
negative_toks = {tokenizer.encode(" sorry", add_special_tokens=False)[0], tokenizer.encode(" cannot", add_special_tokens=False)[0]}
positive_toks = {tokenizer.encode(" Sure", add_special_tokens=False)[0], tokenizer.encode(" Here", add_special_tokens=False)[0]}
```
and pass those in as `positive_toks=`/`negative_toks=` when constructing `ModelAbliterator`.

## Custom chat templates

If your model wasn't tuned on the Llama-3 chat format, pass `chat_template=` as a plain string with an `{instruction}` placeholder. ChatML (used by Qwen and several other instruct models) looks like:
```python
chat_template = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
```

## Local-model quickstart

[`examples/local_model_smoke_test.py`](examples/local_model_smoke_test.py) is a runnable, standalone script covering the whole pipeline against a model already on disk: load, `generate_logits` under `stop_at_eos`, `generate()`, `cache_activations` → `refusal_dirs`, `test_dir` scoring, and `apply_refusal_dirs` inside a context manager (with auto-revert — it doesn't permanently modify your model files).

```bash
python examples/local_model_smoke_test.py /path/to/local/model --official-name meta-llama/Meta-Llama-3-8B-Instruct
```

Run `python examples/local_model_smoke_test.py --help` for the full flag list (`--chat-template`, `--dtype`, `--device`, `--max-tokens`, `--cache-n`).

### Getting a local model

To download a clean, flat local copy of a model (not nested under a hash, so `--local-dir` works directly with the smoke test):
```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```
(`huggingface-cli` was the old command name; recent `huggingface_hub` versions renamed it to plain `hf`, same subcommands.)

Gated models (most Meta Llama and Google Gemma repos) need a free Hugging Face account, accepting the license on the model's page, and being logged in (`hf auth login`) before the download will work.

## Utility functions
Documentation coming soon.

## How to Save as a HuggingFace model
Functionality coming soon. For now, use PyTorch's saving method.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'transformer_lens'` or `HookedTransformer` import errors** — check you have `transformer_lens<4.0` installed (`pip show transformer_lens`); 4.0+ removed `HookedTransformer` entirely. Reinstall from `requirements.txt` if needed.
- **`OSError: Repo id must use alphanumeric chars...` when passing a local path** — this means `transformers` didn't see a directory at that exact path (so it fell through to treating the string as a Hub repo id instead). Double check the path is correct and directly contains `config.json` — `os.path.isdir(your_path)` should return `True`.
- **A downloaded model "isn't found" at the path you expect** — if you used `huggingface-cli download`/`hf download` *without* `--local-dir`, files land in your HF cache under a hashed `snapshots/<hash>/` subfolder, not flat in a folder you named. Use `--local-dir` to get a flat, predictable layout.
- **Symlink warnings on Windows** (`Caching files will still work but in a degraded version...`) — harmless; it just means downloads use plain file copies instead of symlinks unless you enable Developer Mode or run as administrator. Safe to ignore, or silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **Your model came from Ollama or LM Studio** — those store GGUF weights, which neither `transformers` nor TransformerLens can load. You'll need a separate HuggingFace-format download of the same model.
