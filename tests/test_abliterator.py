"""Offline tests on a tiny randomly-initialised HookedTransformer.

Random weights mean the outputs are meaningless; these check mechanics
(shapes, bookkeeping, save/restore, error handling), not ablation quality.
"""
import logging

import pytest
import torch

from abliterator import ModelAbliterator, RESID_DECOMPOSITION_LAYERS, measure_fn, prepare_dataset
from conftest import HARMFUL, HARMLESS, TEMPLATE, assert_same_weights, build_model, weights


def make(tokenizer, **kwargs):
    kwargs.setdefault("positive_toks", {5, 6})
    kwargs.setdefault("negative_toks", {7, 8})
    kwargs.setdefault("chat_template", TEMPLATE)
    return ModelAbliterator(build_model(tokenizer), [HARMFUL, HARMLESS], **kwargs)


@pytest.fixture
def ab(tokenizer):
    return make(tokenizer)


@pytest.fixture
def cached(ab):
    ab.cache_activations(N=2, batch_size=2)
    return ab


# --- module-level helpers -------------------------------------------------

def test_prepare_dataset_forms():
    assert prepare_dataset(([], ["t"])) == ([], ["t"])
    assert prepare_dataset((["a", "b"], ["c"])) == (["a", "b"], ["c"])
    assert prepare_dataset([["a", "b"], ["c"]]) == (["a", "b"], ["c"])  # list pair, e.g. from JSON
    train, test = prepare_dataset([str(i) for i in range(10)])
    assert len(train) + len(test) == 10 and all(isinstance(x, str) for x in train + test)


def test_measure_fn_returns_tensors():
    x = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
    assert measure_fn("max", x).item() == 3.0
    assert torch.equal(measure_fn("max", x, dim=-1), torch.tensor([3.0, 2.0]))
    assert isinstance(measure_fn("median", x, dim=0), torch.Tensor)
    with pytest.raises(NotImplementedError):
        measure_fn("nope", x)


# --- generation -----------------------------------------------------------

def test_generate_logits_keeps_batch_and_pads_unwritten(ab):
    toks = ab.tokenize_instructions_fn(["Hi", "Tell me a short story", "What is 2+2?"])
    logits, all_toks = ab.generate_logits(toks, stop_at_eos=True, drop_refusals=False, max_tokens_generated=4)
    assert logits.shape[0] == 3 and all_toks.shape == (3, toks.shape[1] + 4)

    # Force an immediate refusal break: everything after the first new token stays pad.
    first = ab.generate_logits(toks, max_tokens_generated=1, drop_refusals=False)[1][:, -1]
    ab.negative_toks = {int(first[0])}
    _, broken = ab.generate_logits(toks, max_tokens_generated=4, drop_refusals=True)
    assert ab.model.tokenizer.pad_token_id != 0
    assert (broken[:, toks.shape[1] + 1:] == ab.model.tokenizer.pad_token_id).all()


def test_generate_returns_one_string_per_prompt(ab):
    out = ab.generate(["Hi", "What is 2+2?"], max_tokens_generated=3)
    assert len(out) == 2 and all(isinstance(o, str) for o in out)


# --- caching and directions ----------------------------------------------

def test_cache_activation_layer_variants(ab):
    ab.cache_activations(N=2, batch_size=2, activation_layers=["resid_pre"], preserve_harmless=False)
    assert all("resid_pre" in k for k in ab.harmful.keys()) and len(ab.harmful.keys()) == 3

    ab.cache_activations(N=2, batch_size=2, activation_layers=None, preserve_harmless=False)
    assert "ln_final.hook_scale" in ab.harmful.keys()
    assert all(v.ndim == 2 for v in ab.harmful.cache_dict.values())  # 4-D attention tensors skipped

    ab.cache_activations(N=2, batch_size=2, preserve_harmless=False)
    assert len(ab.harmful.keys()) == 3 * 4 and "ln_final.hook_scale" not in ab.harmful.keys()


def test_refusal_dirs_are_unit_dmodel_and_skip_layer0(cached):
    dirs = cached.refusal_dirs()
    assert dirs and not any(".0." in k for k in dirs)
    for v in dirs.values():
        assert v.shape == (32,) and torch.isfinite(v).all()
        assert abs(v.norm().item() - 1.0) < 1e-4


def test_refusal_dirs_tolerates_mismatched_caches(cached, caplog):
    # harmful re-cached with more layer types while harmless is preserved
    cached.cache_activations(N=2, batch_size=2, activation_layers=None)
    with caplog.at_level(logging.WARNING):
        dirs = cached.refusal_dirs()
    assert dirs and all(k in cached.harmless.keys() for k in dirs)
    assert "no harmless counterpart" in caplog.text


def test_scored_dirs_and_find_best_ordering(cached):
    assert len(cached.scored_dirs()) > 0
    neg = cached.find_best_refusal_dir(N=1)
    pos = cached.find_best_refusal_dir(N=1, positive=True)
    assert [float(s) for s, _ in neg] == sorted(float(s) for s, _ in neg)
    assert [float(s) for s, _ in pos] == sorted((float(s) for s, _ in pos), reverse=True)


def test_test_dir_respects_layers_and_restores_hooks(cached):
    direction = next(iter(cached.refusal_dirs().values()))
    for use_hooks in (True, False):
        before = weights(cached)
        scores = cached.test_dir(direction, N=1, use_hooks=use_hooks, layers=[1])
        assert set(scores) == {"negative", "positive"}
        assert cached.fwd_hooks == [] and not cached.modified
        assert_same_weights(before, weights(cached))


def test_out_of_vocab_score_tokens_raise_clearly(tokenizer):
    ab = make(tokenizer, positive_toks={5}, negative_toks={10_000_000})
    ab.cache_activations(N=2, batch_size=2)
    with pytest.raises(ValueError, match="outside this model's vocabulary"):
        ab.test_dir(next(iter(ab.refusal_dirs().values())), N=1)


def test_constructor_activation_layers_none(tokenizer):
    ab = make(tokenizer, activation_layers=None)
    ab.cache_activations(N=2, batch_size=2)
    dirs = ab.refusal_dirs()
    assert set(ab.test_dir(next(iter(dirs.values())), N=1)) == {"negative", "positive"}
    assert ab.get_layer_dirs(1)["harmful_mean"].shape == (32,)
    assert ab.mse_positive(N=2, batch_size=2)


# --- weight edits and bookkeeping ----------------------------------------

def test_apply_changes_weights_and_removes_direction(cached):
    direction = next(iter(cached.refusal_dirs().values()))
    cached.apply_refusal_dirs([direction.double()], layers=[1, 2])  # other dtype is cast
    assert cached.modified
    for layer in (1, 2):
        assert (cached.layer_mlp(layer) @ direction).abs().max() < 1e-4
        assert (cached.layer_attn(layer) @ direction).abs().max() < 1e-4
    assert 0 not in cached.modified_layers["mlp"]


def test_history_records_old_then_new(cached):
    old = cached.layer_attn(1).clone()
    cached.layer_attn(1, old * 0.5)
    cached.layer_attn(1, old * 0.25)
    history = cached.modified_layers["W_O"][1]
    assert len(history) == 2
    assert torch.equal(history[0][0], old) and torch.equal(history[0][1], old * 0.5)
    assert torch.equal(history[1][0], old * 0.5) and torch.equal(history[1][1], old * 0.25)


def test_blacklisted_layer_is_untouched(cached):
    cached.blacklist_layer([1])
    before = cached.layer_mlp(1).clone()
    cached.apply_refusal_dirs([next(iter(cached.refusal_dirs().values()))], layers=[1, 2])
    assert torch.equal(before, cached.layer_mlp(1)) and 1 not in cached.modified_layers["mlp"]


def test_context_manager_restores_weights_and_bookkeeping(cached):
    direction = next(iter(cached.refusal_dirs().values()))
    before = weights(cached)
    with cached:
        cached.apply_refusal_dirs([direction])
        assert cached.modified and cached.modified_layers["mlp"]
        assert not torch.equal(before["blocks.1.mlp.W_out"], cached.layer_mlp(1))
    assert_same_weights(before, weights(cached))
    assert not cached.modified
    assert cached.modified_layers == {"mlp": {}, "W_O": {}}
    with pytest.raises(RuntimeError):
        with cached:
            with cached:
                pass


def test_checkpoint_is_a_snapshot(cached):
    cached.layer_mlp(1, cached.layer_mlp(1) * 0.5)
    cached.checkpoint()
    cached.layer_mlp(2, cached.layer_mlp(2) * 0.5)
    assert set(cached.checkpoints[0]["mlp"]) == {1}


def test_reset_state_restores_original(cached):
    before = weights(cached)
    cached.apply_refusal_dirs([next(iter(cached.refusal_dirs().values()))])
    cached.reset_state()
    assert_same_weights(before, weights(cached))
    assert not cached.modified


def test_induce_refusal_dir_runs_and_reverts(cached):
    direction = next(iter(cached.refusal_dirs().values()))
    before = weights(cached)
    with cached:
        cached.induce_refusal_dir(direction, layers=[1])
        assert torch.isfinite(cached.layer_mlp(1)).all()
        assert not torch.equal(before["blocks.1.mlp.W_out"], cached.layer_mlp(1))
    assert_same_weights(before, weights(cached))


# --- save / load ----------------------------------------------------------

def test_save_load_roundtrip_reapplies_edits(cached, tokenizer, tmp_path):
    cached.apply_refusal_dirs([next(iter(cached.refusal_dirs().values()))], layers=[1, 2])
    cached.checkpoint()
    fname = str(tmp_path / "state.pth")
    cached.save_activations(fname)
    torch.load(fname, weights_only=True)  # no pickled objects

    fresh = make(tokenizer, cache_fname=fname)
    assert fresh.modified
    assert_same_weights(weights(cached), weights(fresh))
    assert set(fresh.harmful.keys()) == set(cached.harmful.keys())
    assert len(fresh.checkpoints) == 1
    assert fresh.refusal_dirs().keys() == cached.refusal_dirs().keys()


def test_save_load_with_nothing_cached(ab, tokenizer, tmp_path):
    fname = str(tmp_path / "empty.pth")
    ab.save_activations(fname)
    fresh = make(tokenizer, cache_fname=fname)
    assert not fresh.modified and not fresh.harmful and fresh.modified_layers == {"mlp": {}, "W_O": {}}


def test_legacy_cache_needs_opt_in(cached, tokenizer, tmp_path):
    fname = str(tmp_path / "legacy.pth")
    torch.save([cached.harmful, cached.harmless, None, None], fname)  # pre-format-2 layout
    with pytest.raises(RuntimeError, match="allow_unsafe_cache"):
        make(tokenizer, cache_fname=fname)
    fresh = make(tokenizer, cache_fname=fname, allow_unsafe_cache=True)
    assert set(fresh.harmful.keys()) == set(cached.harmful.keys())


# --- residual decomposition ----------------------------------------------

def test_do_resid_needs_the_right_cache(cached):
    with pytest.raises(AssertionError, match="RESID_DECOMPOSITION_LAYERS"):
        cached.decomposed_resid()


@pytest.mark.parametrize("decompose", [True, False])
def test_create_layer_rankings(ab, decompose):
    ab.cache_activations(N=2, batch_size=2, activation_layers=RESID_DECOMPOSITION_LAYERS, preserve_harmless=False)
    labels = (ab.decomposed_resid() if decompose else ab.accumulated_resid())[2]
    rankings = ab.create_layer_rankings({10, 20, 30}, decompose=decompose)
    assert isinstance(rankings, list) and len(rankings) == len(labels) * 3
    d_vocab = ab.model.cfg.d_vocab
    assert all(isinstance(l, str) and 0 <= h < d_vocab and 0 <= hl < d_vocab for l, h, hl in rankings)
    assert rankings == ab.create_layer_rankings({10, 20, 30}, decompose=decompose, token_set_b={10, 20, 30})
    with pytest.raises(ValueError, match="same length"):
        ab.create_layer_rankings({10, 20}, decompose=decompose, token_set_b={10})
    # refusal_dirs still works on this wider cache (non-d_model keys are ignored)
    assert all(v.shape == (32,) for v in ab.refusal_dirs().values())


# --- README usage, as written --------------------------------------------

def test_readme_find_and_apply_flow(cached):
    score, (act_name, my_amazing_dir) = cached.find_best_refusal_dir(N=1)[0]
    cached.apply_refusal_dirs([my_amazing_dir], layers=None)
    with cached:
        cached.apply_refusal_dirs([my_amazing_dir])
        assert cached.mse_positive(N=2, batch_size=2)


# --- reduced-precision models --------------------------------------------

def test_bfloat16_model_gets_float32_directions_and_keeps_weight_dtype(tokenizer):
    ab = ModelAbliterator(
        build_model(tokenizer, dtype=torch.bfloat16), [HARMFUL, HARMLESS],
        positive_toks={5, 6}, negative_toks={7, 8}, chat_template=TEMPLATE,
    )
    ab.cache_activations(N=2, batch_size=2)
    assert next(iter(ab.harmful.cache_dict.values())).dtype == torch.bfloat16
    dirs = ab.refusal_dirs()
    direction = next(iter(dirs.values()))
    assert direction.dtype == torch.float32

    assert set(ab.test_dir(direction, N=1, use_hooks=True)) == {"negative", "positive"}
    with ab:
        ab.apply_refusal_dirs([direction], layers=[1])
        ab.induce_refusal_dir(direction, layers=[2])
        assert ab.layer_mlp(1).dtype == torch.bfloat16 and ab.layer_attn(2).dtype == torch.bfloat16
    assert ab.layer_mlp(1).dtype == torch.bfloat16


# --- research-derived options (opt-in) -----------------------------------

def test_projected_directions_are_orthogonal_to_harmless_mean(cached):
    plain, projected = cached.refusal_dirs(), cached.refusal_dirs(projected=True)
    assert plain.keys() == projected.keys()
    for key, v in projected.items():
        harmless_mean = cached.calculate_mean_dirs(key)["harmless_mean"]
        assert abs(v.norm().item() - 1.0) < 1e-4
        assert abs(torch.dot(v, harmless_mean / harmless_mean.norm()).item()) < 1e-4
    assert any(not torch.allclose(plain[k], projected[k], atol=1e-4) for k in plain)


def test_preserve_norm_keeps_row_norms_and_orthogonality(cached):
    direction = next(iter(cached.refusal_dirs().values()))
    before_mlp, before_attn = cached.layer_mlp(1).clone(), cached.layer_attn(1).clone()
    cached.apply_refusal_dirs([direction], layers=[1], preserve_norm=True)
    for before, after in ((before_mlp, cached.layer_mlp(1)), (before_attn, cached.layer_attn(1))):
        assert torch.allclose(before.norm(dim=-1), after.norm(dim=-1), rtol=1e-4, atol=1e-6)
        assert (after @ direction).abs().max() < 1e-4
        assert not torch.equal(before, after)


def test_kl_divergence_zero_until_modified(cached):
    baseline = cached.harmless_logprobs(N=1, batch_size=1)
    assert baseline.shape == (1, cached.model.cfg.d_vocab)
    assert abs(cached.kl_divergence(baseline, N=1, batch_size=1)) < 1e-6
    with cached:
        cached.apply_refusal_dirs([next(iter(cached.refusal_dirs().values()))])
        assert cached.kl_divergence(baseline, N=1, batch_size=1) > 0
    assert abs(cached.kl_divergence(baseline, N=1, batch_size=1)) < 1e-6
    with pytest.raises(ValueError, match="same N"):
        cached.kl_divergence(torch.zeros(3, 5), N=1)
