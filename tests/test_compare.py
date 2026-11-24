"""compare_ablations.compare() on the tiny model: report structure, and the model is left unmodified."""
import importlib.util
import pathlib

from conftest import HARMFUL, HARMLESS, TEMPLATE, assert_same_weights, build_model, weights
from abliterator import ModelAbliterator

spec = importlib.util.spec_from_file_location(
    "compare_ablations", pathlib.Path(__file__).parent.parent / "examples" / "compare_ablations.py"
)
compare_ablations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_ablations)


def test_score_token_ids_and_refusal_detection(tokenizer):
    ids = compare_ablations.score_token_ids(tokenizer, ["Sure", "Sorry"])
    assert ids and all(isinstance(i, int) and 0 <= i < len(tokenizer) for i in ids)
    assert compare_ablations.is_refusal(" I'm sorry, but I can't help with that")
    assert not compare_ablations.is_refusal(" Sure, here is how")


def test_compare_reports_every_variant_and_restores_model(tokenizer):
    ab = ModelAbliterator(
        build_model(tokenizer), [HARMFUL, HARMLESS],
        positive_toks={5, 6}, negative_toks={7, 8}, chat_template=TEMPLATE,
    )
    before = weights(ab)
    report = compare_ablations.compare(ab, n=2, batch_size=2, n_eval=1, n_score=1, max_tokens=3)

    assert_same_weights(before, weights(ab))
    assert not ab.modified and ab.modified_layers == {"mlp": {}, "W_O": {}}
    assert report["direction"].endswith("hook_resid_pre")
    names = [row["variant"] for row in report["results"]]
    assert names == ["baseline (unmodified)"] + [v[0] for v in compare_ablations.VARIANTS]
    baseline, *variants = report["results"]
    assert abs(baseline["kl_harmless"]) < 1e-6
    assert all(row["kl_harmless"] > 0 for row in variants)
    for row in report["results"]:
        assert 0.0 <= row["refusal_rate"] <= 1.0 and len(row["samples"]) == 1
    import json
    json.dumps(report)  # serialisable
    compare_ablations.print_report(report)
