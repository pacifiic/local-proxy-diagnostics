"""Small synthetic contract tests. No model downloads, GPU use or real-data runs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIMENSIONS, digest, encode_pair, new_directory, shared_article,
                    sha256, validate_inputs, validate_scores, verify_prepared,
                    verify_aligned_source, write_json)
from prepare import make_splits
from analyze import (PathData, bootstrap_coefficients, describe, evaluate_matched,
                     prospective, summarize, write_table)
from score import scalar_score
import prepare
import analyze


def fixture(n=40):
    items, human = [], []
    for i in range(n):
        article_id, model_id = f"article-{i // 4:03}", f"M{i % 4}"
        key = f"{article_id}||{model_id}"
        items.append(dict(article_id=article_id, model_id=model_id, item_id=key,
                          article=f"Original article {i // 4}", summary=f"Summary {i}"))
        for dim in DIMENSIONS:
            for kind, count in [("expert", 3), ("crowd", 5)]:
                for rater in range(count):
                    human.append(dict(article_id=article_id, model_id=model_id, dimension=dim,
                                      rater_type=kind, rater_idx=rater, score=1 + (i + rater) % 5))
    return pd.DataFrame(items), pd.DataFrame(human)


def aligned_fixture(path, items, human):
    with path.open("x") as f:
        for row in items.itertuples():
            h = human[(human.article_id == row.article_id) & (human.model_id == row.model_id)]
            record = dict(id=row.article_id, model_id=row.model_id, decoded=row.summary)
            for pool, field in [("expert", "expert_annotations"), ("crowd", "turker_annotations")]:
                record[field] = [g.set_index("dimension").score.to_dict()
                                 for _, g in h[h.rater_type == pool].groupby("rater_idx", sort=True)]
            f.write(json.dumps(record) + "\n")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "BOS" + messages[0]["content"] + "ASSISTANT" + messages[1]["content"]

    def __call__(self, first, second=None, **kwargs):
        assert not kwargs.get("truncation")
        return {"input_ids": list(range(len(first) + len(second or "") + 2))}


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.value


class Contracts(unittest.TestCase):
    def test_complete_rating_matrices(self):
        items, human = fixture()
        a, b = validate_inputs(items, human, 40)
        self.assertEqual(len(a), 40)
        self.assertEqual(len(b), 40 * 4 * 8)

    def test_canonical_turker_pool_is_preserved(self):
        items, human = fixture()
        human["rater_type"] = human.rater_type.replace({"crowd": "turker"})
        _, checked = validate_inputs(items, human, 40)
        self.assertEqual(set(checked.rater_type), {"expert", "turker"})

    def test_mapped_ids_cannot_be_silently_reassigned(self):
        items, human = fixture()
        items.loc[0, "item_id"] = items.item_id.iloc[1]
        with self.assertRaisesRegex(ValueError, "item_id"):
            validate_inputs(items, human, 40)

    def test_text_conflicts_and_missing_ratings_fail(self):
        items, human = fixture()
        duplicate = items.iloc[[0]].assign(summary="Wrong summary")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            validate_inputs(pd.concat([items, duplicate], ignore_index=True), human, 40)
        with self.assertRaises(ValueError):
            validate_inputs(items, human.iloc[1:], 40)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_inputs(items, pd.concat([human, human.iloc[[0]]]), 40)

    def test_nan_score_fails(self):
        items, human = fixture()
        human["score"] = human.score.astype(float)
        human.loc[0, "score"] = np.inf
        with self.assertRaises(ValueError):
            validate_inputs(items, human, 40)

    def test_score_join_is_exact_and_order_invariant(self):
        items, _ = fixture()
        items["text_sha256"] = [digest(x) for x in items.summary]
        scores = items[["item_id", "text_sha256"]].assign(proxy_score=np.arange(40.))
        np.testing.assert_equal(validate_scores(scores.sample(frac=1, random_state=2), items), np.arange(40.))
        for bad in [scores.iloc[1:], pd.concat([scores, scores.iloc[[0]]]), scores.assign(text_sha256="wrong")]:
            with self.assertRaises(ValueError):
                validate_scores(bad, items)

    def test_common_context_preserves_summary_and_all_budgets(self):
        models = {"a": {"input_format": "chat", "max_length": 100},
                  "b": {"input_format": "pair", "max_length": 80}}
        tokenizers = {k: Tokenizer() for k in models}
        article, sizes = shared_article("x" * 200, "Full summary", tokenizers, models)
        self.assertGreater(len(article), 0)
        self.assertLess(len(article), 200)
        self.assertTrue(all(sizes[k] <= models[k]["max_length"] for k in models))
        with self.assertRaises(ValueError):
            shared_article("article", "s" * 200, tokenizers, models)

    def test_chat_does_not_add_duplicate_bos(self):
        tok = Tokenizer()
        with patch.object(Tokenizer, "__call__", return_value={"input_ids": [1]}) as call:
            encode_pair(tok, {"input_format": "chat"}, "article", "summary")
            self.assertFalse(call.call_args.kwargs["add_special_tokens"])
            self.assertFalse(call.call_args.kwargs["truncation"])

    def test_article_context_is_identical_across_candidate_summaries(self):
        items, _ = fixture()
        items["article"] = items.article + "x" * 200
        items.loc[1, "summary"] = "A substantially longer summary than the other candidates"
        models = {"a": {"input_format": "chat", "max_length": 120},
                  "b": {"input_format": "pair", "max_length": 110}}
        result = pd.DataFrame(prepare.prepare_records(items, {k: Tokenizer() for k in models}, models))
        self.assertTrue(result.groupby("article_id").article.nunique().eq(1).all())
        self.assertTrue(result.tokens_a.le(120).all())
        self.assertTrue(result.tokens_b.le(110).all())
        self.assertEqual(result.summary.tolist(), items.summary.tolist())

    def test_armorm_uses_final_scalar_not_objective_vector(self):
        out = SimpleNamespace(score=FakeTensor([.3]), rewards=FakeTensor(np.ones(19)))
        self.assertAlmostEqual(scalar_score(out, "score"), .3)
        with self.assertRaises(ValueError):
            scalar_score(out, "rewards")
        with self.assertRaises(ValueError):
            scalar_score(SimpleNamespace(logits=FakeTensor([np.nan])), "logits")

    def test_splits_disjoint_reproducible_and_not_alias_dependent(self):
        ids = fixture()[0].item_id.tolist()
        rows = make_splits(ids, 3, .3, 0)
        self.assertEqual(rows, make_splits(ids, 3, .3, 0))
        self.assertNotEqual(rows, make_splits(ids, 3, .3, 1))
        for row in rows:
            self.assertFalse(set(row["calibration"]) & set(row["evaluation"]))
            self.assertEqual(set(row["calibration"] + row["evaluation"]), set(ids))

    def test_prepared_integrity_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data.csv").write_text("original")
            write_json(root / "source_validation.json", dict(status="PASS", n_verified_summaries=40))
            write_json(root / "manifest.json", {"schema_version": 2, "n_items": 40,
                       "files": {n: sha256(root / n) for n in ["data.csv", "source_validation.json"]}})
            verify_prepared(root)
            (root / "data.csv").write_text("changed")
            with self.assertRaises(ValueError):
                verify_prepared(root)
            with self.assertRaises(FileExistsError):
                new_directory(root)
            with self.assertRaises(FileExistsError):
                write_json(root / "manifest.json", {})

    def test_raw_source_catches_within_article_summary_permutation(self):
        items, human = fixture()
        items, human = validate_inputs(items, human, 40)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source.jsonl"
            aligned_fixture(p, items, human)
            verify_aligned_source(items, human, p)
            wrong = items.copy()
            wrong.loc[[0, 1], "summary"] = wrong.loc[[1, 0], "summary"].to_numpy()
            with self.assertRaisesRegex(ValueError, "Summary ownership"):
                verify_aligned_source(wrong, human, p)
            wrong_human = human.copy()
            wrong_human.loc[0, "score"] = 5
            with self.assertRaisesRegex(ValueError, "Human rating"):
                verify_aligned_source(items, wrong_human, p)


class Mathematics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.r = rng.normal(size=80)
        self.j = 3 + .3 * self.r - .1 * self.r ** 2
        self.articles = np.repeat(np.arange(20), 4)

    def test_ab_formula_and_positive_affine_invariance(self):
        r, j = self.r, self.j
        z, jc = (r - r.mean()) / r.std(), j - j.mean()
        got, shifted = describe(j, r), describe(j, 2.5 * r + 10)
        self.assertAlmostEqual(got["A"], np.sqrt(2) * np.mean(jc * z))
        self.assertAlmostEqual(got["B"], np.mean(jc * z ** 2) - 2 / 3 * np.mean(jc * z) * np.mean(z ** 3))
        for name in ["A", "B", "B_tail", "B_skewness"]:
            self.assertAlmostEqual(got[name], shifted[name])
        with self.assertRaises(ValueError):
            describe(j, np.zeros_like(r))

    def test_paired_article_bootstrap(self):
        proxies = {"a": self.r, "b": self.r.copy(), "c": self.r.copy()}
        one, _ = bootstrap_coefficients(self.j, proxies, self.articles, 10, 3)
        two, _ = bootstrap_coefficients(self.j, proxies, self.articles, 10, 3)
        pd.testing.assert_frame_equal(one, two)
        self.assertEqual(len(one), 30)
        np.testing.assert_allclose(one[one.alias == "a"].A, one[one.alias == "b"].A)

    def test_evaluation_gold_not_used_by_predictor(self):
        path = PathData("test", "test", np.arange(80), self.r, self.j, {})
        changed = PathData("test", "test", np.arange(80), self.r, self.j.copy(), {})
        changed.gold_scores[30:] += .5 * self.r[30:]
        kwargs = dict(cal_idx=np.arange(30), eva_idx=np.arange(30, 80), fit_max_d=.1, d_targets=[.05, .1])
        first = prospective.evaluate_one_split(path, **kwargs)
        second = prospective.evaluate_one_split(changed, **kwargs)
        for key in first:
            if key.startswith(("cal_", "pred_theory_", "pred_aonly_", "pred_empfit_", "pred_anu_")):
                self.assertEqual(first[key], second[key])
        self.assertNotAlmostEqual(first["actual_eval_delta_0p05"], second["actual_eval_delta_0p05"])

    def test_common_support_drops_failed_split_for_every_model(self):
        ids = [str(i) for i in range(80)]
        paths = {m: {d: PathData(d, d, np.array(ids), self.r.copy(), self.j, {}) for d in DIMENSIONS}
                 for m in ["a", "b", "c"]}
        paths["c"]["coherence"].proxy_scores[:] = 0
        rows, accounting = evaluate_matched(paths, make_splits(ids, 1, .3, 0), [.05, .1])
        self.assertEqual(set(rows.dimension), set(DIMENSIONS) - {"coherence"})
        self.assertEqual(len(rows), 9)
        self.assertEqual(set(accounting[0]["reasons"]), {"c"})
        summary = summarize(rows, [.05, .1])
        self.assertTrue(summary.n_common_splits.eq(1).all())


class EndToEnd(unittest.TestCase):
    def test_synthetic_preparation_scored_artifacts_and_repeatable_analysis(self):
        # Exercise real file schemas and all saved hashes without HF or GPU access.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            items, human = fixture(1600)
            items["article"] = items.article + "x" * 800
            items["summary"] = [s + "y" * (i % 4 * 10) for i, s in enumerate(items.summary)]
            items.to_csv(root / "source_items.csv", index=False)
            human.to_csv(root / "source_human.csv", index=False)
            aligned_fixture(root / "aligned.jsonl", items, human)
            auto_tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
            argv = ["prepare.py", "--items-csv", str(root / "source_items.csv"),
                    "--human-csv", str(root / "source_human.csv"), "--outdir", str(root / "prepared"),
                    "--aligned-jsonl", str(root / "aligned.jsonl"),
                    "--repeats", "1"]
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=auto_tokenizer)}), patch.object(sys, "argv", argv):
                prepare.main()
            prepared = root / "prepared"
            frozen = pd.read_csv(prepared / "items.csv")
            models = json.loads((prepared / "models.json").read_text())
            for index, (alias, model) in enumerate(models.items()):
                folder = new_directory(root / "scores" / alias)
                scores = frozen[["item_id", "text_sha256"]].copy()
                scores["proxy_score"] = np.random.default_rng(index).normal(size=len(scores))
                scores.to_csv(folder / "scores.csv", index=False)
                write_json(folder / "run_spec.json", dict(alias=alias, model=model,
                           prepared_manifest_sha256=sha256(prepared / "manifest.json")))
                write_json(folder / "complete.json", dict(n_items=len(scores),
                           scores_sha256=sha256(folder / "scores.csv"), spec_sha256=sha256(folder / "run_spec.json")))
            for name in ["analysis_one", "analysis_two"]:
                argv = ["analyze.py", "--prepared", str(prepared), "--scores-root", str(root / "scores"),
                        "--outdir", str(root / name), "--n-boot", "2"]
                with patch.object(sys, "argv", argv):
                    analyze.main()
            for name in ["coefficients.csv", "coefficient_bootstrap.csv", "prospective_paths.csv",
                         "summary.csv", "split_accounting.json", "comparison.md", "run_spec.json"]:
                self.assertEqual((root / "analysis_one" / name).read_bytes(), (root / "analysis_two" / name).read_bytes())
            summary = pd.read_csv(root / "analysis_one/summary.csv")
            self.assertEqual(len(summary), 3 * 4 * 4)
            self.assertTrue(summary.n_common_splits.eq(1).all())


if __name__ == "__main__":
    unittest.main()
