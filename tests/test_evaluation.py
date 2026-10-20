import csv
import tempfile
import unittest
from pathlib import Path

from lean_critic import author_protocol as a
from lean_critic import datasets as d
from lean_critic.evaluation import _block, evaluate
from test_datasets import fixture_row


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        rows = [fixture_row(0, "compile_right_is_right"), fixture_row(1, "compile_right_is_false"),
                fixture_row(2, "compile_false")]
        a._prepare_original_rows(rows, self.prepared, test_fixture=True)
        self.cases = d.read_jsonl(self.prepared / "cases.jsonl")[0]
        self.pred = self.root / "predictions.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def predictions(self, verdicts):
        return [{"case_id": row["case_id"], "source_sha256": d.sha256(row["source_nl"]),
                 "candidate_sha256": d.sha256(row["lean_code"]), "verdict": verdict,
                 "is_assistant_correct": {"supported": "Correct", "contradicted": "Incorrect", "unknown": None}[verdict],
                 "execution_errors": ["api_failure"] if verdict == "unknown" else []}
                for row, verdict in zip(self.cases, verdicts)]

    def evaluate(self):
        return evaluate(self.prepared / "cases.jsonl", self.prepared / "gold.jsonl", self.pred, self.root / "results")

    def test_labels_metrics_denominators_and_compile_subset(self):
        d.write_jsonl(self.pred, self.predictions(["contradicted", "supported", "unknown"]))
        result = self.evaluate()
        metrics = result["overall"]["classification_metrics"]
        self.assertEqual(metrics["ACC"], {"numerator": 0, "denominator": 3, "value": 0.0})
        self.assertEqual(metrics["FNR"], {"numerator": 1, "denominator": 1, "value": 1.0})
        self.assertEqual(metrics["FPR"], {"numerator": 1, "denominator": 2, "value": 0.5})
        self.assertEqual(metrics["unknown_negative_rate"]["value"], 0.5)
        self.assertEqual(result["author_compile_passing_subset"]["n"], 2)
        self.assertEqual(result["technical_failures"]["case_count"], 1)
        self.assertEqual(result["technical_failures"]["by_reason"], {"api_failure": 1})
        self.assertFalse(result["exact_published_scorer_reproduction"])
        self.assertFalse(result["author_compile_tags_freshly_verified"])

    def test_no_mathematical_or_uncertainty_gates_change_model_labels(self):
        rows = self.predictions(["supported", "contradicted", "contradicted"])
        for row in rows:
            row.update({"reasons": "Unverified reason", "unresolved": ["A question"],
                        "findings": [{"invalid_quote": "Missing"}], "gate_errors": ["legacy_gate"]})
        d.write_jsonl(self.pred, rows)
        original = self.pred.read_bytes()
        result = self.evaluate()
        self.assertEqual(result["overall"]["classification_metrics"]["ACC"]["value"], 1)
        self.assertEqual(result["technical_failures"]["case_count"], 0)
        self.assertEqual(self.pred.read_bytes(), original)

    def test_technical_status_does_not_override_valid_model_label(self):
        rows = self.predictions(["supported", "contradicted", "contradicted"])
        rows[0]["execution_errors"] = ["recorder_warning"]
        d.write_jsonl(self.pred, rows)
        result = self.evaluate()
        self.assertEqual(result["overall"]["correct"], 3)
        self.assertEqual(result["technical_failures"]["case_count"], 1)

    def test_missing_invalid_and_malformed_predictions_remain_in_acc_denominator(self):
        rows = self.predictions(["supported", "contradicted", "contradicted"])
        rows.pop()
        rows[1]["verdict"] = ["supported"]
        d.write_jsonl(self.pred, rows + [{"verdict": "supported"}])
        with self.pred.open("a") as handle:
            handle.write("{broken\n")
        result = self.evaluate()
        self.assertEqual(result["overall"]["unknown"], 2)
        self.assertEqual(result["overall"]["classification_metrics"]["ACC"], {"numerator": 1, "denominator": 3, "value": 1 / 3})
        self.assertEqual(result["technical_failures"]["case_count"], 2)

    def test_duplicate_or_stale_predictions_refuse_scoring(self):
        rows = self.predictions(["supported", "contradicted", "contradicted"])
        d.write_jsonl(self.pred, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.evaluate()
        rows[0]["candidate_sha256"] = "wrong"
        d.write_jsonl(self.pred, rows)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.evaluate()

    def test_conflicting_recorded_labels_refuse_scoring(self):
        rows = self.predictions(["supported", "contradicted", "contradicted"])
        rows[0]["is_assistant_correct"] = "Incorrect"
        d.write_jsonl(self.pred, rows)
        with self.assertRaisesRegex(ValueError, "label and verdict disagree"):
            self.evaluate()

    def test_report_csv_preserves_public_order_and_explains_small_test(self):
        d.write_jsonl(self.pred, self.predictions(["supported", "contradicted", "unknown"]))
        result = self.evaluate()
        self.assertEqual(result["cases_sha256"], d.file_hash(self.prepared / "cases.jsonl"))
        with (self.root / "results/per-case.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["case_id"] for row in rows], [row["case_id"] for row in self.cases])
        self.assertEqual([row["model_label"] for row in rows], ["Correct", "Incorrect", ""])
        self.assertIn("连通性测试", (self.root / "results/summary.md").read_text())
        self.assertTrue((self.root / "results/diagnostics.jsonl").is_file())

    def test_class_metrics_unknowns_are_neither_false_positives_nor_false_negatives(self):
        gold = {name: {"expected_verdict": label} for name, label in [
            ("tp", "supported"), ("fn", "supported"), ("up", "supported"),
            ("tn", "contradicted"), ("fp", "contradicted"), ("un", "contradicted")]}
        decisions = {"tp": "supported", "fn": "contradicted", "up": "unknown",
                     "tn": "contradicted", "fp": "supported", "un": "unknown"}
        metrics = _block(list(gold), gold, decisions)["classification_metrics"]
        self.assertEqual(metrics["ACC"]["value"], 1 / 3)
        for key in ("TPR", "FPR", "TNR", "FNR", "unknown_positive_rate", "unknown_negative_rate"):
            self.assertEqual(metrics[key], {"numerator": 1, "denominator": 3, "value": 1 / 3})
        empty = _block([], {}, {})["classification_metrics"]
        for key in ("ACC", "TPR", "FPR", "TNR", "FNR"):
            self.assertIsNone(empty[key]["value"])


if __name__ == "__main__":
    unittest.main()
