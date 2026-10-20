import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lean_critic import author_protocol as a
from lean_critic import datasets as d
from lean_critic.storage import canonical
from test_author_protocol import TEMPLATES


def fixture_row(number=0, tag="compile_right_is_right"):
    source, lean = f"Statement {number}", f"-- preserved\ntheorem t{number} : True := by sorry"
    content = (TEMPLATES[number % 2] + "— Start of Mathematical_Text —\n" + source
               + "\n— End of Mathematical_Text —\n\n— Start of Lean4Code —\n" + lean
               + "\n— End of Lean4Code —")
    return {"refined_statement": source, "autoformalization": lean, "tag": tag,
            "id": number, "source": "Synthetic plumbing fixture",
            "messages": [{"role": "user", "content": content}]}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "raw.jsonl"
        self.out = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def full_fixture(self):
        tags = ["compile_right_is_right"] * 250 + ["compile_right_is_false"] * 200 + ["compile_false"] * 50
        return [fixture_row(i, tag) for i, tag in enumerate(tags)]

    def prepare_fixture(self, rows):
        d.write_jsonl(self.input, rows)
        with patch.object(d, "BENCHMARK_RECORDS_SHA256", d.sha256(canonical(rows))):
            return d.prepare(self.input, self.out)

    def test_full_benchmark_order_labels_private_sidecar_and_no_splits(self):
        rows = self.full_fixture()
        result = self.prepare_fixture(rows)
        cases, _ = d.read_jsonl(self.out / "cases.jsonl")
        gold, _ = d.read_jsonl(self.out / "gold.jsonl")
        self.assertEqual([r["source_nl"] for r in cases], [r["refined_statement"] for r in rows])
        self.assertEqual(set(cases[0]), d.CASE_FIELDS)
        self.assertEqual(result["author_tag_counts"], d.TAG_COUNTS)
        self.assertEqual(result["labeled_count"], 500)
        self.assertNotIn("split_counts", result)
        self.assertNotIn("split", gold[0])
        self.assertEqual(sum(r["expected_verdict"] == "supported" for r in gold), 250)
        self.assertEqual(sum(r["expected_verdict"] == "contradicted" for r in gold), 250)
        self.assertEqual([r["published_row_index"] for r in gold], list(range(500)))
        self.assertEqual(stat.S_IMODE((self.out / "gold.jsonl").stat().st_mode), 0o600)
        self.assertNotIn("Synthetic plumbing fixture", (self.out / "cases.jsonl").read_text())

    def test_rejects_custom_aliases_missing_tags_and_noncomplete_benchmarks(self):
        invalid = [[{"nl": "A", "lean": "L", "label": True}], [fixture_row()]]
        missing_tag = fixture_row()
        del missing_tag["tag"]
        invalid.append([missing_tag])
        for rows in invalid:
            d.write_jsonl(self.input, rows)
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                d.prepare(self.input, self.out)
            self.assertFalse(self.out.exists())

    def test_rejects_modified_records_and_order_even_with_same_tag_counts(self):
        rows = self.full_fixture()
        pinned = d.sha256(canonical(rows))
        for mutated in [list(reversed(rows)), [dict(rows[0], refined_statement="Changed")] + rows[1:]]:
            d.write_jsonl(self.input, mutated)
            with patch.object(d, "BENCHMARK_RECORDS_SHA256", pinned), self.assertRaisesRegex(ValueError, "pinned CriticLeanBench"):
                d.prepare(self.input, self.out)
        self.assertFalse(self.out.exists())

    def test_no_fetch_during_prepare_and_refuses_overwrite(self):
        rows = self.full_fixture()
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            self.prepare_fixture(rows)
        sentinel = (self.out / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.prepare_fixture(rows)
        self.assertEqual((self.out / "manifest.json").read_bytes(), sentinel)

    def test_first_n_selection_preserves_original_messages_and_labels(self):
        rows = [fixture_row(i) for i in range(8)]
        a._prepare_original_rows(rows, self.out, test_fixture=True)
        chosen = self.root / "chosen"
        result = d.select(self.out, chosen, limit=5)
        cases = d.read_jsonl(chosen / "cases.jsonl")[0]
        self.assertEqual([r["source_nl"] for r in cases], [r["refined_statement"] for r in rows[:5]])
        prompts = a.load_original_prompts(cases, chosen / a.AUTHOR_PROMPTS_FILE)
        self.assertEqual([prompts[r["case_id"]] for r in cases], [r["messages"] for r in rows[:5]])
        self.assertFalse(result["selection_uses_labels"])
        self.assertNotIn("split", result)
        with self.assertRaisesRegex(ValueError, "new or empty"):
            d.select(self.out, chosen, limit=5)

    def test_selection_rejects_stale_gold_and_invalid_limits(self):
        a._prepare_original_rows([fixture_row()], self.out, test_fixture=True)
        for limit in [0, -1, True]:
            with self.assertRaisesRegex(ValueError, "positive"):
                d.select(self.out, self.root / "bad", limit=limit)
        gold = d.read_jsonl(self.out / "gold.jsonl")[0]
        gold[0]["candidate_sha256"] = "0" * 64
        d.write_jsonl(self.out / "gold.jsonl", gold)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.select(self.out, self.root / "bad", limit=1)

    def test_duplicate_keys_and_source_provenance_rejected(self):
        self.input.write_text('{"id":"A","id":"B"}\n')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            d.prepare(self.input, self.out)
        rows = self.full_fixture()
        d.write_json(self.root / "source-manifest.json", {"jsonl_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "provenance JSONL hash"):
            self.prepare_fixture(rows)

    def test_pinned_fingerprint_matches_canonical_definition(self):
        self.assertEqual(len(d.BENCHMARK_RECORDS_SHA256), 64)
        self.assertEqual(d.TAG_COUNTS, {"compile_right_is_right": 250, "compile_right_is_false": 200, "compile_false": 50})
        self.assertEqual(d.REVISION, "6225522bce304f62d4907aedb9642255f9417bf0")


if __name__ == "__main__":
    unittest.main()
