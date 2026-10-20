import json
import tempfile
import unittest
from pathlib import Path

from lean_critic import author_protocol as a
from lean_critic import datasets as d


# Fixed prefixes copied from the two public templates; no problem or label data.
TEMPLATES = ('Role: Lean & Formal Verification Expert\n\nInput:\n- Mathematical_Text: A math problem and its answer (no proof).\n- Lean4Code: A Lean 4 theorem statement formalizing the problem. Proof is intentionally omitted (e.g., sorry).\n\nGoal:\nDetermine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.  \n**Do not evaluate or consider the answer or the proof. Your sole task is to verify the correctness of the formalization.**\n\nEvaluation Stages (All required):\n\n1. Mathematical Text Analysis  \n   Identify all structurally and semantically relevant components of the mathematical problem, including variables, types, quantifiers, constraints, logic structure, conclusion, and so on. The analysis should be based on the actual content of the text.\n\n2. Lean4 Code Analysis (ignore proof part)  \n   Extract all structurally and semantically relevant components from the Lean statement, including variables, types, conditions, quantifiers, constraints, the final claim, and so on. The analysis should reflect the actual content present in the Lean code.\n\n3. Comparative Analysis  \n   Check for exact correspondence between the math and Lean statements; you may refer to aspects like:\n   - Semantic alignment, logic structure, and quantifier correctness.\n   - Preservation of constraints and boundary assumptions.\n   - Accurate typing and use of variables.\n   - Strict adherence to Lean\'s specific syntactic and semantic rules in interpreting the Lean code.\n   - Syntactic validity and proper Lean usage (free from errors).\n   - Use of symbols and constructs without semantic drift.\n   - No missing elements, no unjustified additions, and no automatic corrections or completions.\n\n4. Accuracy Confirmation  \n   If correct: clearly confirm why all elements match.  \n   If incorrect: list all mismatches and explain how each one affects correctness.\n\nNote: While the analysis may be broad and open to interpreting all relevant features, the final judgment must be based only on what is explicitly and formally expressed in the Lean statement.  \n**Do not consider or assess any part of the proof. Your judgment should be entirely about the accuracy of the statement formalization.**\n\nOutput Format:\nReturn exactly one JSON object:\n```json\n{\n    "reasons": "\\\\n1. Mathematical Text Analysis: [...]\\\\n2.  Lean4 Code Analysis (: [...]\\\\n3. Comparative Analysis: [...]\\\\n4. Accuracy Confirmation: [...match confirmation or list of discrepancies...]",\n    "is_assistant_correct": "[Correct/Incorrect]"\n}\n```\n\n', 'You are an expert in Lean 4 and formal methods. Your task is to evaluate whether the assistant has successfully translated a mathematical problem and its answer into correct and meaningful Lean 4 *definitions and theorem statements*.\nIn your evaluation, you should consider whether the assistant\'s formalization accurately captures the mathematical content and structure *excluding the proofs*. The assessment should focus on whether the Lean code (definitions and statements only):\nRepresents the math problem faithfully and meaningfully.\nExpresses the intended concepts in a coherent and structurally appropriate way\n(e.g., the syntax, type usage, and formulation of definitions or theorems follow Lean’s conventions and make the math meaningfully formal).\nDemonstrates clarity, internal consistency, and alignment with common Lean 4 style.\nMaps the mathematical ideas correctly into Lean constructs\n(e.g., sets, quantifiers, functions, logical operations).\nIs reasonably organized and maintainable.\n\n**Note:**\n    Your evaluation must be based *only* on whether the structure and content of the formalization (definitions, theorem statements) correctly reflect the original mathematical intent.\n   **You must completely ignore the proofs.** You are **not** required to verify the proof itself. The use of `sorry`, `by sorry`, `admit`, any placeholder tactic, or incomplete proof scripts is acceptable and **should not be considered an error**. The correctness, presence, or absence of a proof is **irrelevant** for this task.\n     You must return your evaluation strictly in the following JSON format. Other formats will not be accepted.\n```json\n{\n    "reasons": "Explain whether the Lean code\'s definitions and statements correctly capture the mathematical intent and structure, noting any major problems or successes in the formalization (ignoring proofs).",\n    "is_assistant_correct": "[Correct/Incorrect]"\n}\n```\n\nInput Data:\n')


class AuthorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "raw.jsonl"
        self.out = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def raw(self, template=0, source="For every n, n = n.", lean="-- keep this comment\ntheorem t (n : Nat) : n = n := by rfl"):
        content = (TEMPLATES[template] + "— Start of Mathematical_Text —\n" + source
                   + "\n— End of Mathematical_Text —\n\n— Start of Lean4Code —\n" + lean
                   + "\n— End of Lean4Code —")
        return {"refined_statement": source, "autoformalization": lean,
                "messages": [{"role": "user", "content": content}],
                "tag": "compile_right_is_false", "id": 991,
                "source": "PRIVATE_PROVENANCE"}

    def prepare(self, rows):
        d.write_jsonl(self.input, rows)
        return a._prepare_original_rows(rows, self.out, test_fixture=True)

    def cases(self):
        return d.read_jsonl(self.out / "cases.jsonl")[0]

    def prompts(self):
        return d.read_jsonl(self.out / a.AUTHOR_PROMPTS_FILE)[0]

    def rewrite_prompts(self, rows):
        d.write_jsonl(self.out / a.AUTHOR_PROMPTS_FILE, rows)

    def test_public_entry_point_rejects_small_or_custom_benchmarks(self):
        d.write_jsonl(self.input, [self.raw()])
        with self.assertRaisesRegex(ValueError, "complete published 500"):
            a.prepare_original(self.input, self.out)
        self.assertFalse(self.out.exists())

    def test_fixture_helper_is_explicitly_marked(self):
        manifest = self.prepare([self.raw()])
        self.assertTrue(manifest["fixture_only"])
        self.assertEqual(manifest["count"], 1)

    def test_both_public_templates_preserved_with_private_labels_excluded(self):
        rows = [self.raw(0), self.raw(1, "Another statement.", "-- comment\ntheorem s : True := by trivial")]
        d.write_jsonl(self.input, rows)
        original = self.input.read_bytes()
        manifest = a._prepare_original_rows(rows, self.out, test_fixture=True)
        self.assertEqual(self.input.read_bytes(), original)
        loaded = a.load_original_prompts(self.cases(), self.out / a.AUTHOR_PROMPTS_FILE)
        self.assertEqual({v[0]["content"] for v in loaded.values()},
                         {r["messages"][0]["content"] for r in rows})
        self.assertEqual(manifest["author_prompt_template_counts"],
                         {"evaluation_stages": 1, "definitions_and_statements": 1})
        self.assertEqual(manifest["author_prompts_file_sha256"], d.file_hash(self.out / a.AUTHOR_PROMPTS_FILE))
        self.assertEqual(manifest, json.loads((self.out / "manifest.json").read_text()))
        for case in self.cases():
            self.assertEqual(set(case), {"case_id", "source_nl", "lean_code"})
            self.assertIn("--", case["lean_code"])
        for item in self.prompts():
            self.assertEqual(set(item), {"case_id", "source_sha256", "candidate_sha256", "messages"})
            self.assertNotIn("PRIVATE_", item["messages"][0]["content"])
        self.assertEqual(d.read_jsonl(self.out / "gold.jsonl")[0][0]["expected_verdict"], "contradicted")

    def test_actual_input_whitespace_and_comments_are_not_stripped(self):
        row = self.raw(source="\n  Statement with whitespace.  \n", lean="\n-- comment\n theorem t : True := by trivial  \n")
        self.prepare([row])
        case = self.cases()[0]
        self.assertEqual(case["source_nl"], row["refined_statement"])
        self.assertEqual(case["lean_code"], row["autoformalization"])
        actual = a.load_original_prompts([case], self.out / a.AUTHOR_PROMPTS_FILE)
        self.assertEqual(actual[case["case_id"]], row["messages"])

    def test_refuses_nonempty_output_without_overwriting(self):
        self.out.mkdir()
        sentinel = self.out / "manifest.json"
        sentinel.write_text("existing run")
        d.write_jsonl(self.input, [self.raw()])
        with self.assertRaisesRegex(ValueError, "new or empty"):
            a._prepare_original_rows([self.raw()], self.out, test_fixture=True)
        self.assertEqual(sentinel.read_text(), "existing run")
        self.assertEqual(list(self.out.iterdir()), [sentinel])

    def test_only_one_plain_user_message_is_permitted(self):
        base = self.raw()["messages"][0]
        invalid = [[], [dict(base, role="system")], [base, base],
                   [dict(base, tag="compile_right_is_right")], [{"role": "user", "content": 12}],
                   [{"role": "user"}], {"role": "user", "content": "text"}]
        for messages in invalid:
            with self.subTest(messages=messages):
                row = self.raw()
                row["messages"] = messages
                d.write_jsonl(self.input, [row])
                with self.assertRaisesRegex(ValueError, "exactly one user"):
                    a._prepare_original_rows([row], self.out, test_fixture=True)
                self.assertFalse(self.out.exists())

    def test_raw_message_must_bind_exact_source_and_lean(self):
        for field in ("refined_statement", "autoformalization"):
            with self.subTest(field=field):
                row = self.raw()
                row[field] += "CHANGED"
                d.write_jsonl(self.input, [row])
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    a._prepare_original_rows([row], self.out, test_fixture=True)
                self.assertFalse(self.out.exists())

    def test_added_prefix_suffix_or_ambiguous_boundaries_are_rejected(self):
        content = self.raw()["messages"][0]["content"]
        invalid = ["Correct label\n" + content, content + "\nCorrect label",
                   content.replace("— Start of Lean4Code —", "— End of Lean4Code —"),
                   content + "— Start of Mathematical_Text —",
                   content.replace("— End of Mathematical_Text —\n\n", "— End of Mathematical_Text —\nCorrect label\n")]
        for bad in invalid:
            with self.subTest(content=bad[-70:]):
                row = self.raw()
                row["messages"][0]["content"] = bad
                d.write_jsonl(self.input, [row])
                with self.assertRaises(ValueError):
                    a._prepare_original_rows([row], self.out, test_fixture=True)
                self.assertFalse(self.out.exists())

    def test_whole_dataset_sidecar_can_serve_selected_subset(self):
        self.prepare([self.raw(0), self.raw(1, source="Different source.")])
        case = self.cases()[0]
        loaded = a.load_original_prompts([case], self.out / a.AUTHOR_PROMPTS_FILE)
        self.assertEqual(set(loaded), {case["case_id"]})
        self.assertEqual(len(loaded[case["case_id"]]), 1)
        self.assertEqual(loaded[case["case_id"]][0]["role"], "user")

    def test_selected_case_hash_mismatch_is_rejected(self):
        self.prepare([self.raw()])
        for field in ("source_nl", "lean_code"):
            with self.subTest(field=field):
                case = dict(self.cases()[0])
                case[field] += "CHANGED"
                with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                    a.load_original_prompts([case], self.out / a.AUTHOR_PROMPTS_FILE)

    def test_missing_selected_case_is_rejected(self):
        self.prepare([self.raw()])
        self.rewrite_prompts([])
        with self.assertRaisesRegex(ValueError, "is missing"):
            a.load_original_prompts(self.cases(), self.out / a.AUTHOR_PROMPTS_FILE)

    def test_duplicate_unselected_ids_are_rejected(self):
        self.prepare([self.raw()])
        rows = self.prompts()
        extra = dict(rows[0], case_id="not_selected")
        self.rewrite_prompts(rows + [extra, extra])
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            a.load_original_prompts(self.cases(), self.out / a.AUTHOR_PROMPTS_FILE)

    def test_unselected_messages_are_still_validated(self):
        self.prepare([self.raw()])
        rows = self.prompts()
        extra = dict(rows[0], case_id="not_selected", source_sha256="0" * 64)
        self.rewrite_prompts(rows + [extra])
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            a.load_original_prompts(self.cases(), self.out / a.AUTHOR_PROMPTS_FILE)

    def test_labels_or_other_extra_sidecar_fields_are_rejected(self):
        self.prepare([self.raw()])
        rows = self.prompts()
        rows[0]["tag"] = "compile_right_is_false"
        self.rewrite_prompts(rows)
        with self.assertRaisesRegex(ValueError, "only case_id"):
            a.load_original_prompts(self.cases(), self.out / a.AUTHOR_PROMPTS_FILE)


if __name__ == "__main__":
    unittest.main()
