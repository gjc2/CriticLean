"""Offline runner checks; published-template validation has its own tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lean_critic import engine, providers
from lean_critic.storage import parse, read_jsonl, write_json, write_jsonl


class LabelParsingTests(unittest.TestCase):
    def test_json_fence_and_explicit_field_are_accepted(self):
        samples = [('{"is_assistant_correct":"Correct","reasons":"x"}', "json"),
                   ('```json\n{"is_assistant_correct":"Incorrect"}\n```', "fenced_json"),
                   ('Result: {"is_assistant_correct":"Correct",}', "explicit_field_regex")]
        for text, mode in samples:
            with self.subTest(mode=mode):
                output, parser = engine.parse_original_response(text)
                self.assertIn(output["is_assistant_correct"], ("Correct", "Incorrect"))
                self.assertEqual(parser, mode)

    def test_ambiguous_duplicate_and_guessed_labels_are_rejected(self):
        for text in ["Correct", '{"verdict":"Correct"}',
                     '{"is_assistant_correct":"correct"}',
                     '{"is_assistant_correct":[]}',
                     '{"is_assistant_correct":"Correct","is_assistant_correct":"Incorrect"}',
                     '{"is_assistant_correct":"Correct","reasons":"x","reasons":"y"}',
                     '{"is_assistant_correct":"Correct","other":{"is_assistant_correct":"Incorrect"}}',
                     'A: {"is_assistant_correct":"Correct"} B: {"is_assistant_correct":"Incorrect"}',
                     '{"is_assistant_correct":"Correct","is_assistant_correct":bad,}',
                     '{"is_assistant_correct":"Correct","is_assistant_\\u0063orrect":"Incorrect"}',
                     '{"is_assistant_correct":"Correct","is_assistant_\\u0063orrect":"Incorrect",}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                engine.parse_original_response(text)

    def test_reasons_do_not_change_or_block_label(self):
        text = ('{"is_assistant_correct":"Incorrect","reasons":"I am uncertain; quote unavailable.",'
                '"unresolved":["missing definition"],"findings":[]}')
        output, _ = engine.parse_original_response(text)
        self.assertEqual(output["is_assistant_correct"], "Incorrect")
        output, _ = engine.parse_original_response('{"is_assistant_correct":"Correct","reasons":null}')
        self.assertIsNone(output["reasons"])

    def test_invalid_reason_escape_before_label_is_opaque(self):
        text = r'{"reasons":"The bound is \frac{1}{2} and \alpha is positive.","is_assistant_correct":"Correct"}'
        output, parser = engine.parse_original_response(text)
        self.assertEqual(output["is_assistant_correct"], "Correct")
        self.assertEqual(parser, "explicit_field_regex")
        self.assertEqual(output["reasons"], text)

    def test_malformed_quoted_reason_does_not_inject_fake_label(self):
        reason = r'{"reasons":"\alpha; example: \"is_assistant_correct\":\"Incorrect\""'
        output, parser = engine.parse_original_response(reason + ',"is_assistant_correct":"Correct"}')
        self.assertEqual(output["is_assistant_correct"], "Correct")
        self.assertEqual(parser, "explicit_field_regex")
        with self.assertRaisesRegex(ValueError, "missing_or_ambiguous"):
            engine.parse_original_response(reason + '}')

    def test_invalid_reason_does_not_hide_actual_duplicate_labels(self):
        prefix = r'{"reasons":"\alpha","is_assistant_correct":"Correct",'
        for label in ('Correct', 'Incorrect'):
            for key in ('is_assistant_correct', r'is_assistant_\u0063orrect'):
                with self.subTest(label=label, key=key), self.assertRaisesRegex(ValueError, "missing_or_ambiguous"):
                    engine.parse_original_response(prefix + f'"{key}":"{label}"' + '}')


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = {"case_id": "case_test", "source_nl": "For every natural number n, n + 0 = n.",
                     "lean_code": "-- original comment retained\ntheorem t (n : Nat) : n + 0 = n := by sorry"}
        self.messages = {"case_test": [{"role": "user", "content": "PUBLISHED PROMPT\n" + self.case["source_nl"] + "\n" + self.case["lean_code"]}]}
        write_jsonl(self.root / "cases.jsonl", [self.case])
        write_jsonl(self.root / "author-prompts.jsonl", [{"fixture": "verified separately"}])
        self.config = {"name": "fixture", "method": "critic_original", "author_prompts": "author-prompts.jsonl",
                       "provider": {"type": "deepseek", "model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                                    "api_key_env": "DEEPSEEK_API_KEY", "structured_output": False}}
        self.config_path = self.root / "config.json"
        self.out = self.root / "run"
        self.tasks = []
        self.loader = patch.object(engine.author_protocol, "load_original_prompts", return_value=self.messages).start()
        self.addCleanup(patch.stopall)

    def dispatch(self, config, task, **kwargs):
        self.tasks.append(copy.deepcopy(task))
        # Confirm the dispatch record exists before the remote operation starts.
        path = self.out / "traces" / f"{task['request_id']}.json"
        self.assertEqual(parse(path.read_text())["status"], "dispatched")
        budget = {**engine.DEFAULT_BUDGET, **self.config.get("budget", {})}
        self.assertEqual(kwargs["max_output_tokens"], budget["max_output_tokens_per_call"])
        self.assertEqual(kwargs["timeout_s"], engine._request_timeout(budget))
        return providers.ProviderResponse('{"is_assistant_correct":"Correct","reasons":"same statement"}',
                    {"input_tokens": 12, "output_tokens": 7, "reasoning_output_tokens": 4},
                    {"status": "completed", "model": "deepseek-flash", "response_id": "fixture_response"})

    def run_case(self, *, dispatch=None, resume=False):
        write_json(self.config_path, self.config)
        return engine.run(self.root / "cases.jsonl", self.config_path, self.out,
                          resume=resume, dispatch=dispatch or self.dispatch)

    def predictions(self):
        return read_jsonl(self.out / "predictions.jsonl")

    def test_one_verbatim_request_only_and_no_added_instructions(self):
        result = self.run_case()
        self.assertEqual(len(self.tasks), 1)
        task = self.tasks[0]
        self.assertEqual(set(task), {"request_id", "case_id", "stage", "messages"})
        self.assertEqual(task["messages"], self.messages["case_test"])
        self.assertIn("-- original comment retained", task["messages"][0]["content"])
        self.assertEqual(result["usage"]["requests"], 1)
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertIsNone(result["usage"]["cost_usd"])
        self.assertEqual(result["valid_label_cases"], 1)
        self.assertEqual(result["technical_failure_cases"], 0)
        self.assertEqual(result["label_status"], "complete")
        self.assertIsNone(result["client_timeout_seconds"])
        self.assertIsNone(result["max_wall_seconds_per_case"])
        self.assertEqual(result["requested_output_token_cap"], 393216)
        self.assertEqual(result["server_output_token_max"], 393216)
        record = self.predictions()[0]
        self.assertEqual(record["verdict"], "supported")
        self.assertEqual(record["is_assistant_correct"], "Correct")
        self.assertEqual(record["reasons"], "same statement")
        self.assertEqual(record["execution_errors"], [])
        self.assertNotIn("gate_errors", record)
        self.assertNotIn("evidence_gate_applied", record)
        self.assertEqual(parse((self.out / "author-messages.json").read_text()), self.messages)
        trace = parse(next((self.out / "traces").glob("*.json")).read_text())
        self.assertEqual(trace["status"], "parsed")
        self.assertEqual(trace["metadata"]["response_id"], "fixture_response")
        self.assertIn('"is_assistant_correct":"Correct"', trace["response_text"])
        self.assertIsNone(trace["client_timeout_seconds"])

    def test_model_label_retained_regardless_of_reasons_extra_fields(self):
        def answer(*args, **kwargs):
            return providers.ProviderResponse('{"is_assistant_correct":"Incorrect","reasons":null,'
                    '"unresolved":["unknown definition"],"findings":[{"quote":"not in original"}]}', {}, {})
        self.run_case(dispatch=answer)
        record = self.predictions()[0]
        self.assertEqual(record["verdict"], "contradicted")
        self.assertIsNone(record["reasons"])
        self.assertEqual(record["execution_status"], "model_label_recorded")

    def test_technical_failures_remain_separate_without_retry(self):
        calls = []
        def failure(*args, **kwargs):
            calls.append(1)
            raise providers.ProviderError("fixture_network_failure")
        result = self.run_case(dispatch=failure)
        self.assertEqual(calls, [1])
        self.assertEqual(result["usage"]["requests"], 1)
        self.assertIsNone(result["usage"]["cost_usd"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["label_status"], "incomplete")
        self.assertEqual(result["valid_label_cases"], 0)
        self.assertEqual(result["technical_failure_cases"], 1)
        record = self.predictions()[0]
        self.assertEqual(record["verdict"], "unknown")
        self.assertIsNone(record["is_assistant_correct"])
        self.assertEqual(record["execution_status"], "technical_failure")
        self.assertEqual(record["execution_errors"], ["fixture_network_failure"])
        self.run_case(dispatch=failure, resume=True)
        self.assertEqual(calls, [1])

    def test_malformed_response_retained_for_inspection(self):
        def response(*args, **kwargs):
            return providers.ProviderResponse("No explicit label here", {"input_tokens": 2}, {})
        self.run_case(dispatch=response)
        self.assertEqual(self.predictions()[0]["verdict"], "unknown")
        trace = parse(next((self.out / "traces").glob("*.json")).read_text())
        self.assertEqual(trace["response_text"], "No explicit label here")
        self.assertEqual(trace["usage"]["input_tokens"], 2)

    def test_truncated_response_is_technical_failure(self):
        def response(*args, **kwargs):
            return providers.ProviderResponse('{"is_assistant_correct":"Correct"}', {}, {"truncated": True})
        self.run_case(dispatch=response)
        self.assertEqual(self.predictions()[0]["execution_errors"], ["provider_incomplete_or_refused"])

    def test_input_budget_prevents_dispatch(self):
        self.config["budget"] = {"max_input_chars_per_case": 5}
        result = self.run_case()
        self.assertEqual(self.tasks, [])
        self.assertEqual(result["usage"]["requests"], 0)
        self.assertEqual(self.predictions()[0]["execution_errors"], ["input_character_budget_exhausted"])

    def test_only_one_method_and_attempt_allowed_before_output(self):
        for modification in [{"method": "direct"}, {"stages": {}}, {"decision_policy": "model"},
                             {"budget": {"max_attempts": 2}}, {"budget": {"max_calls_per_case": 2}}]:
            with self.subTest(modification=modification):
                path = self.root / "invalid.json"
                write_json(path, {**self.config, **modification})
                with self.assertRaises(ValueError):
                    engine.run(self.root / "cases.jsonl", path, self.out, dispatch=self.dispatch)
                self.assertFalse(self.out.exists())

    def test_default_maximum_and_nullable_local_timeouts_are_explicit_in_plan(self):
        write_json(self.config_path, self.config)
        config = engine.load_config(self.config_path)
        self.assertEqual(config["budget"]["max_output_tokens_per_call"], 393216)
        self.assertIsNone(config["budget"]["request_timeout_seconds"])
        self.assertIsNone(config["budget"]["max_wall_seconds_per_case"])
        planned = engine.plan(self.root / "cases.jsonl", self.config_path)
        self.assertEqual(planned["reserved_output_token_ceiling_per_case"], 393216)
        self.assertEqual(planned["server_output_token_max"], 393216)
        self.assertIsNone(planned["request_timeout_seconds"])
        self.assertIsNone(planned["client_timeout_seconds"])
        self.assertIsNone(planned["max_wall_seconds_per_case"])
        self.assertEqual(self.tasks, [])

    def test_old_bounded_budget_preserves_smallest_explicit_timeout(self):
        self.config["budget"] = {"max_output_tokens_per_call": 8192,
                                 "request_timeout_seconds": 180, "max_wall_seconds_per_case": 900}
        result = self.run_case()
        self.assertEqual(result["client_timeout_seconds"], 180)
        self.assertEqual(result["requested_output_token_cap"], 8192)
        self.assertEqual(result["max_wall_seconds_per_case"], 900)
        for request, wall, expected in ((None, 3.5, 3.5), (8, None, 8), (9, 4, 4), (None, None, None)):
            with self.subTest(request=request, wall=wall):
                self.config["budget"].update(request_timeout_seconds=request, max_wall_seconds_per_case=wall)
                write_json(self.config_path, self.config)
                planned = engine.plan(self.root / "cases.jsonl", self.config_path)
                self.assertEqual(planned["client_timeout_seconds"], expected)

    def test_invalid_budget_values_and_above_server_cap_are_rejected(self):
        invalid = [(key, value) for key in engine.DEFAULT_BUDGET for value in (True, False, 0, -1, "10")]
        invalid += [(key, None) for key in engine.DEFAULT_BUDGET if key not in engine._TIMEOUT_KEYS]
        invalid += [(key, 1.5) for key in engine.DEFAULT_BUDGET if key not in engine._TIMEOUT_KEYS]
        invalid.append(("max_output_tokens_per_call", 393217))
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                self.config["budget"] = {key: value}
                write_json(self.config_path, self.config)
                with self.assertRaises(ValueError):
                    engine.load_config(self.config_path)
        for key in engine._TIMEOUT_KEYS:
            for value in (float("inf"), float("-inf"), float("nan")):
                with self.subTest(key=key, value=value):
                    self.config["budget"] = {key: value}
                    self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        engine.load_config(self.config_path)

    def test_invalid_provider_rejected_before_artifacts(self):
        self.config["provider"]["api_key"] = "do-not-save"
        with self.assertRaises((providers.ProviderError, ValueError)):
            self.run_case()
        self.assertFalse(self.out.exists())

    def test_resume_skips_completed_and_refuses_nonempty_new_run(self):
        self.run_case()
        self.run_case(resume=True)
        self.assertEqual(len(self.tasks), 1)
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_case()

    def test_resume_rejects_changed_inputs_config_code_and_prompts(self):
        self.run_case()
        write_json(self.config_path, self.config)
        with patch.object(engine, "implementation_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "changed"):
                engine.run(self.root / "cases.jsonl", self.config_path, self.out, resume=True)
        self.case["lean_code"] += "\n-- changed"
        write_jsonl(self.root / "cases.jsonl", [self.case])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_case(resume=True)
        self.case["lean_code"] = self.case["lean_code"].removesuffix("\n-- changed")
        write_jsonl(self.root / "cases.jsonl", [self.case])
        self.config["name"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_case(resume=True)
        self.config["name"] = "fixture"
        write_jsonl(self.root / "author-prompts.jsonl", [{"fixture": "changed"}])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_case(resume=True)

    def test_resume_rejects_changed_trace_or_saved_snapshots(self):
        self.run_case()
        path = next((self.out / "traces").glob("*.json"))
        original = parse(path.read_text())
        changed = copy.deepcopy(original)
        changed["usage"]["input_tokens"] = 12345
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "trace response or usage changed"):
            self.run_case(resume=True)
        write_json(path, original)
        write_json(self.out / "author-messages.json", {})
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.run_case(resume=True)

    def test_interrupted_dispatch_not_replayed_even_if_cost_unknown(self):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_case(dispatch=interrupt)
        result = self.run_case(resume=True)
        self.assertEqual(self.tasks, [])
        self.assertEqual(self.predictions()[0]["execution_errors"], ["interrupted_case_not_replayed"])
        self.assertEqual(result["usage"]["requests"], 1)
        self.assertIsNone(result["elapsed_seconds"])
        self.assertIsNone(result["usage"]["input_tokens"])

    def test_resume_recovers_every_checkpoint_write_window_without_replay(self):
        original_json, original_jsonl = engine.write_json, engine.write_jsonl
        for save_number in (1, 2, 3):
            for filename in ("checkpoint.json", "predictions.jsonl", "manifest.json"):
                for moment in ("before", "after"):
                    with self.subTest(save=save_number, file=filename, moment=moment):
                        self.out = self.root / f"fault-{save_number}-{filename}-{moment}"
                        self.tasks = []
                        state = {"save": 0, "interrupted": False}

                        def invoke(writer, path, value):
                            if path.name == "checkpoint.json":
                                state["save"] += 1
                            trigger = state["save"] == save_number and path.name == filename
                            if trigger and moment == "before":
                                state["interrupted"] = True
                                raise KeyboardInterrupt()
                            writer(path, value)
                            if trigger and moment == "after":
                                state["interrupted"] = True
                                raise KeyboardInterrupt()

                        with patch.object(engine, "write_json", side_effect=lambda p, v: invoke(original_json, p, v)), \
                             patch.object(engine, "write_jsonl", side_effect=lambda p, v: invoke(original_jsonl, p, v)):
                            with self.assertRaises(KeyboardInterrupt):
                                self.run_case()
                        self.assertTrue(state["interrupted"])
                        result = self.run_case(resume=True)
                        self.assertEqual(result["status"], "complete")
                        self.assertEqual(len(self.tasks), 1)
                        record = self.predictions()[0]
                        # Before the second checkpoint commit, the completed
                        # remote request is deliberately retained as an orphan.
                        if save_number == 2 and filename == "checkpoint.json" and moment == "before":
                            self.assertEqual(record["execution_errors"], ["interrupted_case_not_replayed"])
                        else:
                            self.assertEqual(record["is_assistant_correct"], "Correct")
                            self.assertEqual(record["verdict"], "supported")
                        checkpoint = parse((self.out / "checkpoint.json").read_text())
                        self.assertEqual(checkpoint["manifest"], parse((self.out / "manifest.json").read_text()))
                        self.assertEqual(checkpoint["predictions"], self.predictions())

    def test_checkpoint_and_mirror_tampering_are_not_repaired(self):
        self.run_case()
        checkpoint_path = self.out / "checkpoint.json"
        checkpoint = parse(checkpoint_path.read_text())
        changed = copy.deepcopy(checkpoint)
        changed["predictions"][0]["verdict"] = "contradicted"
        write_json(checkpoint_path, changed)
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            self.run_case(resume=True)
        write_json(checkpoint_path, checkpoint)
        for name, value in [("predictions.jsonl", [{"case_id": "tampered"}]),
                            ("manifest.json", {"tampered": True})]:
            path = self.out / name
            original = path.read_text()
            if name.endswith("jsonl"):
                write_jsonl(path, value)
            else:
                write_json(path, value)
            with self.assertRaisesRegex(ValueError, "outside a committed checkpoint"):
                self.run_case(resume=True)
            self.assertEqual(parse(path.read_text()) if name.endswith("json") else read_jsonl(path), value)
            path.write_text(original)
        checkpoint_path.unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint is missing"):
            self.run_case(resume=True)
        self.assertEqual(len(self.tasks), 1)

    def test_plan_never_dispatches_and_missing_method_defaults(self):
        self.config.pop("method")
        write_json(self.config_path, self.config)
        result = engine.plan(self.root / "cases.jsonl", self.config_path)
        self.assertFalse(result["model_called"])
        self.assertEqual(result["method"], "critic_original")
        self.assertEqual(result["provider_attempt_upper_bound"], 1)
        self.assertEqual(result["retries"], 0)
        self.assertEqual(self.tasks, [])
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
