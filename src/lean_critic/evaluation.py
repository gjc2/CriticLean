"""Score the model's recorded labels against the unchanged public benchmark.

This module never judges the mathematics, evidence, reasons or uncertainty.
Technical failures stay in the accuracy denominator and are reported separately.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .datasets import (VERDICTS, file_hash, index_rows, read_jsonl, sha256,
                       validate_bindings, validate_cases, write_json, write_jsonl)

_LABELS = {"supported": "Correct", "contradicted": "Incorrect", "unknown": ""}


def ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _block(ids: list[str], gold: dict[str, dict], decisions: dict[str, str]) -> dict:
    positive = [case_id for case_id in ids if gold[case_id]["expected_verdict"] == "supported"]
    negative = [case_id for case_id in ids if gold[case_id]["expected_verdict"] == "contradicted"]
    counts = {"true_support": 0, "false_support": 0, "true_rejection": 0, "false_rejection": 0,
              "unknown_positive": 0, "unknown_negative": 0}
    for case_id in ids:
        is_positive = gold[case_id]["expected_verdict"] == "supported"
        verdict = decisions[case_id]
        key = ("unknown_positive" if is_positive else "unknown_negative") if verdict == "unknown" else (
            ("true_support" if is_positive else "false_support") if verdict == "supported" else
            ("false_rejection" if is_positive else "true_rejection"))
        counts[key] += 1
    correct = counts["true_support"] + counts["true_rejection"]
    supported = counts["true_support"] + counts["false_support"]
    contradicted = counts["true_rejection"] + counts["false_rejection"]
    decided = supported + contradicted
    classification = {
        "positive_class": "Correct",
        "definitions": "standard_binary_with_technical_failures_retained_in_class_denominators",
        "ACC": ratio(correct, len(ids)),
        "TPR": ratio(counts["true_support"], len(positive)),
        "FPR": ratio(counts["false_support"], len(negative)),
        "TNR": ratio(counts["true_rejection"], len(negative)),
        "FNR": ratio(counts["false_rejection"], len(positive)),
        "unknown_positive_rate": ratio(counts["unknown_positive"], len(positive)),
        "unknown_negative_rate": ratio(counts["unknown_negative"], len(negative)),
    }
    return {"n": len(ids), "labeled": len(ids), "gold_supported": len(positive), "gold_contradicted": len(negative),
            "correct": correct, "counts": counts, "supported": supported, "contradicted": contradicted,
            "unknown": len(ids) - decided, "coverage": ratio(decided, len(ids)),
            "agreement_over_labeled": ratio(correct, len(ids)),
            "classification_metrics": classification}


def _execution_errors(prediction: dict) -> list[str]:
    value = prediction.get("execution_errors", [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return ["malformed_execution_errors"]
    return list(dict.fromkeys(value))


def _load(cases_path: Path, gold_path: Path, predictions_path: Path):
    cases = validate_cases(read_jsonl(Path(cases_path))[0])
    if not cases:
        raise ValueError("cannot evaluate an empty case file")
    gold = index_rows(read_jsonl(Path(gold_path))[0], "gold")
    # A connection-test run may use a subset of the full private label sidecar.
    validate_bindings(cases, gold, exact=False)
    rows, diagnostics = read_jsonl(Path(predictions_path), tolerant=True)
    predictions = []
    for prediction in rows:
        if not isinstance(prediction.get("case_id"), str) or not prediction["case_id"]:
            diagnostics.append({"error": "prediction_missing_case_id"})
        else:
            predictions.append(prediction)
    indexed = index_rows(predictions, "predictions")
    if indexed.keys() - cases.keys():
        raise ValueError("prediction IDs outside this run's case file")
    decisions, failures = {}, {}
    for case_id, case in cases.items():
        prediction = indexed.get(case_id)
        if prediction is None:
            decisions[case_id] = "unknown"
            failures[case_id] = ["missing_prediction"]
            diagnostics.append({"case_id": case_id, "error": "missing_prediction"})
            continue
        if (prediction.get("source_sha256") != sha256(case["source_nl"])
                or prediction.get("candidate_sha256") != sha256(case["lean_code"])):
            raise ValueError(f"{case_id}: prediction input hash mismatch; refusing to score unrelated inputs")
        verdict = prediction.get("verdict")
        errors = _execution_errors(prediction)
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            verdict = "unknown"
            errors.append("invalid_verdict")
        if verdict == "unknown" and not errors:
            errors.append("no_model_label")
        # Preserve a valid model label even if a recorder also lists a technical
        # diagnostic. We never examine reasons, quotes or unresolved questions.
        decisions[case_id] = verdict
        if errors:
            failures[case_id] = list(dict.fromkeys(errors))
            diagnostics.extend({"case_id": case_id, "error": error} for error in failures[case_id])
        raw_label = prediction.get("is_assistant_correct")
        if isinstance(raw_label, str) and raw_label in {"Correct", "Incorrect"} and verdict != "unknown" and raw_label != _LABELS[verdict]:
            raise ValueError(f"{case_id}: recorded model label and verdict disagree")
    return cases, gold, indexed, decisions, failures, diagnostics


def _shown(metric: dict) -> str:
    value = metric["value"]
    return "—" if value is None else f"{metric['numerator']}/{metric['denominator']} ({value:.2%})"


def evaluate(cases_path: Path, gold_path: Path, predictions_path: Path, out: Path) -> dict:
    """Compute public-label agreement without altering any model judgment.

    Invalid/missing labels are technical failures, not Incorrect predictions.
    They contribute zero correct answers but remain in the full denominator.
    Input hash mismatches and duplicate records reject evaluation altogether.
    """
    cases, gold, predictions, decisions, failures, diagnostics = _load(cases_path, gold_path, predictions_path)
    ids = list(cases)
    compile_passing = [case_id for case_id in ids if gold[case_id]["public_tag"] != "compile_false"]
    tags = sorted({gold[case_id]["public_tag"] for case_id in ids})
    reason_counts = Counter(error for errors in failures.values() for error in errors)
    result = {
        "schema_version": "criticlean-deepseek-evaluation-v1",
        "interpretation": "Agreement of the unchanged model labels with public author annotations; no programmatic semantic judgment.",
        "exact_published_scorer_reproduction": False,
        "published_metric_warning": "The authors' complete scorer and decoding configuration are not available here. ACC uses all run cases; TPR/FPR/TNR/FNR use explicitly defined standard class denominators.",
        "positive_class": "Correct", "negative_class": "Incorrect",
        "cases_sha256": file_hash(Path(cases_path)), "gold_sha256": file_hash(Path(gold_path)),
        "predictions_sha256": file_hash(Path(predictions_path)),
        "full_500_record_count": len(ids) == 500,
        "overall": _block(ids, gold, decisions),
        "author_compile_passing_subset": _block(compile_passing, gold, decisions),
        "per_public_tag": {tag: _block([case_id for case_id in ids if gold[case_id]["public_tag"] == tag], gold, decisions) for tag in tags},
        "technical_failures": {"case_count": len(failures), "by_reason": dict(sorted(reason_counts.items())),
                               "cases": [{"case_id": case_id, "execution_errors": failures[case_id]} for case_id in ids if case_id in failures]},
        "diagnostic_count": len(diagnostics), "author_compile_tags_freshly_verified": False,
        "limitations": ["Public reference labels may contain disputed cases and are not changed by this evaluator.",
                        "Compiler tags are the authors' annotations; Lean is not executed by this experiment.",
                        "A small connection test does not estimate full-benchmark performance."]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)
    with (out / "per-case.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["published_row_index", "case_id", "author_id", "author_source", "public_tag", "gold_label", "model_label", "raw_model_verdict", "agrees_with_gold", "execution_errors"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in ids:
            reference = gold[case_id]
            writer.writerow({"published_row_index": reference.get("published_row_index", ""), "case_id": case_id,
                             "author_id": reference.get("author_id", ""), "author_source": reference.get("author_source", ""),
                             "public_tag": reference["public_tag"], "gold_label": _LABELS[reference["expected_verdict"]],
                             "model_label": _LABELS[decisions[case_id]], "raw_model_verdict": predictions.get(case_id, {}).get("verdict", ""),
                             "agrees_with_gold": decisions[case_id] == reference["expected_verdict"],
                             "execution_errors": ";".join(failures.get(case_id, []))})
    lines = ["# DeepSeek / CriticLeanBench 评测结果", "",
             "直接统计模型的 Correct / Incorrect，与作者公开标签比较。程序不判断数学理由、引文或疑点，也不改写模型结论。", "",
             "| 范围 | 样本数 | 与标签一致 / 全部样本（ACC） | 模型未给出有效标签 |",
             "| --- | ---: | ---: | ---: |"]
    for title, block in [("全部运行样本", result["overall"]), ("作者标为编译通过的子集", result["author_compile_passing_subset"])]:
        lines.append(f"| {title} | {block['n']} | {_shown(block['classification_metrics']['ACC'])} | {block['unknown']} |")
    lines += ["", f"技术诊断涉及 **{len(failures)}** 条样本；没有有效模型标签的样本保留在 ACC 分母中。",
              "", "| 公开 tag | 样本数 | ACC |", "| --- | ---: | ---: |"]
    for tag, block in result["per_public_tag"].items():
        lines.append(f"| `{tag}` | {block['n']} | {_shown(block['classification_metrics']['ACC'])} |")
    lines += ["", "`compile_right_is_right` 对应 Correct；其余两个 tag 对应 Incorrect。编译失败类包含形式化失败，与纯语义偏差应分开查看。",
              "", "`metrics.json` 提供标准二分类指标及完整分母；`per-case.csv` 提供逐题公开标签、模型标签和技术状态；原始请求及回复保存在 run 目录。",
              "", "本报告不是独立数学证明。作者完整解码设置与评分程序未在此复现，不能把本地分数称为与论文完全一致的复现实验。", ""]
    if len(ids) != 500:
        lines.insert(3, "**这是少量样本的连通性测试，不能作为完整 CriticLeanBench 分数。**\n")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result
