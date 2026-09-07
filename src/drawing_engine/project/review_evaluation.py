"""Task-level evaluation from engineer-reviewed graph candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping

from src.drawing_engine.project.review_feedback import (
    TASK_TYPES,
    build_review_candidate_catalog,
    validate_review_record,
)


PRIMARY_EVALUATION_TASKS = (
    "primitive_role",
    "view_role",
    "leader_target",
    "dimension_ownership",
    "cross_view_identity",
)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [item for item in rows if item["decision"] == "accepted"]
    negative = [item for item in rows if item["decision"] != "accepted"]
    auto_accepted = [item for item in rows if item["engine_decision"] == "accepted"]
    auto_rejected = [item for item in rows if item["engine_decision"] == "rejected"]
    abstained = [item for item in rows if item["engine_decision"] in {"abstained", "proposed"}]
    true_positive = sum(item["decision"] == "accepted" for item in auto_accepted)
    false_positive = len(auto_accepted) - true_positive
    true_negative = sum(item["decision"] != "accepted" for item in auto_rejected)
    false_negative = len(auto_rejected) - true_negative
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, len(positive))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    scored = [item for item in rows if isinstance(item.get("score"), (int, float))]
    brier = None
    if scored:
        brier = sum((float(item["score"]) - (1.0 if item["decision"] == "accepted" else 0.0)) ** 2 for item in scored) / len(scored)
    covered = true_positive + false_positive + true_negative + false_negative
    return {
        "reviewed_candidate_count": len(rows),
        "accepted_count": len(positive),
        "rejected_count": sum(item["decision"] == "rejected" for item in rows),
        "corrected_count": sum(item["decision"] == "corrected" for item in rows),
        "candidate_precision": _ratio(len(positive), len(rows)),
        "auto_accepted_count": len(auto_accepted),
        "auto_rejected_count": len(auto_rejected),
        "abstained_or_proposed_count": len(abstained),
        "abstention_rate": _ratio(len(abstained), len(rows)),
        "unnecessary_abstention_count": sum(item["decision"] == "accepted" for item in abstained),
        "wrong_auto_accept_count": false_positive,
        "confusion_on_automatic_decisions": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "automatic_precision": precision,
        "automatic_recall": recall,
        "automatic_f1": f1,
        "covered_accuracy": _ratio(true_positive + true_negative, covered),
        "scored_candidate_count": len(scored),
        "brier_score": brier,
    }


def evaluate_review_records(
    canonical_graph: Mapping[str, Any],
    review_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate reviewed candidates by semantic task without schedule feedback."""

    catalog = build_review_candidate_catalog(canonical_graph)
    candidate_by_id = {item["candidate_id"]: item for item in catalog["candidates"]}
    decisions_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation_errors: list[str] = []
    invalid_record_count = 0
    record_count = 0
    raw_decision_count = 0
    for record_index, record in enumerate(review_records):
        record_count += 1
        validation = validate_review_record(canonical_graph, record)
        if validation["status"] != "pass":
            invalid_record_count += 1
            validation_errors.extend(f"record[{record_index}]: {error}" for error in validation["errors"])
            continue
        for decision in record["decisions"]:
            raw_decision_count += 1
            candidate = candidate_by_id[decision["candidate_id"]]
            decisions_by_candidate[decision["candidate_id"]].append(
                {
                    "candidate_id": decision["candidate_id"],
                    "task_type": decision["task_type"],
                    "decision": decision["decision"],
                    "reason_code": decision["reason_code"],
                    "correction": decision.get("correction"),
                    "engine_decision": candidate["engine_decision"],
                    "score": candidate.get("score"),
                }
            )
    conflicts: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    for candidate_id, rows in sorted(decisions_by_candidate.items()):
        outcomes = {item["decision"] for item in rows}
        corrections = {
            json.dumps(item.get("correction"), sort_keys=True, ensure_ascii=False)
            for item in rows
            if item["decision"] == "corrected"
        }
        if len(outcomes) > 1 or len(corrections) > 1:
            conflicts.append({"candidate_id": candidate_id, "decisions": sorted(outcomes)})
            continue
        reviewed.append(rows[0])
    by_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed:
        by_task_rows[row["task_type"]].append(row)
    tasks = sorted(TASK_TYPES | set(by_task_rows))
    reason_counts = Counter(row["reason_code"] for row in reviewed)
    result = {
        "schema_version": "0.1.0",
        "layer": "review_task_evaluation",
        "document_key": catalog["document_key"],
        "base_canonical_graph_sha256": catalog["base_canonical_graph_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "overall": _task_metrics(reviewed),
        "by_task": {task: _task_metrics(by_task_rows.get(task, [])) for task in tasks},
        "priority_tasks": list(PRIMARY_EVALUATION_TASKS),
        "review_effort": {
            "review_record_count": record_count,
            "raw_decision_count": raw_decision_count,
            "unique_reviewed_candidate_count": len(reviewed),
            "conflicting_candidate_count": len(conflicts),
            "correction_count": sum(item["decision"] == "corrected" for item in reviewed),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "conflicts": conflicts,
        "validation": {
            "status": "pass" if not validation_errors and not conflicts else "review_required",
            "errors": validation_errors,
            "invalid_record_count": invalid_record_count,
            "conflicting_candidate_count": len(conflicts),
            "schedule_values_used": False,
        },
        "contract": {
            "evaluation_uses_reviewed_graph_candidates_only": True,
            "declared_schedule_is_not_a_label_source": True,
            "abstentions_are_measured_separately": True,
            "conflicting_reviews_fail_closed": True,
        },
    }
    return result
