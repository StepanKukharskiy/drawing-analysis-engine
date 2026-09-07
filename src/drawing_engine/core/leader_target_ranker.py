"""Small calibrated ranker for existing leader-to-rebar candidates.

The model ranks canonical geometric candidates only.  It cannot create a path,
accept an identity, or bypass the deterministic uniqueness and topology gate.
Training uses frozen graph-to-PDF examples with known target IDs.
"""

from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL_SCHEMA_VERSION = "0.1.0"
FEATURE_NAMES = (
    "terminal_proximity",
    "endpoint_proximity",
    "length_fraction",
    "stroke_support",
    "path_candidate_score",
    "repetition_support",
    "inside_view",
    "leader_crossing_angle",
    "native_source",
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "src/drawing_engine/resources/models/leader_target_ranker.v1.json"


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    fraction = 0.0 if denominator == 0 else max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    return math.dist(point, (start[0] + fraction * dx, start[1] + fraction * dy))


def _polyline_distance(point: tuple[float, float], points: list[list[float]]) -> float:
    return min(
        (_point_segment_distance(point, tuple(left), tuple(right)) for left, right in zip(points, points[1:])),
        default=math.inf,
    )


def _polyline_length(points: list[list[float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _angle(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return math.degrees(math.atan2(points[-1][1] - points[0][1], points[-1][0] - points[0][0])) % 180.0


def candidate_features(
    terminal: tuple[float, float],
    candidate: Mapping[str, Any],
    *,
    view_bbox: Iterable[float],
    leader_angle_deg: float | None = None,
) -> dict[str, float]:
    geometry = candidate.get("geometry") or {}
    points = geometry.get("points_display") or []
    box = [float(value) for value in view_bbox]
    view_diagonal = max(math.hypot(box[2] - box[0], box[3] - box[1]), 1.0)
    distance = _polyline_distance(terminal, points)
    endpoint_distance = min((math.dist(terminal, tuple(point)) for point in (points[:1] + points[-1:])), default=math.inf)
    length = float(geometry.get("length_points") or _polyline_length(points))
    width = float((candidate.get("style") or {}).get("width_pt") or 0.0)
    path_angle = float(geometry.get("angle_deg") if geometry.get("angle_deg") is not None else _angle(points))
    if leader_angle_deg is None:
        crossing = 0.5
    else:
        delta = abs(path_angle - leader_angle_deg) % 180.0
        delta = min(delta, 180.0 - delta)
        crossing = math.sin(math.radians(delta))
    center = (
        sum(float(point[0]) for point in points) / len(points) if points else math.inf,
        sum(float(point[1]) for point in points) / len(points) if points else math.inf,
    )
    return {
        "terminal_proximity": math.exp(-min(distance, 100.0) / 4.0),
        "endpoint_proximity": math.exp(-min(endpoint_distance, 100.0) / 7.0),
        "length_fraction": min(1.0, length / view_diagonal),
        "stroke_support": min(1.0, width / 2.0),
        "path_candidate_score": max(0.0, min(1.0, float(candidate.get("candidate_score") or 0.0))),
        "repetition_support": 1.0 if candidate.get("repetition_group_ids") else 0.0,
        "inside_view": 1.0 if box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3] else 0.0,
        "leader_crossing_angle": crossing,
        "native_source": 0.0 if str(candidate.get("observation_method") or "native").startswith("opencv") else 1.0,
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def train_logistic_ranker(
    examples: Iterable[Mapping[str, Any]],
    *,
    epochs: int = 2400,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict[str, Any]:
    rows = [dict(item) for item in examples]
    if not rows or not {int(item["label"]) for item in rows} == {0, 1}:
        raise ValueError("leader ranker training requires positive and negative examples")
    weights = [0.0 for _ in FEATURE_NAMES]
    bias = 0.0
    positives = sum(int(item["label"]) == 1 for item in rows)
    negatives = len(rows) - positives
    positive_weight = negatives / positives
    for _ in range(epochs):
        gradient = [0.0 for _ in FEATURE_NAMES]
        bias_gradient = 0.0
        weight_sum = 0.0
        for row in rows:
            features = row["features"]
            vector = [float(features[name]) for name in FEATURE_NAMES]
            label = int(row["label"])
            sample_weight = positive_weight if label else 1.0
            prediction = _sigmoid(bias + sum(weight * value for weight, value in zip(weights, vector)))
            error = sample_weight * (prediction - label)
            for index, value in enumerate(vector):
                gradient[index] += error * value
            bias_gradient += error
            weight_sum += sample_weight
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradient[index] / weight_sum + l2 * weights[index])
        bias -= learning_rate * bias_gradient / weight_sum
    source_hash = sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "deterministic_logistic_leader_target_ranker",
        "feature_names": list(FEATURE_NAMES),
        "weights": [round(value, 10) for value in weights],
        "bias": round(bias, 10),
        "training": {
            "example_count": len(rows),
            "positive_count": positives,
            "negative_count": negatives,
            "source_sha256": source_hash,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
        },
        "acceptance_contract": {
            "ranking_only": True,
            "probability_cannot_accept_identity": True,
            "deterministic_uniqueness_gate_required": True,
            "schedule_values_used": False,
        },
    }
    model["model_sha256"] = model_sha256(model)
    return model


def model_sha256(model: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in model.items() if key != "model_sha256"}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_model(model: Mapping[str, Any]) -> None:
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported leader ranker schema")
    if tuple(model.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("leader ranker feature schema mismatch")
    if len(model.get("weights") or []) != len(FEATURE_NAMES):
        raise ValueError("leader ranker weight count mismatch")
    if model.get("model_sha256") != model_sha256(model):
        raise ValueError("leader ranker hash mismatch")
    if model.get("acceptance_contract", {}).get("ranking_only") is not True:
        raise ValueError("leader ranker must remain ranking-only")


def score_features(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    validate_model(model)
    value = float(model["bias"]) + sum(
        float(weight) * float(features[name])
        for weight, name in zip(model["weights"], FEATURE_NAMES)
    )
    return _sigmoid(value)


def rank_candidates(
    model: Mapping[str, Any],
    terminal: tuple[float, float],
    candidates: Iterable[Mapping[str, Any]],
    *,
    view_bbox: Iterable[float],
    leader_angle_deg: float | None = None,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        features = candidate_features(
            terminal,
            candidate,
            view_bbox=view_bbox,
            leader_angle_deg=leader_angle_deg,
        )
        ranked.append(
            {
                "candidate_id": str(candidate["id"]),
                "probability": round(score_features(model, features), 6),
                "features": {name: round(features[name], 6) for name in FEATURE_NAMES},
            }
        )
    return sorted(ranked, key=lambda item: (-item["probability"], item["candidate_id"]))


def deterministic_acceptance_gate(
    ranked: list[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    terminal: tuple[float, float],
    *,
    minimum_probability: float = 0.65,
    minimum_probability_margin: float = 0.12,
) -> dict[str, Any]:
    if not ranked:
        return {"status": "abstained", "reason": "no ranked candidates"}
    winner = ranked[0]
    runner_probability = float(ranked[1]["probability"]) if len(ranked) > 1 else 0.0
    candidate = candidates_by_id[str(winner["candidate_id"])]
    points = candidate.get("geometry", {}).get("points_display", [])
    winner_distance = _polyline_distance(terminal, points)
    other_distances = sorted(
        _polyline_distance(terminal, row.get("geometry", {}).get("points_display", []))
        for candidate_id, row in candidates_by_id.items()
        if candidate_id != winner["candidate_id"]
    )
    geometry_margin = (other_distances[0] - winner_distance) if other_distances else math.inf
    probability_margin = float(winner["probability"]) - runner_probability
    accepted = (
        float(winner["probability"]) >= minimum_probability
        and probability_margin >= minimum_probability_margin
        and winner_distance <= 5.0
        and geometry_margin >= 1.25
    )
    return {
        "status": "accepted" if accepted else "abstained",
        "candidate_id": winner["candidate_id"] if accepted else None,
        "ranked_winner_id": winner["candidate_id"],
        "winner_probability": winner["probability"],
        "probability_margin": round(probability_margin, 6),
        "terminal_distance_points": round(winner_distance, 4),
        "geometry_margin_points": None if math.isinf(geometry_margin) else round(geometry_margin, 4),
        "reason": None if accepted else "learned winner did not close deterministic distance and uniqueness gates",
        "contract": {
            "model_ranks_only": True,
            "geometry_uniqueness_required": True,
            "schedule_values_used": False,
        },
    }


def load_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    validate_model(model)
    return model


@lru_cache(maxsize=1)
def load_default_model() -> dict[str, Any] | None:
    return load_model(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.is_file() else None
