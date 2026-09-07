#!/usr/bin/env python3
"""Generate the reviewed coverage audit and frozen real M7B count result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_bounded_discrete_counts import (  # noqa: E402
    build_mep_bounded_discrete_counts,
    validate_mep_bounded_discrete_counts,
)
from src.drawing_engine.disciplines.mep.mep_occurrence_coverage_audit import (  # noqa: E402
    build_mep_occurrence_coverage_audit,
    validate_mep_occurrence_coverage_audit,
)


FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m4_discrete_coverage"
DEFAULT_M0 = FIXTURE_ROOT / "m_and_p_coordination.annotation-observations.json"
DEFAULT_M1 = FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"
DEFAULT_M4 = CHECKPOINT_ROOT / "full_package.attribute-bindings.json"
DEFAULT_TARGET_REVIEW = CHECKPOINT_ROOT / "full_package.discrete-target-review.json"
DEFAULT_TRUTH = FIXTURE_ROOT / "m_and_p_coordination_m7b_coverage_truth.json"
DEFAULT_AUDIT = FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json"
DEFAULT_COUNTS = FIXTURE_ROOT / "m_and_p_coordination.bounded-discrete-counts.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return path


def generate(
    *,
    m0_path: Path,
    m1_path: Path,
    m4_path: Path,
    target_review_path: Path | None,
    truth_path: Path,
    audit_path: Path,
    counts_path: Path,
) -> tuple[Path, Path]:
    m0 = _load(m0_path)
    m1 = _load(m1_path)
    m4 = _load(m4_path)
    audit = build_mep_occurrence_coverage_audit(
        sheet_registry=m1,
        attribute_bindings=m4,
        annotation_observations=m0,
        reviewed_coverage=_load(truth_path),
        discrete_target_review=(
            _load(target_review_path) if target_review_path is not None else None
        ),
    )
    errors = validate_mep_occurrence_coverage_audit(audit)
    if errors:
        raise ValueError("\n".join(errors))
    counts = build_mep_bounded_discrete_counts(
        sheet_registry=m1,
        attribute_bindings=m4,
        occurrence_coverage_audit=audit,
        identity_evidence=[],
    )
    errors = validate_mep_bounded_discrete_counts(counts)
    if errors:
        raise ValueError("\n".join(errors))
    return _write(audit_path, audit), _write(counts_path, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", type=Path, default=DEFAULT_M0)
    parser.add_argument("--m1", type=Path, default=DEFAULT_M1)
    parser.add_argument("--m4", type=Path, default=DEFAULT_M4)
    parser.add_argument("--target-review", type=Path, default=DEFAULT_TARGET_REVIEW)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    args = parser.parse_args()
    for path in generate(
        m0_path=args.m0.resolve(),
        m1_path=args.m1.resolve(),
        m4_path=args.m4.resolve(),
        target_review_path=args.target_review.resolve(),
        truth_path=args.truth.resolve(),
        audit_path=args.audit.resolve(),
        counts_path=args.counts.resolve(),
    ):
        print(path)


if __name__ == "__main__":
    main()
