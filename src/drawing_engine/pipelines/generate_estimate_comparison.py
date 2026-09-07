#!/usr/bin/env python3
"""Compare frozen drawing-derived quantities with independently parsed schedules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.concrete.schedule_comparison import write_estimate_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "output" / "object_agnostic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "estimates")
    args = parser.parse_args()
    for source in args.inputs:
        graph = args.graph_dir / f"{source.stem}.engineering-graph.json"
        if not graph.exists():
            parser.error(f"frozen engineering graph not found: {graph}")
        output = args.output_dir / f"{source.stem}.estimate-comparison.json"
        print(write_estimate_comparison(source, graph, output))


if __name__ == "__main__":
    main()
