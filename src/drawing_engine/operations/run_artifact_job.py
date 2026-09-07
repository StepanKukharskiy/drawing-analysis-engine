#!/usr/bin/env python3
"""Wrap any artifact producer with disk preflight, monitoring and accounting."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.operations.artifact_disk_usage import (GIB, DEFAULT_LEDGER, DiskBudgetError, artifact_disk_usage,
                                     check_capacity, measure_paths)


def run(command, *, outputs, temporary=(), ledger=None, min_free_bytes=3*GIB,
        reserve_bytes=0, poll_seconds=2, max_growth_bytes=None, required_outputs=(),
        **process_options):
    if not command or poll_seconds <= 0:
        raise ValueError("command and positive poll interval required")
    if max_growth_bytes is not None and max_growth_bytes < 0:
        raise ValueError("growth budget must be non-negative")
    ledger = Path(ledger or os.environ.get('REBAR_DISK_USAGE_LOG', DEFAULT_LEDGER))
    measured_paths = [*outputs, *temporary]
    before = measure_paths(measured_paths, exclude=[ledger])

    def check_growth():
        if max_growth_bytes is not None:
            current = measure_paths(measured_paths, exclude=[ledger])
            if not current['complete']:
                raise DiskBudgetError('cannot measure disk growth')
            if current['allocated_bytes'] - before['allocated_bytes'] > max_growth_bytes:
                raise DiskBudgetError('artifact job exceeded its allocated-byte growth budget; outputs retained')

    with artifact_disk_usage("artifact-job:" + Path(command[0]).name, outputs,
                             temporary_paths=temporary, ledger=ledger,
                             min_free_bytes=min_free_bytes, reserve_bytes=reserve_bytes):
        process = subprocess.Popen(command, start_new_session=True, **process_options)
        try:
            while True:
                try:
                    code = process.wait(timeout=poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    check_capacity([*outputs, *temporary], min_free_bytes=min_free_bytes)
                    check_growth()
            if code:
                raise subprocess.CalledProcessError(code, command)
            if any(not Path(path).is_file() for path in required_outputs):
                raise ValueError('stage did not publish its required artifacts')
            check_capacity([*outputs, *temporary], min_free_bytes=min_free_bytes)
            check_growth()
        except BaseException:
            # Signal only the newly created job process group, including workers.
            import signal
            def stop(sig):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            stop(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            stop(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop(signal.SIGKILL)
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, action="append", required=True,
                        help="exact output file/directory; repeat for additional scopes")
    parser.add_argument("--temporary", type=Path, action="append", default=[],
                        help="job-local staging/render directory, not all of tmp")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=3)
    parser.add_argument("--reserve-gib", type=float, default=0,
                        help="expected additional peak output/WAL/temp space")
    parser.add_argument("--max-growth-gib", type=float,
                        help="optional allocated growth budget across output and staging scopes")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    run(command, outputs=args.output, temporary=args.temporary, ledger=args.ledger,
        min_free_bytes=int(args.min_free_gib*GIB), reserve_bytes=int(args.reserve_gib*GIB),
        max_growth_bytes=None if args.max_growth_gib is None else int(args.max_growth_gib*GIB))


if __name__ == "__main__":
    main()
