#!/usr/bin/env python3
"""Run the project CLI from any PDF folder: python3 /path/to/rebar.py --help."""
from src.drawing_engine.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
