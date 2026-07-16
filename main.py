"""Entry point for Good Model Labs workflow benchmark framework."""

from __future__ import annotations

import argparse

from examples.run_food_pantry_benchmark import run as run_food_pantry
from examples.run_nonprofit_tool_matching_benchmark import run as run_nonprofit_tool_matching


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Good Model Labs benchmarks.")
    parser.add_argument(
        "--track",
        choices=["nonprofit_tool_matching", "food_pantry", "all"],
        default="nonprofit_tool_matching",
        help="Benchmark track to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.track == "nonprofit_tool_matching":
        run_nonprofit_tool_matching()
    elif args.track == "food_pantry":
        run_food_pantry()
    else:
        run_nonprofit_tool_matching()
        run_food_pantry()
