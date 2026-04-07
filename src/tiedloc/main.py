"""Entry point for the tiedloc Network-to-Network service simulator.

Usage:
    python -m tiedloc.main [path_to_input_file]
    tiedloc [path_to_input_file]
"""

from __future__ import annotations

import argparse
import json
import sys

from tiedloc.api import SimulationConfig, run
from tiedloc import simulations


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the simulation, and output results.

    Args:
        argv: Optional list of command-line arguments (defaults to sys.argv).
    """
    parser = argparse.ArgumentParser(description="Tiedloc: Network-to-Network service simulator")
    parser.add_argument(
        "pathToInputFile",
        default="input.json",
        metavar="pathToInputFile",
        type=str,
        help="Path to the JSON input configuration file",
        nargs="?",
    )
    args = parser.parse_args(argv)

    # Load and display input (single read)
    try:
        with open(args.pathToInputFile, "r") as json_data:
            inputs = json.load(json_data)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.pathToInputFile}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in {args.pathToInputFile}: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Input:")
    print(json.dumps(inputs, sort_keys=True, indent=4, separators=(",", ": ")))

    # Run simulation via API
    config = SimulationConfig.from_dict(inputs)
    results = run(config)

    # Results handling
    print("Results:")
    print(json.dumps(results.stats, sort_keys=True, indent=4, separators=(",", ": ")))
    simulations.save_results(args.pathToInputFile, results.stats, results.samples)


if __name__ == "__main__":
    main()
