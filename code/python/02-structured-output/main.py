"""Entry point for the chapter-02 prompt-engineering demos.

Use --mode to switch between the three examples referenced by the chapter.
Structured-output examples still run directly from their own scripts.
"""

from __future__ import annotations

import argparse

from chain_of_thought import main as chain_of_thought_main
from few_shot_comparison import main as few_shot_main
from prompt_template import main as prompt_template_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("template", "few-shot", "chain-of-thought"),
        default="template",
        help="Which chapter-02 example to run.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "template":
        prompt_template_main()
    elif args.mode == "few-shot":
        few_shot_main()
    else:
        chain_of_thought_main()
