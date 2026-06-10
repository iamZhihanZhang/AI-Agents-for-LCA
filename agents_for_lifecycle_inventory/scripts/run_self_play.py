# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - CLI entrypoint.

See README.md for usage and system overview.
"""

import sys
from agentic_lci.agents.self_play import run_self_play


def main():
    args = sys.argv
    kwargs = {}
    for i in range(len(args)):
        if args[i].startswith("--") and i + 1 < len(args):
            kwargs[args[i].removeprefix("--")] = args[i + 1]

    # TODO: parse --llm-provider and --llm-model flags

    lci, critique = run_self_play(
        kwargs.get("query"),
        kwargs.get("image"),
        max_rounds=int(kwargs.get("max-rounds", "5")),
    )
    print("LCI:")
    print(lci.model_dump_json(indent=4))  # type: ignore
    print("Final Critique:")
    print(critique.model_dump_json(indent=4))  # type: ignore


if __name__ == "__main__":
    main()
