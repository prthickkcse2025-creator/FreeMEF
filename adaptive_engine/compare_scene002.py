#!/usr/bin/env python3

import os
import sys

sys.path.insert(
    0,
    os.path.expanduser("~/FreeMEF")
)

from adaptive_engine.evaluator import evaluate


BASE = os.path.expanduser("~/FreeMEF")

NORMAL = os.path.join(
    BASE,
    "my_test",
    "scene002",
    "02_normal.JPG"
)

CANDIDATES = {
    "V3.3 Client-Bright": os.path.join(
        BASE,
        "adaptive_outputs",
        "scene002_v33",
        "FINAL_SCENE002_V33.jpg"
    ),

    "Mertens V2": os.path.join(
        BASE,
        "adaptive_outputs",
        "mertens_scene002.png"
    ),

    "MEF-Net": os.path.join(
        BASE,
        "adaptive_outputs",
        "mefnet_scene002.png"
    )
}


def main():

    print()
    print("=" * 70)
    print("SCENE002 — THREE-METHOD COMPARISON")
    print("=" * 70)

    print(
        f"\nNormal reference:\n{NORMAL}"
    )

    results = {}

    for name, path in CANDIDATES.items():

        if not os.path.isfile(path):

            print(
                f"\nERROR: {name} output missing:"
            )

            print(path)

            continue

        scores = evaluate(
            path,
            NORMAL
        )

        results[name] = scores

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        print(
            f"Brightness         : "
            f"{scores['brightness']:.4f}"
        )

        print(
            f"Shadow             : "
            f"{scores['shadow']:.4f}"
        )

        print(
            f"Highlight          : "
            f"{scores['highlight']:.4f}"
        )

        print(
            f"Color              : "
            f"{scores['color']:.4f}"
        )

        print(
            f"Structure          : "
            f"{scores['structure']:.4f}"
        )

        print(
            f"Artificial lift    : "
            f"{scores['artificial_lift']:.4f}"
        )

        print(
            f"Brightness delta   : "
            f"{scores['brightness_delta']:+.4f}"
        )

        print(
            f"FINAL SCORE        : "
            f"{scores['final']:.4f}"
        )

    if not results:

        raise RuntimeError(
            "No candidate outputs were found."
        )

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["final"],
        reverse=True
    )

    print()
    print("=" * 70)
    print("SCENE002 RANKING")
    print("=" * 70)

    for index, (name, scores) in enumerate(
        ranking,
        start=1
    ):

        print(
            f"{index}. "
            f"{name:<24} "
            f"{scores['final']:.4f}"
        )

    print()
    print(
        f"WINNER: {ranking[0][0]}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
