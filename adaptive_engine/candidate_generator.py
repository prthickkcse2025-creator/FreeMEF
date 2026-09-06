#!/usr/bin/env python3

"""
FREE MEF - THREE CANDIDATE GENERATOR

Candidate A:
    V3.3 Client-Bright adaptive fusion

Candidate B:
    Mertens exposure fusion

Candidate C:
    MEF-Net neural fusion

INPUT:
    Underexposed image
    Normal exposure image
    Overexposed image

OUTPUT:
    candidate_A.jpg
    candidate_B.jpg
    candidate_C.jpg
"""

import os
import sys
import argparse
import traceback


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# Make ~/FreeMEF importable.
#
# This is required because this script lives inside:
#
#     ~/FreeMEF/adaptive_engine/
#
# while the candidate modules live inside:
#
#     ~/FreeMEF/production/
#     ~/FreeMEF/adaptive_modules/
#
if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "adaptive_outputs"
)

os.makedirs(
    DEFAULT_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CANDIDATE A
# ============================================================

from production.adaptive_fusion_v3_3_client_bright import (
    process as run_candidate_a
)


# ============================================================
# CANDIDATE B
# ============================================================

from adaptive_modules.mertens.mertens_adapter import (
    run_mertens
)


# ============================================================
# CANDIDATE C
# ============================================================

from adaptive_modules.mefnet.mefnet_adapter import (
    run_mefnet
)


# ============================================================
# FILE VALIDATION
# ============================================================

def check_input_file(
    path,
    name
):

    if path is None:

        raise ValueError(
            f"{name} image path is missing."
        )

    path = os.path.abspath(
        os.path.expanduser(
            path
        )
    )

    if not os.path.isfile(
        path
    ):

        raise FileNotFoundError(
            f"{name} image was not found:\n"
            f"{path}"
        )

    return path


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def check_output_file(
    path,
    candidate_name
):

    if not os.path.isfile(
        path
    ):

        raise RuntimeError(
            f"{candidate_name} did not create "
            f"an output image:\n"
            f"{path}"
        )

    size = os.path.getsize(
        path
    )

    if size <= 0:

        raise RuntimeError(
            f"{candidate_name} created an empty file:\n"
            f"{path}"
        )

    return path


# ============================================================
# CANDIDATE A
# ============================================================

def generate_candidate_a(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 70)
    print("CANDIDATE A")
    print("V3.3 CLIENT-BRIGHT")
    print("=" * 70)

    print()
    print(
        "Under  :",
        under_path
    )

    print(
        "Normal :",
        normal_path
    )

    print(
        "Over   :",
        over_path
    )

    print()
    print(
        "Output :",
        output_path
    )

    # --------------------------------------------------------
    # Run existing V3.3 implementation
    # --------------------------------------------------------

    run_candidate_a(
        under_path,
        normal_path,
        over_path,
        output_path
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    check_output_file(
        output_path,
        "Candidate A"
    )

    print()
    print(
        "Candidate A COMPLETE"
    )

    return output_path


# ============================================================
# CANDIDATE B
# ============================================================

def generate_candidate_b(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 70)
    print("CANDIDATE B")
    print("MERTENS EXPOSURE FUSION")
    print("=" * 70)

    print()
    print(
        "Under  :",
        under_path
    )

    print(
        "Normal :",
        normal_path
    )

    print(
        "Over   :",
        over_path
    )

    print()
    print(
        "Output :",
        output_path
    )

    # --------------------------------------------------------
    # Run existing Mertens implementation
    # --------------------------------------------------------

    run_mertens(
        under_path,
        normal_path,
        over_path,
        output_path
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    check_output_file(
        output_path,
        "Candidate B"
    )

    print()
    print(
        "Candidate B COMPLETE"
    )

    return output_path


# ============================================================
# CANDIDATE C
# ============================================================

def generate_candidate_c(
    under_path,
    normal_path,
    over_path,
    output_path,
    use_cuda=False
):

    print()
    print("=" * 70)
    print("CANDIDATE C")
    print("MEF-NET DIRECT INFERENCE")
    print("=" * 70)

    print()
    print(
        "Under  :",
        under_path
    )

    print(
        "Normal :",
        normal_path
    )

    print(
        "Over   :",
        over_path
    )

    print()
    print(
        "CUDA   :",
        use_cuda
    )

    print()
    print(
        "Output :",
        output_path
    )

    # --------------------------------------------------------
    # Run existing MEF-Net implementation
    # --------------------------------------------------------

    run_mefnet(
        under_path,
        normal_path,
        over_path,
        output_path,
        use_cuda=use_cuda
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    check_output_file(
        output_path,
        "Candidate C"
    )

    print()
    print(
        "Candidate C COMPLETE"
    )

    return output_path


# ============================================================
# GENERATE ALL CANDIDATES
# ============================================================

def generate_candidates(
    under_path,
    normal_path,
    over_path,
    output_dir=None,
    use_cuda=False
):

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    under_path = check_input_file(
        under_path,
        "Underexposed"
    )

    normal_path = check_input_file(
        normal_path,
        "Normal"
    )

    over_path = check_input_file(
        over_path,
        "Overexposed"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    if output_dir is None:

        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = os.path.abspath(
        os.path.expanduser(
            output_dir
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    candidate_a_path = os.path.join(
        output_dir,
        "candidate_A.jpg"
    )

    candidate_b_path = os.path.join(
        output_dir,
        "candidate_B.jpg"
    )

    candidate_c_path = os.path.join(
        output_dir,
        "candidate_C.jpg"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print("FREE MEF")
    print("THREE-CANDIDATE FUSION PIPELINE")
    print("=" * 70)

    print()
    print("INPUT IMAGES")
    print("-" * 70)

    print(
        "Underexposed :",
        under_path
    )

    print(
        "Normal       :",
        normal_path
    )

    print(
        "Overexposed  :",
        over_path
    )

    print()
    print("OUTPUT DIRECTORY")
    print("-" * 70)

    print(
        output_dir
    )

    # ========================================================
    # CANDIDATE A
    # ========================================================

    candidate_a = generate_candidate_a(
        under_path,
        normal_path,
        over_path,
        candidate_a_path
    )

    # ========================================================
    # CANDIDATE B
    # ========================================================

    candidate_b = generate_candidate_b(
        under_path,
        normal_path,
        over_path,
        candidate_b_path
    )

    # ========================================================
    # CANDIDATE C
    # ========================================================

    candidate_c = generate_candidate_c(
        under_path,
        normal_path,
        over_path,
        candidate_c_path,
        use_cuda=use_cuda
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "A": candidate_a,

        "B": candidate_b,

        "C": candidate_c

    }

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print("=" * 70)
    print("THREE-CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    for label, path in results.items():

        size_mb = (
            os.path.getsize(path)
            /
            (1024.0 * 1024.0)
        )

        print()
        print(
            f"Candidate {label}"
        )

        print(
            f"  {path}"
        )

        print(
            f"  Size: {size_mb:.2f} MB"
        )

    print()
    print("=" * 70)
    print("NEXT STAGE:")
    print("Evaluate Candidate A / B / C")
    print("=" * 70)

    return results


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Generate three fusion candidates "
            "from underexposed, normal and "
            "overexposed images."
        )

    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    parser.add_argument(

        "--under",

        required=True,

        help=(
            "Path to underexposed image"
        )

    )

    parser.add_argument(

        "--normal",

        required=True,

        help=(
            "Path to normal exposure image"
        )

    )

    parser.add_argument(

        "--over",

        required=True,

        help=(
            "Path to overexposed image"
        )

    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(

        "--output-dir",

        default=None,

        help=(
            "Directory where candidate images "
            "will be saved. "
            "Default: adaptive_outputs"
        )

    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    parser.add_argument(

        "--cuda",

        action="store_true",

        help=(
            "Use CUDA for MEF-Net when available."
        )

    )

    args = parser.parse_args()

    # ========================================================
    # RUN
    # ========================================================

    try:

        results = generate_candidates(

            under_path=args.under,

            normal_path=args.normal,

            over_path=args.over,

            output_dir=args.output_dir,

            use_cuda=args.cuda

        )

        print()
        print(
            "SUCCESS"
        )

        print()
        print(
            "Generated candidates:"
        )

        for label, path in results.items():

            print(
                f"Candidate {label}: {path}"
            )

    except Exception as exc:

        print()
        print("=" * 70)
        print("THREE-CANDIDATE GENERATION FAILED")
        print("=" * 70)

        print()
        print(
            f"Error: {exc}"
        )

        print()
        print(
            "Full traceback:"
        )

        traceback.print_exc()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
