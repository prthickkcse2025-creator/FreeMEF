#!/usr/bin/env python3

"""
MERTENS EXPOSURE FUSION ADAPTER

Candidate 2 for the adaptive FreeMEF system.

Uses the real Mertens/Laplacian-pyramid repository from:

    adaptive_modules/mertens/ExposureFusion_py3/

The upstream repository is NOT modified by this adapter.

Workflow:

    Original 3 exposures
            ↓
    Adaptive working-resolution resize
            ↓
    Real Mertens exposure fusion
            ↓
    Resize result back to original resolution
            ↓
    Save candidate output
"""

import os
import sys
import shutil
import tempfile

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

REPO_DIR = os.path.join(
    PROJECT_ROOT,
    "adaptive_modules",
    "mertens",
    "ExposureFusion_py3"
)


# ============================================================
# SETTINGS
# ============================================================

# The original Mertens implementation performs its contrast
# calculation using Python loops, so full 6000x4000 processing
# is extremely slow.
#
# We therefore use a smaller working image and restore the
# result to the original resolution afterward.

MAX_WORKING_WIDTH = 1600

PYRAMID_HEIGHT = 6

CONTRAST_POWER = 1.0
SATURATION_POWER = 1.0
EXPOSURE_POWER = 1.0


# ============================================================
# READ IMAGE
# ============================================================

def read_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return image


# ============================================================
# RESIZE FOR MERTENS
# ============================================================

def resize_for_processing(
    image,
    max_width
):

    h, w = image.shape[:2]

    if w <= max_width:

        return image.copy()

    scale = (
        max_width /
        float(w)
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# IMPORT REAL REPOSITORY
# ============================================================

def import_repository():

    if not os.path.isdir(
        REPO_DIR
    ):

        raise FileNotFoundError(
            "Mertens repository copy not found:\n"
            f"{REPO_DIR}"
        )

    if REPO_DIR not in sys.path:

        sys.path.insert(
            0,
            REPO_DIR
        )

    try:

        import laplacianfusion

    except Exception as exc:

        raise RuntimeError(
            "Could not import the Python-3 "
            "Mertens repository.\n"
            "Check the ExposureFusion_py3 copy."
        ) from exc

    return laplacianfusion


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    path,
    image
):

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    # Mertens repository returns RGB.
    image_rgb = np.uint8(
        image * 255.0
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    ok = cv2.imwrite(
        path,
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    if not ok:

        raise RuntimeError(
            f"Could not save:\n{path}"
        )


# ============================================================
# RUN MERTENS
# ============================================================

def run_mertens(
    under_path,
    normal_path,
    over_path,
    output_path,
    max_working_width=MAX_WORKING_WIDTH
):

    print()
    print("=" * 60)
    print("CANDIDATE 2: REAL MERTENS EXPOSURE FUSION")
    print("=" * 60)

    print()
    print(
        f"Repository:\n{REPO_DIR}"
    )

    print()
    print("Original inputs:")

    print(
        f"  Under  : {under_path}"
    )

    print(
        f"  Normal : {normal_path}"
    )

    print(
        f"  Over   : {over_path}"
    )

    # ========================================================
    # LOAD ORIGINAL IMAGES
    # ========================================================

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    original_h, original_w = (
        normal.shape[:2]
    )

    print()
    print(
        f"Original resolution: "
        f"{original_w} x {original_h}"
    )

    # ========================================================
    # MATCH NORMAL RESOLUTION
    # ========================================================

    if under.shape[:2] != (
        original_h,
        original_w
    ):

        under = cv2.resize(
            under,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (
        original_h,
        original_w
    ):

        over = cv2.resize(
            over,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # ========================================================
    # DOWN-SCALE FOR OLD MERTENS IMPLEMENTATION
    # ========================================================

    under_small = resize_for_processing(
        under,
        max_working_width
    )

    normal_small = resize_for_processing(
        normal,
        max_working_width
    )

    over_small = resize_for_processing(
        over,
        max_working_width
    )

    work_h, work_w = (
        normal_small.shape[:2]
    )

    print(
        f"Working resolution: "
        f"{work_w} x {work_h}"
    )

    print(
        f"Working width limit: "
        f"{max_working_width}"
    )

    # ========================================================
    # IMPORT REAL MERTENS CODE
    # ========================================================

    laplacianfusion = (
        import_repository()
    )

    # ========================================================
    # TEMPORARY REPOSITORY IMAGE SET
    # ========================================================

    temp_root = tempfile.mkdtemp(
        prefix="mertens_run_",
        dir=os.path.join(
            PROJECT_ROOT,
            "adaptive_outputs"
        )
    )

    image_set_dir = os.path.join(
        temp_root,
        "image_set",
        "jpeg"
    )

    os.makedirs(
        image_set_dir,
        exist_ok=True
    )

    original_cwd = os.getcwd()

    try:

        # ----------------------------------------------------
        # Save working images.
        #
        # They are written as RGB-compatible JPEGs because
        # the repository loads them and converts to RGB.
        # ----------------------------------------------------

        working_images = [

            (
                "01_under.jpg",
                under_small
            ),

            (
                "02_normal.jpg",
                normal_small
            ),

            (
                "03_over.jpg",
                over_small
            )

        ]

        for filename, image_bgr in (
            working_images
        ):

            destination = os.path.join(
                image_set_dir,
                filename
            )

            ok = cv2.imwrite(
                destination,
                image_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    100
                ]
            )

            if not ok:

                raise RuntimeError(
                    f"Could not create "
                    f"temporary image:\n"
                    f"{destination}"
                )

        # ----------------------------------------------------
        # Change into temp root.
        #
        # The upstream Image class expects:
        #
        # image_set/jpeg/<filename>
        # ----------------------------------------------------

        os.chdir(
            temp_root
        )

        names = [

            "01_under.jpg",
            "02_normal.jpg",
            "03_over.jpg"

        ]

        print()
        print(
            "Running real Mertens "
            "Laplacian fusion..."
        )

        # ----------------------------------------------------
        # REAL REPOSITORY CALL
        # ----------------------------------------------------

        lap = (
            laplacianfusion.LaplacianMap(
                "jpeg",
                names,
                n=PYRAMID_HEIGHT
            )
        )

        result = (
            lap.result_exposure(
                CONTRAST_POWER,
                SATURATION_POWER,
                EXPOSURE_POWER
            )
        )

    finally:

        os.chdir(
            original_cwd
        )

    # ========================================================
    # CLEANUP TEMPORARY DATA
    # ========================================================

    shutil.rmtree(
        temp_root,
        ignore_errors=True
    )

    # ========================================================
    # UPSCALE RESULT TO ORIGINAL RESOLUTION
    # ========================================================

    result = np.asarray(
        result,
        dtype=np.float32
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    result = cv2.resize(
        result,
        (
            original_w,
            original_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_image(
        output_path,
        result
    )

    print()
    print(
        "Mertens candidate completed."
    )

    print(
        f"Final output:\n"
        f"{output_path}"
    )

    return output_path


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    base = os.path.expanduser(
        "~/FreeMEF"
    )

    under_path = os.path.join(
        base,
        "my_test",
        "scene005",
        "01_under.jpg"
    )

    normal_path = os.path.join(
        base,
        "my_test",
        "scene005",
        "02_normal.jpg"
    )

    over_path = os.path.join(
        base,
        "my_test",
        "scene005",
        "03_over.jpg"
    )

    output_path = os.path.join(
        base,
        "adaptive_outputs",
        "mertens_scene005.jpg"
    )

    run_mertens(
        under_path,
        normal_path,
        over_path,
        output_path
    )
