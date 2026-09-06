import cv2
import numpy as np
import os
import glob


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = (
    "/mnt/e/HDR_Project_Backups/"
    "HDR_BASELINE_V10_FINAL/"
    "raw_scenes"
)

OUTPUT_DIR = "production/output/batch_hdr"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HDR FUSION FUNCTION
# ============================================================

def fuse_hdr(under, normal, over):

    # --------------------------------------------------------
    # CONVERT TO FLOAT
    # --------------------------------------------------------

    under_f = under.astype(np.float32) / 255.0
    normal_f = normal.astype(np.float32) / 255.0
    over_f = over.astype(np.float32) / 255.0


    # --------------------------------------------------------
    # CALCULATE LUMINANCE
    # --------------------------------------------------------

    normal_gray = cv2.cvtColor(
        normal_f,
        cv2.COLOR_BGR2GRAY
    )

    under_gray = cv2.cvtColor(
        under_f,
        cv2.COLOR_BGR2GRAY
    )

    over_gray = cv2.cvtColor(
        over_f,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # SHADOW RECOVERY
    #
    # Use OVER exposure only in darker regions
    # ========================================================

    shadow_mask = np.clip(
        (0.40 - normal_gray) / 0.40,
        0.0,
        1.0
    )

    # Focus more strongly on genuinely dark areas
    shadow_mask = shadow_mask ** 1.8

    # Smooth mask boundaries
    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (0, 0),
        sigmaX=5,
        sigmaY=5
    )

    # Maximum contribution from OVER exposure
    shadow_weight = shadow_mask * 0.35


    # --------------------------------------------------------
    # CALCULATE BRIGHTNESS GAIN
    # --------------------------------------------------------

    shadow_gain = (
        over_gray /
        (normal_gray + 1e-4)
    )

    # Prevent unrealistic brightness amplification
    shadow_gain = np.clip(
        shadow_gain,
        1.0,
        1.8
    )

    brightness_gain = (
        1.0
        + (shadow_gain - 1.0)
        * shadow_weight
    )


    # --------------------------------------------------------
    # APPLY SHADOW RECOVERY
    # --------------------------------------------------------

    result = (
        normal_f
        * brightness_gain[:, :, None]
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )


    # ========================================================
    # HIGHLIGHT RECOVERY
    #
    # Use UNDER exposure for bright / clipped regions
    # ========================================================

    result_gray = cv2.cvtColor(
        result.astype(np.float32),
        cv2.COLOR_BGR2GRAY
    )

    highlight_mask = np.clip(
        (result_gray - 0.72) / 0.25,
        0.0,
        1.0
    )

    highlight_mask = highlight_mask ** 1.5

    highlight_mask = cv2.GaussianBlur(
        highlight_mask,
        (0, 0),
        sigmaX=4,
        sigmaY=4
    )

    # Maximum UNDER exposure contribution
    highlight_weight = (
        highlight_mask * 0.45
    )


    # --------------------------------------------------------
    # APPLY HIGHLIGHT RECOVERY
    # --------------------------------------------------------

    result = (
        result
        * (1.0 - highlight_weight[:, :, None])
        + under_f
        * highlight_weight[:, :, None]
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )


    # ========================================================
    # COLOR / WHITE BALANCE ANCHOR
    #
    # Pull slightly toward NORMAL exposure
    # ========================================================

    color_anchor = 0.08

    result = (
        result * (1.0 - color_anchor)
        + normal_f * color_anchor
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )


    # ========================================================
    # CONVERT BACK TO UINT8
    # ========================================================

    result_u8 = np.round(
        result * 255.0
    ).astype(np.uint8)

    return result_u8


# ============================================================
# FIND ALL IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.png",
    "*.PNG"
]

files = []

for extension in extensions:

    found_files = glob.glob(
        os.path.join(
            INPUT_DIR,
            extension
        )
    )

    files.extend(found_files)


# Sort by filename
files = sorted(files)


# ============================================================
# REPORT
# ============================================================

print("=" * 60)
print("TOTAL IMAGES FOUND:", len(files))
print("=" * 60)


# Check if image count is divisible by 3
if len(files) % 3 != 0:

    print()
    print("WARNING!")
    print(
        "Image count is not perfectly divisible by 3."
    )

    print(
        "Extra images:",
        len(files) % 3
    )


# Number of complete HDR groups
scene_count = len(files) // 3


print()
print("TOTAL HDR GROUPS:", scene_count)
print()


# ============================================================
# PROCESS ALL HDR GROUPS
# ============================================================

for i in range(scene_count):


    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{i + 1:03d}.png"
    )


    # --------------------------------------------------------
    # RESUME FEATURE
    #
    # Skip scenes already generated
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print(
            f"SKIPPING COMPLETED SCENE "
            f"{i + 1}/{scene_count}"
        )

        continue


    # --------------------------------------------------------
    # GET THREE IMAGES
    # --------------------------------------------------------

    start = i * 3

    under_path = files[start]

    normal_path = files[start + 1]

    over_path = files[start + 2]


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"PROCESSING SCENE "
        f"{i + 1}/{scene_count}"
    )

    print(
        "UNDER :",
        os.path.basename(under_path)
    )

    print(
        "NORMAL:",
        os.path.basename(normal_path)
    )

    print(
        "OVER  :",
        os.path.basename(over_path)
    )


    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    under = cv2.imread(
        under_path,
        cv2.IMREAD_COLOR
    )

    normal = cv2.imread(
        normal_path,
        cv2.IMREAD_COLOR
    )

    over = cv2.imread(
        over_path,
        cv2.IMREAD_COLOR
    )


    # --------------------------------------------------------
    # CHECK IMAGE LOADING
    # --------------------------------------------------------

    if (
        under is None
        or normal is None
        or over is None
    ):

        print(
            "ERROR: Could not load one "
            "or more images."
        )

        continue


    # --------------------------------------------------------
    # ENSURE SAME RESOLUTION
    # --------------------------------------------------------

    h, w = normal.shape[:2]


    if under.shape[:2] != (h, w):

        under = cv2.resize(
            under,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )


    if over.shape[:2] != (h, w):

        over = cv2.resize(
            over,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )


    # --------------------------------------------------------
    # RUN HDR FUSION
    # --------------------------------------------------------

    try:

        result = fuse_hdr(
            under,
            normal,
            over
        )


        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        success = cv2.imwrite(
            output_path,
            result
        )


        if success:

            print(
                "SAVED:",
                output_path
            )

        else:

            print(
                "ERROR: Failed to save:",
                output_path
            )


    except Exception as error:

        print(
            "ERROR PROCESSING SCENE",
            i + 1
        )

        print(error)

        continue


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)

print(
    "BATCH HDR PROCESSING COMPLETE"
)

print(
    "OUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)

print("=" * 60)
