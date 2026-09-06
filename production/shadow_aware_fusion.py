import os
import cv2
import numpy as np

# ============================================================
# SHADOW-AWARE HDR FUSION
# ============================================================
# Input:
#   UNDER  -> dark exposure
#   NORMAL -> reference exposure
#   OVER   -> bright exposure
#
# Strategy:
#   1. Detect shadow/dark regions from NORMAL image
#   2. Increase OVER exposure contribution only in shadows
#   3. Preserve NORMAL exposure in midtones
#   4. Use UNDER exposure for highlight protection
#   5. Use soft masks to avoid artifacts and halos
# ============================================================


# ============================================================
# PATHS
# Change these three filenames for each test
# ============================================================

UNDER_PATH = "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/0J0A9350.jpg"
NORMAL_PATH = "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/0J0A9351.jpg"
OVER_PATH = "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/0J0A9352.jpg"

OUTPUT_DIR = "production/output/shadow_aware"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "0J0A9350_9351_9352_shadow_aware.png"
)


# ============================================================
# PARAMETERS
# ============================================================

# Shadow detection threshold
SHADOW_THRESHOLD = 0.42

# Highlight detection threshold
HIGHLIGHT_THRESHOLD = 0.75

# Gaussian blur for smooth masks
MASK_BLUR = 51

# How strongly the bright exposure is used in shadows
SHADOW_OVER_BOOST = 2.5

# Normal exposure base importance
NORMAL_WEIGHT = 1.0

# Under exposure contribution for highlight protection
UNDER_HIGHLIGHT_BOOST = 2.0

# Small value to prevent divide-by-zero
EPS = 1e-8


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_image(path):
    """
    Load image as float32 RGB in range [0, 1].
    """

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"\nERROR: Could not load image:\n{path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image.astype(np.float32) / 255.0


def save_image(path, image):
    """
    Save float RGB image.
    """

    image = np.clip(image, 0, 1)

    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, image)


def get_luminance(image):
    """
    Calculate perceptual luminance.
    """

    return (
        0.2126 * image[:, :, 0] +
        0.7152 * image[:, :, 1] +
        0.0722 * image[:, :, 2]
    )


def smoothstep(edge0, edge1, x):
    """
    Smooth transition between 0 and 1.
    """

    t = np.clip(
        (x - edge0) / (edge1 - edge0 + EPS),
        0.0,
        1.0
    )

    return t * t * (3.0 - 2.0 * t)


# ============================================================
# SHADOW MASK
# ============================================================

def create_shadow_mask(normal_image):

    luminance = get_luminance(normal_image)

    # --------------------------------------------------------
    # Dark pixels get high values
    #
    # luminance < threshold
    #       -> shadow mask close to 1
    #
    # luminance > threshold
    #       -> shadow mask close to 0
    # --------------------------------------------------------

    shadow_mask = 1.0 - smoothstep(
        SHADOW_THRESHOLD * 0.5,
        SHADOW_THRESHOLD,
        luminance
    )

    # Smooth the mask
    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (MASK_BLUR, MASK_BLUR),
        0
    )

    shadow_mask = np.clip(
        shadow_mask,
        0.0,
        1.0
    )

    return shadow_mask


# ============================================================
# HIGHLIGHT MASK
# ============================================================

def create_highlight_mask(normal_image):

    luminance = get_luminance(normal_image)

    # Bright regions get high values
    highlight_mask = smoothstep(
        HIGHLIGHT_THRESHOLD,
        0.95,
        luminance
    )

    # Smooth mask
    highlight_mask = cv2.GaussianBlur(
        highlight_mask,
        (MASK_BLUR, MASK_BLUR),
        0
    )

    highlight_mask = np.clip(
        highlight_mask,
        0.0,
        1.0
    )

    return highlight_mask


# ============================================================
# WELL-EXPOSEDNESS WEIGHT
# ============================================================

def exposure_quality(image):

    luminance = get_luminance(image)

    # Pixels around middle brightness are preferred
    sigma = 0.25

    weight = np.exp(
        -((luminance - 0.5) ** 2) /
        (2 * sigma * sigma)
    )

    return weight


# ============================================================
# MAIN SHADOW-AWARE FUSION
# ============================================================

def shadow_aware_fusion(under, normal, over):

    # ========================================================
    # STEP 1: CREATE MASKS
    # ========================================================

    shadow_mask = create_shadow_mask(normal)

    highlight_mask = create_highlight_mask(normal)


    # ========================================================
    # STEP 2: EXPOSURE QUALITY
    # ========================================================

    under_quality = exposure_quality(under)

    normal_quality = exposure_quality(normal)

    over_quality = exposure_quality(over)


    # ========================================================
    # STEP 3: BASE WEIGHTS
    # ========================================================

    under_weight = under_quality.copy()

    normal_weight = normal_quality.copy()

    over_weight = over_quality.copy()


    # ========================================================
    # STEP 4: SHADOW RECOVERY
    #
    # In dark regions:
    #   Increase contribution from OVER exposure
    #
    # Normal remains available so color/appearance is preserved.
    # ========================================================

    over_weight = over_weight * (
        1.0 +
        SHADOW_OVER_BOOST * shadow_mask
    )

    # Preserve normal exposure as reference
    normal_weight = normal_weight * NORMAL_WEIGHT


    # ========================================================
    # STEP 5: HIGHLIGHT PROTECTION
    #
    # In bright regions:
    #   Increase UNDER exposure contribution
    # ========================================================

    under_weight = under_weight * (
        1.0 +
        UNDER_HIGHLIGHT_BOOST * highlight_mask
    )


    # ========================================================
    # STEP 6: REDUCE WRONG EXPOSURE CONTRIBUTION
    #
    # Avoid using OVER image too much in highlights
    # Avoid using UNDER image too much in deep shadows
    # ========================================================

    over_weight = over_weight * (
        1.0 - 0.85 * highlight_mask
    )

    under_weight = under_weight * (
        1.0 - 0.75 * shadow_mask
    )


    # ========================================================
    # STEP 7: ADD MINIMUM NORMAL WEIGHT
    #
    # This makes NORMAL the visual/color anchor.
    # ========================================================

    normal_weight = normal_weight + 0.15


    # ========================================================
    # STEP 8: NORMALIZE WEIGHTS
    # ========================================================

    total_weight = (
        under_weight +
        normal_weight +
        over_weight +
        EPS
    )

    under_weight = under_weight / total_weight

    normal_weight = normal_weight / total_weight

    over_weight = over_weight / total_weight


    # ========================================================
    # STEP 9: EXPAND WEIGHTS TO RGB
    # ========================================================

    under_w = under_weight[:, :, np.newaxis]

    normal_w = normal_weight[:, :, np.newaxis]

    over_w = over_weight[:, :, np.newaxis]


    # ========================================================
    # STEP 10: FUSE
    # ========================================================

    fused = (
        under * under_w +
        normal * normal_w +
        over * over_w
    )


    # ========================================================
    # STEP 11: RETURN EVERYTHING
    # ========================================================

    return (
        fused,
        shadow_mask,
        highlight_mask,
        under_weight,
        normal_weight,
        over_weight
    )


# ============================================================
# CREATE VISUALIZATION MAP
# ============================================================

def save_mask(path, mask):

    mask_image = np.clip(
        mask * 255,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(path, mask_image)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("SHADOW-AWARE HDR FUSION")
    print("=" * 65)

    print("\nUNDER : ", UNDER_PATH)
    print("NORMAL: ", NORMAL_PATH)
    print("OVER  : ", OVER_PATH)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading images...")

    under = load_image(UNDER_PATH)

    normal = load_image(NORMAL_PATH)

    over = load_image(OVER_PATH)


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    print(
        "\nResolution:",
        normal.shape
    )

    if (
        under.shape != normal.shape or
        over.shape != normal.shape
    ):

        raise ValueError(
            "\nERROR: All three images must have "
            "the same resolution."
        )


    # --------------------------------------------------------
    # Run fusion
    # --------------------------------------------------------

    print("\nDetecting shadow regions...")
    print("Recovering shadows using bright exposure...")
    print("Protecting highlights using dark exposure...")

    (
        fused,
        shadow_mask,
        highlight_mask,
        under_weight,
        normal_weight,
        over_weight
    ) = shadow_aware_fusion(
        under,
        normal,
        over
    )


    # --------------------------------------------------------
    # Save final output
    # --------------------------------------------------------

    save_image(
        OUTPUT_PATH,
        fused
    )


    # --------------------------------------------------------
    # Save diagnostic maps
    # --------------------------------------------------------

    save_mask(
        os.path.join(
            OUTPUT_DIR,
            "shadow_mask.png"
        ),
        shadow_mask
    )

    save_mask(
        os.path.join(
            OUTPUT_DIR,
            "highlight_mask.png"
        ),
        highlight_mask
    )

    save_mask(
        os.path.join(
            OUTPUT_DIR,
            "under_weight.png"
        ),
        under_weight
    )

    save_mask(
        os.path.join(
            OUTPUT_DIR,
            "normal_weight.png"
        ),
        normal_weight
    )

    save_mask(
        os.path.join(
            OUTPUT_DIR,
            "over_weight.png"
        ),
        over_weight
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("MEAN BRIGHTNESS")
    print("-" * 65)

    print(
        f"Under : {get_luminance(under).mean():.4f}"
    )

    print(
        f"Normal: {get_luminance(normal).mean():.4f}"
    )

    print(
        f"Over  : {get_luminance(over).mean():.4f}"
    )

    print(
        f"Final : {get_luminance(fused).mean():.4f}"
    )


    print("\n" + "-" * 65)
    print("MEAN FUSION WEIGHTS")
    print("-" * 65)

    print(
        f"Under weight : {under_weight.mean():.4f}"
    )

    print(
        f"Normal weight: {normal_weight.mean():.4f}"
    )

    print(
        f"Over weight  : {over_weight.mean():.4f}"
    )


    shadow_coverage = (
        (shadow_mask > 0.5).mean()
        * 100
    )

    print(
        f"\nShadow coverage: "
        f"{shadow_coverage:.2f} %"
    )


    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)

    print(
        "\nFinal output:"
    )

    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )

    print(
        "\nDiagnostic files:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()

