import os
import cv2
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

EPS = 1e-8

# Change this if you want to process another scene
INPUT_DIR = os.path.expanduser(
    "~/FreeMEF/my_test/scene006"
)

OUTPUT_DIR = os.path.expanduser(
    "~/FreeMEF/production/output/adaptive_fusion_v3_9"
)


# ==========================================================
# IMAGE SEARCH
# ==========================================================

def find_image(folder, names):

    for name in names:

        path = os.path.join(
            folder,
            name
        )

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find image in {folder}\n"
        f"Tried: {names}"
    )


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return (
        image.astype(np.float32)
        / 255.0
    )


# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_image(path, image):

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image_u8 = np.uint8(
        image * 255.0
    )

    image_bgr = cv2.cvtColor(
        image_u8,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )


# ==========================================================
# SAVE DIAGNOSTIC MAP
# ==========================================================

def save_map(path, image):

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image_u8 = np.uint8(
        image * 255.0
    )

    cv2.imwrite(
        path,
        image_u8
    )


# ==========================================================
# LUMINANCE
# ==========================================================

def luminance(image):

    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )


# ==========================================================
# SMOOTHSTEP
# ==========================================================

def smoothstep(x, edge0, edge1):

    t = np.clip(
        (x - edge0)
        / (edge1 - edge0 + EPS),
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


# ==========================================================
# EXPOSURE QUALITY
# ==========================================================

def exposure_quality(lum):

    # Pixels around middle exposure
    # receive the highest quality score.

    sigma = 0.25

    quality = np.exp(
        -(
            (lum - 0.5) ** 2
        )
        /
        (2.0 * sigma * sigma)
    )

    return quality.astype(
        np.float32
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION V3.9")
    print("BRIGHTNESS RETENTION + DEPTH PRESERVATION")
    print("====================================")
    print()

    # ------------------------------------------------------
    # FIND INPUT IMAGES
    # ------------------------------------------------------

    under_path = find_image(
        INPUT_DIR,
        [
            "01_under.JPG",
            "01_under.jpg",
            "01_under.PNG",
            "01_under.png",
            "under.JPG",
            "under.jpg",
            "under.png"
        ]
    )

    normal_path = find_image(
        INPUT_DIR,
        [
            "02_normal.JPG",
            "02_normal.jpg",
            "02_normal.PNG",
            "02_normal.png",
            "normal.JPG",
            "normal.jpg",
            "normal.png"
        ]
    )

    over_path = find_image(
        INPUT_DIR,
        [
            "03_over.JPG",
            "03_over.jpg",
            "03_over.PNG",
            "03_over.png",
            "over.JPG",
            "over.jpg",
            "over.png"
        ]
    )

    print(
        f"Under : {under_path}"
    )

    print(
        f"Normal: {normal_path}"
    )

    print(
        f"Over  : {over_path}"
    )

    print()

    # ------------------------------------------------------
    # LOAD IMAGES
    # ------------------------------------------------------

    under = load_image(
        under_path
    )

    normal = load_image(
        normal_path
    )

    over = load_image(
        over_path
    )

    # ------------------------------------------------------
    # MATCH RESOLUTION
    # ------------------------------------------------------

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

    print(
        f"Resolution: {w} x {h}"
    )

    # ======================================================
    # LUMINANCE
    # ======================================================

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    lum_over = luminance(
        over
    )

    # ======================================================
    # HIGHLIGHT DETECTION
    #
    # Keep the original V3.3 highlight behavior.
    # This is responsible for the brightness/highlight
    # result you already liked.
    # ======================================================

    highlight_mask = smoothstep(
        lum_normal,
        0.55,
        0.82
    )

    # ======================================================
    # DEPTH-PRESERVING SHADOW DETECTION
    #
    # IMPORTANT:
    #
    # V3.3 used:
    #     0.18 -> 0.55
    #
    # This included medium-dark tones and lifted them,
    # reducing depth.
    #
    # V3.9 restricts shadow recovery to genuinely
    # dark areas.
    # ======================================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.08,
            0.30
        )
    )

    # ======================================================
    # DEEP SHADOW DETECTION
    #
    # Only very dark/crushed shadows get stronger
    # recovery.
    # ======================================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.03,
            0.16
        )
    )

    # ------------------------------------------------------
    # COMBINE SHADOW MASKS
    # ------------------------------------------------------

    shadow_recovery_mask = np.maximum(
        shadow_mask,
        deep_shadow_mask
    )

    # ======================================================
    # CHECK WHETHER OVEREXPOSED IMAGE
    # CONTAINS USEFUL INFORMATION
    # ======================================================

    over_detail_mask = smoothstep(
        lum_over,
        0.05,
        0.20
    )

    shadow_recovery_mask = (
        shadow_recovery_mask *
        over_detail_mask
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ======================================================
    # PATCH-FREE MASK SMOOTHING
    #
    # ONLY masks are smoothed.
    # Original image pixels remain sharp.
    #
    # Reduced smoothing prevents recovery from spreading
    # into medium-tone areas.
    # ======================================================

    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(
            np.float32
        ),
        (0, 0),
        12
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask.astype(
            np.float32
        ),
        (0, 0),
        10
    )

    deep_shadow_mask = cv2.GaussianBlur(
        deep_shadow_mask.astype(
            np.float32
        ),
        (0, 0),
        12
    )

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask.astype(
            np.float32
        ),
        (0, 0),
        10
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ======================================================
    # EXPOSURE QUALITY
    # ======================================================

    q_under = exposure_quality(
        lum_under
    )

    q_normal = exposure_quality(
        lum_normal
    )

    q_over = exposure_quality(
        lum_over
    )

    # ======================================================
    # UNDEREXPOSED IMAGE WEIGHT
    #
    # Keep original V3.3 behavior.
    # ======================================================

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    # ======================================================
    # NORMAL IMAGE WEIGHT
    #
    # Increased normal dominance.
    #
    # This preserves:
    # - brightness
    # - natural depth
    # - color
    # - white balance
    # - medium-tone contrast
    # ======================================================

    w_normal = (
        3.2 +
        1.8 * q_normal -
        0.35 * deep_shadow_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.5
    )

    # ======================================================
    # OVEREXPOSED IMAGE WEIGHT
    #
    # CRITICAL CHANGE:
    #
    # In V3.3:
    #     + 0.7 * q_over
    #
    # This allowed the overexposed image to influence
    # areas outside genuine shadows.
    #
    # Now q_over is restricted by the shadow mask.
    # ======================================================

    w_over = (
        0.02 +
        1.6 * shadow_recovery_mask +
        0.15 *
        q_over *
        shadow_recovery_mask
    )

    # ======================================================
    # WEIGHT SMOOTHING
    #
    # Smooth transitions only.
    # Images are NOT blurred.
    # ======================================================

    w_under = cv2.GaussianBlur(
        w_under.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    w_normal = cv2.GaussianBlur(
        w_normal.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    w_over = cv2.GaussianBlur(
        w_over.astype(
            np.float32
        ),
        (0, 0),
        12
    )

    # ======================================================
    # PREVENT NEGATIVE WEIGHTS
    # ======================================================

    w_under = np.maximum(
        w_under,
        0.0
    )

    w_normal = np.maximum(
        w_normal,
        0.0
    )

    w_over = np.maximum(
        w_over,
        0.0
    )

    # ======================================================
    # NORMALIZE WEIGHTS
    # ======================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under = (
        w_under /
        weight_sum
    )

    w_normal = (
        w_normal /
        weight_sum
    )

    w_over = (
        w_over /
        weight_sum
    )

    # ======================================================
    # HDR FUSION
    #
    # Original images remain completely sharp.
    # ======================================================

    fused = (

        under *
        w_under[:, :, None]

        +

        normal *
        w_normal[:, :, None]

        +

        over *
        w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # VERY SUBTLE NORMAL-EXPOSURE ANCHOR
    #
    # Keeps the excellent V3.3 brightness and color
    # appearance while preventing excessive tonal drift.
    #
    # This is intentionally subtle.
    # ======================================================

    normal_protection = smoothstep(
        lum_normal,
        0.20,
        0.65
    )

    protection_strength = (
        0.08 *
        normal_protection
    )

    fused = (

        fused *
        (
            1.0 -
            protection_strength[:, :, None]
        )

        +

        normal *
        protection_strength[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # CREATE OUTPUT DIRECTORY
    # ======================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ======================================================
    # SAVE DIAGNOSTICS
    # ======================================================

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "01_highlight_mask.png"
        ),
        highlight_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "02_shadow_mask.png"
        ),
        shadow_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "03_deep_shadow_mask.png"
        ),
        deep_shadow_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "04_shadow_recovery_mask.png"
        ),
        shadow_recovery_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "05_weight_under.png"
        ),
        w_under
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "06_weight_normal.png"
        ),
        w_normal
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "07_weight_over.png"
        ),
        w_over
    )

    # ======================================================
    # SAVE FINAL IMAGE
    # ======================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "FINAL_ADAPTIVE_FUSION_V3_9.jpg"
    )

    save_image(
        output_path,
        fused
    )

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION V3.9 COMPLETED")
    print("====================================")
    print()

    print(
        f"Output: {output_path}"
    )

    print()

    print(
        "Mean fusion weights:"
    )

    print(
        f"Under : {np.mean(w_under):.4f}"
    )

    print(
        f"Normal: {np.mean(w_normal):.4f}"
    )

    print(
        f"Over  : {np.mean(w_over):.4f}"
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "Under  -> highlight recovery"
    )

    print(
        "Normal -> brightness, color and depth anchor"
    )

    print(
        "Over   -> deep shadow recovery only"
    )

    print()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
