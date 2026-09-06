import os
import cv2
import numpy as np


# ============================================================
# ADAPTIVE HDR FUSION V3.4
#
# NORMAL-ANCHOR / BRIGHTNESS-LOCKED FUSION
#
# Philosophy:
#
# FINAL = NORMAL IMAGE
#       + highlight detail from UNDER
#       + shadow detail from OVER
#
# The normal exposure remains the visual anchor.
#
# Under and over exposures DO NOT globally change:
#   - brightness
#   - white balance
#   - color appearance
#
# They are only allowed to contribute where the normal exposure
# has genuinely lost useful information.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "/home/prthick/FreeMEF/my_test/scene006"

UNDER_PATH = os.path.join(
    INPUT_DIR,
    "01_under.jpg"
)

NORMAL_PATH = os.path.join(
    INPUT_DIR,
    "02_normal.jpg"
)

OVER_PATH = os.path.join(
    INPUT_DIR,
    "03_over.jpg"
)

OUTPUT_DIR = (
    "/home/prthick/FreeMEF/"
    "production/output/"
    "adaptive_fusion_v3_4"
)

FINAL_NAME = (
    "FINAL_ADAPTIVE_FUSION_V3_4_"
    "BRIGHTNESS_LOCKED.jpg"
)


# ============================================================
# PARAMETERS
# ============================================================

EPS = 1e-6

# ------------------------------------------------------------
# Brightness preservation
#
# 0.0 = completely force normal brightness
# 1.0 = allow recovered HDR brightness completely
#
# We keep this low intentionally.
# ------------------------------------------------------------

BRIGHTNESS_ADAPTATION = 0.12


# ------------------------------------------------------------
# Maximum contribution from auxiliary exposures
#
# Prevents under/over from replacing the normal image.
# ------------------------------------------------------------

MAX_HIGHLIGHT_RECOVERY = 0.85
MAX_SHADOW_RECOVERY = 0.70


# ------------------------------------------------------------
# Mask smoothing
#
# ONLY masks are blurred.
# Actual image pixels are never blurred.
# ------------------------------------------------------------

HIGHLIGHT_MASK_SIGMA = 9
SHADOW_MASK_SIGMA = 9


# ------------------------------------------------------------
# Residual clipping
#
# Prevents extreme color or brightness injection.
# ------------------------------------------------------------

RESIDUAL_CLIP = 0.35


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def read_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    return image


def write_image(
    path,
    image
):

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image = (
        image * 255.0
    ).astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )


def save_map(
    path,
    image
):

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image = (
        image * 255.0
    ).astype(np.uint8)

    cv2.imwrite(
        path,
        image
    )


# ============================================================
# COLOR SPACE FUNCTIONS
# ============================================================

def srgb_to_linear(image):

    return np.where(

        image <= 0.04045,

        image / 12.92,

        (
            (image + 0.055)
            / 1.055
        ) ** 2.4
    )


def linear_to_srgb(image):

    image = np.maximum(
        image,
        0.0
    )

    return np.where(

        image <= 0.0031308,

        image * 12.92,

        (
            1.055
            * (
                image ** (
                    1.0 / 2.4
                )
            )
            - 0.055
        )
    )


def luminance(image):

    return (

        0.2126
        * image[:, :, 0]

        +

        0.7152
        * image[:, :, 1]

        +

        0.0722
        * image[:, :, 2]
    )


# ============================================================
# MASK FUNCTIONS
# ============================================================

def smoothstep(
    x,
    edge0,
    edge1
):

    t = np.clip(

        (
            x - edge0
        )

        /

        (
            edge1
            - edge0
            + EPS
        ),

        0.0,
        1.0
    )

    return (

        t
        * t
        * (
            3.0
            - 2.0
            * t
        )
    )


def gaussian_mask(
    mask,
    sigma
):

    mask = cv2.GaussianBlur(

        mask.astype(
            np.float32
        ),

        (0, 0),

        sigmaX=sigma,
        sigmaY=sigma
    )

    return np.clip(
        mask,
        0.0,
        1.0
    )


# ============================================================
# DETAIL / QUALITY DETECTION
# ============================================================

def gradient_strength(image):

    lum = luminance(
        image
    )

    gx = cv2.Sobel(

        lum,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(

        lum,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(

        gx * gx
        +
        gy * gy
    )

    return gradient


def local_detail_confidence(
    image
):

    gray = luminance(
        image
    )

    mean = cv2.GaussianBlur(

        gray,
        (0, 0),
        3
    )

    detail = np.abs(
        gray - mean
    )

    confidence = smoothstep(

        detail,
        0.003,
        0.030
    )

    return confidence


# ============================================================
# SATURATION / CLIPPING DETECTION
# ============================================================

def highlight_clipping_mask(
    image
):

    lum = luminance(
        image
    )

    channel_max = np.max(
        image,
        axis=2
    )

    channel_min = np.min(
        image,
        axis=2
    )

    # Brightness-based highlight detection
    bright = smoothstep(

        lum,
        0.72,
        0.94
    )

    # Detect channel clipping
    clipped = smoothstep(

        channel_max,
        0.94,
        0.995
    )

    # Low contrast in very bright region often means
    # highlight detail is disappearing.
    contrast = channel_max - channel_min

    low_detail = (

        1.0
        -
        smoothstep(
            contrast,
            0.015,
            0.10
        )
    )

    mask = np.maximum(

        bright * 0.70,

        clipped * 0.90
    )

    # Only slightly increase confidence where
    # bright region has low local detail.
    mask = np.maximum(

        mask,

        bright
        * low_detail
        * 0.45
    )

    return np.clip(
        mask,
        0.0,
        1.0
    )


def shadow_clipping_mask(
    image
):

    lum = luminance(
        image
    )

    channel_max = np.max(
        image,
        axis=2
    )

    dark = (

        1.0
        -
        smoothstep(
            lum,
            0.035,
            0.22
        )
    )

    clipped = (

        1.0
        -
        smoothstep(
            channel_max,
            0.025,
            0.10
        )
    )

    mask = np.maximum(

        dark * 0.80,

        clipped * 0.75
    )

    return np.clip(
        mask,
        0.0,
        1.0
    )


# ============================================================
# AUXILIARY IMAGE USEFULNESS
# ============================================================

def highlight_source_quality(
    under,
    normal
):

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    # Under image should not itself be too dark
    under_valid = smoothstep(

        lum_under,
        0.04,
        0.30
    )

    # It should contain visible local structure
    detail = local_detail_confidence(
        under
    )

    # Under exposure is useful if normal is brighter
    # and under retains information.
    exposure_difference = smoothstep(

        lum_normal
        - lum_under,

        0.03,
        0.30
    )

    quality = (

        under_valid
        * (
            0.35
            +
            0.65
            * detail
        )
        * exposure_difference
    )

    return np.clip(
        quality,
        0.0,
        1.0
    )


def shadow_source_quality(
    over,
    normal
):

    lum_over = luminance(
        over
    )

    lum_normal = luminance(
        normal
    )

    # Over image must contain usable visible pixels
    over_valid = smoothstep(

        lum_over,
        0.05,
        0.22
    )

    detail = local_detail_confidence(
        over
    )

    exposure_difference = smoothstep(

        lum_over
        - lum_normal,

        0.03,
        0.30
    )

    quality = (

        over_valid
        * (
            0.35
            +
            0.65
            * detail
        )
        * exposure_difference
    )

    return np.clip(
        quality,
        0.0,
        1.0
    )


# ============================================================
# LUMINANCE PRESERVATION
# ============================================================

def brightness_lock(
    fused,
    normal
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    # Calculate luminance ratio required
    # to restore normal brightness.
    scale = (

        normal_lum
        /
        (
            fused_lum
            + EPS
        )
    )

    # Do not allow extreme scaling.
    scale = np.clip(

        scale,
        0.50,
        2.00
    )

    locked = (
        fused
        * scale[:, :, None]
    )

    # IMPORTANT:
    #
    # Do not completely destroy recovered HDR information.
    #
    # Blend between:
    #   fused HDR result
    #   brightness-locked result
    #
    # Most of the output follows normal brightness.
    result = (

        fused
        * BRIGHTNESS_ADAPTATION

        +

        locked
        * (
            1.0
            -
            BRIGHTNESS_ADAPTATION
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# MAIN FUSION
# ============================================================

def adaptive_hdr_fusion(
    under,
    normal,
    over
):

    # --------------------------------------------------------
    # Convert to linear light.
    #
    # Fusion in linear space is more physically meaningful
    # and avoids gamma-space brightness artifacts.
    # --------------------------------------------------------

    under_lin = srgb_to_linear(
        under
    )

    normal_lin = srgb_to_linear(
        normal
    )

    over_lin = srgb_to_linear(
        over
    )


    # --------------------------------------------------------
    # NORMAL EXPOSURE IS THE BASE
    # --------------------------------------------------------

    result = normal_lin.copy()


    # ========================================================
    # HIGHLIGHT RECOVERY
    # ========================================================

    normal_highlight_mask = (
        highlight_clipping_mask(
            normal
        )
    )

    under_quality = (
        highlight_source_quality(
            under,
            normal
        )
    )

    highlight_mask = (

        normal_highlight_mask
        * under_quality
    )

    highlight_mask = gaussian_mask(

        highlight_mask,

        HIGHLIGHT_MASK_SIGMA
    )

    highlight_mask = np.clip(

        highlight_mask
        * MAX_HIGHLIGHT_RECOVERY,

        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Residual injection
    #
    # Instead of:
    #
    # result = mix(normal, under)
    #
    # we inject only the missing information:
    #
    # result += mask * (under - normal)
    #
    # This keeps normal exposure dominant.
    # --------------------------------------------------------

    highlight_residual = (

        under_lin
        - normal_lin
    )

    highlight_residual = np.clip(

        highlight_residual,

        -RESIDUAL_CLIP,
        RESIDUAL_CLIP
    )

    result = (

        result

        +

        highlight_mask[:, :, None]
        * highlight_residual
    )


    # ========================================================
    # SHADOW RECOVERY
    # ========================================================

    normal_shadow_mask = (
        shadow_clipping_mask(
            normal
        )
    )

    over_quality = (
        shadow_source_quality(
            over,
            normal
        )
    )

    shadow_mask = (

        normal_shadow_mask
        * over_quality
    )

    shadow_mask = gaussian_mask(

        shadow_mask,

        SHADOW_MASK_SIGMA
    )

    shadow_mask = np.clip(

        shadow_mask
        * MAX_SHADOW_RECOVERY,

        0.0,
        1.0
    )


    shadow_residual = (

        over_lin
        - normal_lin
    )

    shadow_residual = np.clip(

        shadow_residual,

        -RESIDUAL_CLIP,
        RESIDUAL_CLIP
    )

    result = (

        result

        +

        shadow_mask[:, :, None]
        * shadow_residual
    )


    # ========================================================
    # COLOR / CHROMA PROTECTION
    #
    # Prevent auxiliary exposures from shifting
    # the original normal-image color too aggressively.
    # ========================================================

    normal_lum_lin = luminance(
        normal_lin
    )

    result_lum_lin = luminance(
        result
    )

    normal_chroma = (

        normal_lin

        /

        (
            normal_lum_lin[:, :, None]
            + EPS
        )
    )

    # Preserve luminance from HDR fusion,
    # but keep chromatic relationship closer to normal.
    chroma_protected = (

        normal_chroma
        * result_lum_lin[:, :, None]
    )

    # Small amount of HDR color is allowed
    # to preserve recovered real detail.
    result = (

        result
        * 0.20

        +

        chroma_protected
        * 0.80
    )


    # ========================================================
    # CONVERT BACK TO DISPLAY SPACE
    # ========================================================

    result = linear_to_srgb(
        result
    )

    result = np.clip(

        result,
        0.0,
        1.0
    )


    # ========================================================
    # FINAL BRIGHTNESS LOCK
    #
    # This is the most important part for your requirement.
    #
    # The final image is pulled back toward the brightness
    # structure of the normal exposure.
    # ========================================================

    result = brightness_lock(

        result,
        normal
    )


    # ========================================================
    # RETURN
    # ========================================================

    diagnostics = {

        "highlight_mask":
            highlight_mask,

        "shadow_mask":
            shadow_mask,

        "normal_highlight_mask":
            normal_highlight_mask,

        "normal_shadow_mask":
            normal_shadow_mask,

        "under_quality":
            under_quality,

        "over_quality":
            over_quality
    }

    return (
        result,
        diagnostics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "===================================="
    )

    print(
        "ADAPTIVE HDR FUSION V3.4"
    )

    print(
        "NORMAL-ANCHOR BRIGHTNESS-LOCKED"
    )

    print(
        "===================================="
    )

    print()

    print(
        f"Under : {UNDER_PATH}"
    )

    print(
        f"Normal: {NORMAL_PATH}"
    )

    print(
        f"Over  : {OVER_PATH}"
    )

    print()


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    ensure_dir(
        OUTPUT_DIR
    )


    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    under = read_image(
        UNDER_PATH
    )

    normal = read_image(
        NORMAL_PATH
    )

    over = read_image(
        OVER_PATH
    )


    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if (

        under.shape != normal.shape

        or

        normal.shape != over.shape

    ):

        raise RuntimeError(

            "Input image dimensions do not match.\n"
            "Align/resize the bracket before fusion."
        )


    height, width = normal.shape[:2]

    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print()


    # --------------------------------------------------------
    # Run fusion
    # --------------------------------------------------------

    result, diagnostics = (

        adaptive_hdr_fusion(

            under,
            normal,
            over
        )
    )


    # --------------------------------------------------------
    # Save final result
    # --------------------------------------------------------

    output_path = os.path.join(

        OUTPUT_DIR,

        FINAL_NAME
    )

    write_image(

        output_path,

        result
    )


    # --------------------------------------------------------
    # Save diagnostic maps
    # --------------------------------------------------------

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "01_highlight_mask.png"
        ),

        diagnostics[
            "highlight_mask"
        ]
    )

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "02_shadow_mask.png"
        ),

        diagnostics[
            "shadow_mask"
        ]
    )

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "03_normal_highlight_problem.png"
        ),

        diagnostics[
            "normal_highlight_mask"
        ]
    )

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "04_normal_shadow_problem.png"
        ),

        diagnostics[
            "normal_shadow_mask"
        ]
    )

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "05_under_detail_quality.png"
        ),

        diagnostics[
            "under_quality"
        ]
    )

    save_map(

        os.path.join(

            OUTPUT_DIR,

            "06_over_detail_quality.png"
        ),

        diagnostics[
            "over_quality"
        ]
    )


    # --------------------------------------------------------
    # Calculate brightness statistics
    # --------------------------------------------------------

    normal_lum = np.mean(
        luminance(
            normal
        )
    )

    result_lum = np.mean(
        luminance(
            result
        )
    )

    brightness_difference = (

        (
            result_lum
            - normal_lum
        )

        /

        (
            normal_lum
            + EPS
        )

        * 100.0
    )


    print()

    print(
        "===================================="
    )

    print(
        "ADAPTIVE HDR FUSION COMPLETED"
    )

    print(
        "===================================="
    )

    print()

    print(
        f"Output: {output_path}"
    )

    print()

    print(
        "Brightness validation:"
    )

    print(
        f"Normal mean luminance: "
        f"{normal_lum:.6f}"
    )

    print(
        f"Output mean luminance: "
        f"{result_lum:.6f}"
    )

    print(
        f"Brightness difference: "
        f"{brightness_difference:+.3f}%"
    )

    print()

    print(
        "Mean recovery masks:"
    )

    print(
        f"Highlight recovery: "
        f"{np.mean(diagnostics['highlight_mask']):.6f}"
    )

    print(
        f"Shadow recovery   : "
        f"{np.mean(diagnostics['shadow_mask']):.6f}"
    )

    print()

    print(
        "Done."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
