# -*- coding: utf-8 -*-

"""
Stable Gaussian / Laplacian pyramid utilities
for FreeMEF Mertens exposure fusion.

The implementation preserves the original Mertens
multiresolution-fusion concept while using numerically
stable OpenCV pyramid operations.

Images are expected to be float32 in [0, 1].
"""

import cv2
import numpy as np


EPS = 1e-8


# ============================================================
# KERNEL
# ============================================================

def kernel_1D(i, a=0.6):

    if i == 0:
        return 0.25 + a

    if abs(i) == 1:
        return 0.25

    return 0.0


def kernel_old(i, j, a=0.6):

    return (
        kernel_1D(i, a)
        *
        kernel_1D(j, a)
    )


def get_kernel(a=0.6):

    kernel = np.zeros(
        (5, 5),
        dtype=np.float32
    )

    for i in range(5):

        for j in range(5):

            kernel[i, j] = kernel_old(
                i - 2,
                j - 2,
                a
            )

    return kernel


# ============================================================
# VALIDATE IMAGE
# ============================================================

def _validate_image(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(image)
    ):

        raise ValueError(
            "Pyramid input contains NaN or Inf."
        )

    return image


# ============================================================
# REDUCE ONE LEVEL
# ============================================================

def Reduce1(
    image,
    a=0.6
):

    image = _validate_image(
        image
    )

    # OpenCV pyrDown uses a stable Gaussian
    # low-pass filter followed by 2x reduction.

    if image.ndim == 2:

        result = cv2.pyrDown(
            image,
            borderType=cv2.BORDER_REFLECT_101
        )

    elif image.ndim == 3:

        result = cv2.pyrDown(
            image,
            borderType=cv2.BORDER_REFLECT_101
        )

    else:

        raise ValueError(
            "Reduce1 expects a 2D or 3D image."
        )

    return np.asarray(
        result,
        dtype=np.float32
    )


# ============================================================
# REDUCE
# ============================================================

def Reduce(
    image,
    n,
    a=0.6
):

    result = _validate_image(
        image
    )

    for _ in range(
        int(n)
    ):

        result = Reduce1(
            result,
            a
        )

    return result.astype(
        np.float32
    )


# ============================================================
# EXPAND ONE LEVEL
# ============================================================

def Expand1(
    image,
    a=0.6
):

    image = _validate_image(
        image
    )

    if image.ndim == 2:

        height, width = image.shape

    elif image.ndim == 3:

        height, width = image.shape[:2]

    else:

        raise ValueError(
            "Expand1 expects a 2D or 3D image."
        )

    # pyrUp produces exactly 2x resolution.

    result = cv2.pyrUp(
        image,
        dstsize=(
            width * 2,
            height * 2
        ),
        borderType=cv2.BORDER_REFLECT_101
    )

    return np.asarray(
        result,
        dtype=np.float32
    )


# ============================================================
# EXPAND
# ============================================================

def Expand(
    image,
    n,
    a=0.6
):

    result = _validate_image(
        image
    )

    for _ in range(
        int(n)
    ):

        result = Expand1(
            result,
            a
        )

    return result.astype(
        np.float32
    )


# ============================================================
# SAFE RESIZE
# ============================================================

def resize_to(
    image,
    height,
    width
):

    image = _validate_image(
        image
    )

    result = cv2.resize(
        image,
        (
            int(width),
            int(height)
        ),
        interpolation=cv2.INTER_LINEAR
    )

    return np.asarray(
        result,
        dtype=np.float32
    )


# ============================================================
# PYRAMID SIZE MATCHING
# ============================================================

def match_size(
    image,
    reference
):

    if image.shape == reference.shape:

        return image

    if image.ndim == 2:

        h, w = reference.shape

    else:

        h, w = reference.shape[:2]

    return resize_to(
        image,
        h,
        w
    )


# ============================================================
# NUMERICAL SANITIZATION
# ============================================================

def sanitize(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return image


# ============================================================
# LEGACY COMPATIBILITY
# ============================================================

def Reduce_old(
    image,
    n,
    a=0.6
):

    return Reduce(
        image,
        n,
        a
    )


def Expand_old(
    image,
    n,
    a=0.6
):

    return Expand(
        image,
        n,
        a
    )


# ============================================================
# WEIGHTED SUM
# ============================================================

def weighted_sum(
    image,
    i,
    j,
    a=0.6
):

    """
    Legacy compatibility function.

    This is retained so older repository code does not
    fail if it imports weighted_sum.

    New fusion uses Expand1()/Expand() instead.
    """

    image = _validate_image(
        image
    )

    if image.ndim != 2:

        raise ValueError(
            "weighted_sum expects a 2D image."
        )

    h, w = image.shape

    ii = int(i)
    jj = int(j)

    if (
        ii < 0
        or
        ii >= h
        or
        jj < 0
        or
        jj >= w
    ):

        return 0.0

    return float(
        image[
            ii,
            jj
        ]
    )
