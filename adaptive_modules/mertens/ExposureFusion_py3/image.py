# -*- coding: utf-8 -*-

import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def weightedAverage(pixel):
    return (
        0.299 * pixel[0]
        + 0.587 * pixel[1]
        + 0.114 * pixel[2]
    )


def exponential_euclidean(canal, sigma):
    return np.exp(
        -((canal - 0.5) ** 2)
        / (2.0 * sigma ** 2)
    )


# ============================================================
# DISPLAY
# ============================================================

def show(color_array):

    plt.imshow(
        np.clip(
            color_array,
            0.0,
            1.0
        )
    )

    plt.axis("off")
    plt.show()


def show_gray(gray_array):

    plt.imshow(
        gray_array,
        cmap=plt.cm.Greys_r
    )

    plt.axis("off")
    plt.show()


# ============================================================
# IMAGE
# ============================================================

class Image(object):

    def __init__(
        self,
        fmt,
        path,
        crop=False,
        n=0
    ):

        self.path = os.path.join(
            "image_set",
            fmt,
            str(path)
        )

        self.fmt = fmt

        # ----------------------------------------------------
        # Replacement for obsolete scipy.misc.imread
        # ----------------------------------------------------

        image = cv2.imread(
            self.path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image:\n"
                f"{self.path}"
            )

        # OpenCV BGR -> repository RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        self.array = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # Original repository crop behavior
        # ----------------------------------------------------

        if crop:

            self.crop_image(
                n
            )

        self.shape = (
            self.array.shape
        )


    # ========================================================
    # CROP
    # ========================================================

    def crop_image(self, n):

        resolution = (
            2 ** int(n)
        )

        height, width, _ = (
            self.array.shape
        )

        max_height = (
            resolution *
            (
                height //
                resolution
            )
        )

        max_width = (
            resolution *
            (
                width //
                resolution
            )
        )

        begin_height = (
            height -
            max_height
        ) // 2

        begin_width = (
            width -
            max_width
        ) // 2

        self.array = self.array[
            begin_height:
            begin_height + max_height,

            begin_width:
            begin_width + max_width,

            :
        ]


    # ========================================================
    # GRAYSCALE
    # ========================================================

    @property
    def grayScale(self):

        rgb = self.array

        self._grayScale = np.dot(
            rgb[..., :3],
            np.array(
                [
                    0.299,
                    0.587,
                    0.114
                ],
                dtype=np.float32
            )
        )

        return self._grayScale


    # ========================================================
    # SATURATION
    # ========================================================

    def saturation(self):

        red_canal = (
            self.array[:, :, 0]
        )

        green_canal = (
            self.array[:, :, 1]
        )

        blue_canal = (
            self.array[:, :, 2]
        )

        mean = (
            red_canal
            + green_canal
            + blue_canal
        ) / 3.0

        saturation = np.sqrt(
            (
                (red_canal - mean) ** 2
                +
                (green_canal - mean) ** 2
                +
                (blue_canal - mean) ** 2
            ) / 3.0
        )

        return saturation


    # ========================================================
    # CONTRAST
    # ========================================================

    def contrast(self):

        grey = self.grayScale

        height, width = (
            grey.shape
        )

        contrast = np.zeros(
            (
                height,
                width
            ),
            dtype=np.float32
        )

        grey_extended = np.zeros(
            (
                height + 2,
                width + 2
            ),
            dtype=np.float32
        )

        grey_extended[
            1:height + 1,
            1:width + 1
        ] = grey

        kernel = np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ],
            dtype=np.float32
        )

        for row in range(height):

            for col in range(width):

                contrast[
                    row,
                    col
                ] = abs(
                    (
                        kernel
                        *
                        grey_extended[
                            row:row + 3,
                            col:col + 3
                        ]
                    ).sum()
                )

        contrast -= np.min(
            contrast
        )

        maximum = np.max(
            contrast
        )

        if maximum > 0.0:

            contrast /= maximum

        return contrast


    # ========================================================
    # SOBEL
    # ========================================================

    def sobel(self):

        grey = self.grayScale

        height, width = (
            grey.shape
        )

        sobel_h = np.zeros(
            (
                height,
                width
            ),
            dtype=np.float32
        )

        sobel_v = np.zeros(
            (
                height,
                width
            ),
            dtype=np.float32
        )

        grey_extended = np.zeros(
            (
                height + 2,
                width + 2
            ),
            dtype=np.float32
        )

        grey_extended[
            1:height + 1,
            1:width + 1
        ] = grey

        kernel1 = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=np.float32
        )

        kernel2 = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=np.float32
        )

        for row in range(height):

            for col in range(width):

                patch = (
                    grey_extended[
                        row:row + 3,
                        col:col + 3
                    ]
                )

                sobel_h[
                    row,
                    col
                ] = abs(
                    (
                        kernel1 *
                        patch
                    ).sum()
                )

                sobel_v[
                    row,
                    col
                ] = abs(
                    (
                        kernel2 *
                        patch
                    ).sum()
                )

        return (
            sobel_h,
            sobel_v
        )


    # ========================================================
    # WELL-EXPOSEDNESS
    # ========================================================

    def exposedness(self):

        red_canal = (
            self.array[:, :, 0]
        )

        green_canal = (
            self.array[:, :, 1]
        )

        blue_canal = (
            self.array[:, :, 2]
        )

        sigma = 0.2

        red_exp = exponential_euclidean(
            red_canal,
            sigma
        )

        green_exp = exponential_euclidean(
            green_canal,
            sigma
        )

        blue_exp = exponential_euclidean(
            blue_canal,
            sigma
        )

        return (
            red_exp
            * green_exp
            * blue_exp
        )
