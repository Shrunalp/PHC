"""
Condition images before topological feature extraction.

Contents
--------
preprocess : class
    Thresholds and dilates greyscale slides to sharpen cell boundaries.
"""

import numpy as np
import cv2


class preprocess:

    """
    A NumPy and OpenCV based conditioning step that removes background noise and thickens
    cell boundaries before persistence is computed.

    Parameters
    ----------
    thresh : int | None
        Pixels brighter than this value are set to zero, default None (no threshold).

    kernel_size : int
        Side length of the square dilation kernel, default 2.

    iterate : int
        Number of dilation passes over the image, default 1.
    """

    def __init__(
            self,
            thresh: int | None = None,
            kernel_size: int = 2,
            iterate: int = 1
            ):
        self.thresh = thresh
        self.kernel = kernel_size
        self.iterate = iterate

    def threshold(self, img: np.ndarray) -> np.ndarray:

        """
        Filters out bright background pixels so later steps only see tissue structure.

        Parameters
        ----------
        img : np.ndarray of int - size (n, m)
            Greyscale pathology slide with a single channel.

        Returns
        -------
        thresh_img : np.ndarray of int - size (n, m)
            Slide with every pixel above the threshold set to zero.
        """

        if self.thresh is None:
            thresh_img = img  # no threshold requested
        else:
            thresh_img = np.where(img <= self.thresh, img, 0)
        return thresh_img

    def dilate(self, thresh_img: np.ndarray) -> np.ndarray:

        """
        Thickens the remaining structures so that cell boundaries form closed rings, which the
        filtrations then detect as 1-dimensional features.

        Parameters
        ----------
        thresh_img : np.ndarray of int - size (n, m)
            Thresholded greyscale pathology slide.

        Returns
        -------
        dilated_img : np.ndarray of uint8 - size (n, m)
            Slide after `iterate` dilations with a (kernel_size, kernel_size) square kernel.
        """

        kernel = np.ones((self.kernel, self.kernel), np.uint8)
        dilated_img = cv2.dilate(cv2.convertScaleAbs(thresh_img), kernel, iterations=self.iterate)
        return dilated_img
