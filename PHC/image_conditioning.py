"""
Condition subimages in the PHC data generation pipeline.
"""

import numpy as np
import cv2

class preprocess:

    """ A NumPy and CV2 based image preprocessing class used to condition images for topological extraction

    Parameters
    ----------
    thresh : int
        Threshold value used to filter noise

    kernel_size : int
        Creates (n,n) np.ndarray to dilate on image

    iterate : int
        Number of iterations to dilate over the subimage 

    """

    def __init__(
        self, 
        thresh: int = None, 
        kernel_size: int = 2, 
        iterate: int = 1
        ):
        
        self.thresh = thresh
        self.kernel = kernel_size
        self.iterate = iterate
        
        
    def threshold(self, img: np.ndarray):

        """
        Parameters
        -----------
        img : np.ndarray of int - size (n,n)
            greyscaled pathology slide, must be depth one or zero

        Returns
        --------
        thresh_img : np.ndarray of int - size (n,n)
            filter out noise from pathology slides to improve topological extraction
        """

        thresh_img = np.where(img <= self.thresh, img, 0) 
        return thresh_img

    def dilate(self, thresh_img: np.ndarray):

        """
        Parameters
        -----------
        thresh_img : np.ndarray of int - size (n,n)
            filtered pathology slide, must be greyscaled

        Returns
        --------
        dilated_img : np.ndarray of int - size (n,n)
            dilated image to emphasize the boundary of cells in a pathology slide
        """

        kernel = np.ones((self.kernel, self.kernel), np.uint8)
        dilated_img = cv2.dilate(cv2.convertScaleAbs(thresh_img), kernel, iterations=self.iterate)
        dilated_img = np.array(dilated_img)
        return dilated_img 
