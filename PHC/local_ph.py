"""
Generates PHC data
"""

from gudhi.representations import Silhouette
from gudhi.representations import PersistenceImage
import numpy as np
from .filtrations import *

class PHC:

    """ Computes localized peristence using Persistent Homology Convolutions

    Parameters
    ----------
    persistence_type : str
        Determines which method of persistence to compute

    window_size : int
        Localizes computations to (n,n) subwindow of the orginal image

    stride : int
        Value used to translate subwindow across the image

    vectorization : str
        Converts persistence diagrams into persistent images or persistent landscapes 

    vector_resolution : int
        Fixes the dimension output in the vectorization of the persistence diagrams

    """

    def __init__(
            self, 
            persistence_type: str = "cube", 
            window_size: int = 32, 
            stride: int = 32, 
            vectorization: str = "PI", 
            vector_resolution: int = 20
            ):
        self.persistence_type = persistence_type
        self.window_size = window_size
        self.stride =  stride
        self.vectorization = vectorization
        self.vector_resolution = vector_resolution
        
    def process_window(self, img: np.ndarray, x_cord: int, y_cord: int, window_width: int, window_length: int) -> np.ndarray:

        """
        Parameters
        -----------
        img : np.ndarray of floats
            greyscaled pathology slide
        
        x_cord : int
            positioning of subimage horizontally wise
        
        y_cord : int
            positioning of subimage vertically wise

        window_width : int
            width from corresponding x_cord 
        
        window_length : int
            length from corresponding y_cord 

        Returns
        --------
        vectorized_pd : np.ndarray of float - size (self.vector_resolution, self.vector_resolution)
            vectorized representation of local persistent homology data
        """

        subimg = img[x_cord:x_cord+window_width, y_cord:y_cord+window_length]
        if self.persistence_type == "cubical":
            pd = cubicalcomplex(subimg)
        elif self.persistence_type == "alpha":
            pd = alphacomplex(subimg)
        elif self.persistence_type == "extended":
            pd = ext_persistence(subimg) #condition image before inputting
        else:
            raise ValueError("Unknown Persistence Method.")


        if self.vectorization == "PL":
            SH = Silhouette(resolution=self.vector_resolution, weight=lambda x: np.power(x[1]-x[0],1)) #Initalize vectoization
            vectorized_pd = SH.fit_transform(pd)
        elif self.vectorization == "PI":
            PI = PersistenceImage(resolution=(self.vector_resolution, self.vector_resolution), bandwidth=1.0) #Initalize vectoization
            PI.fit(pd)
            vectorized_pd = PI.transform(pd)[0]

        return vectorized_pd


    def convolve(self, img: np.ndarray) -> list[np.ndarray]:

        """
        Parameters
        -----------
        img : np.ndarray of floats
            greyscaled pathology slide


        Returns
        --------
        windows : list of np.ndarray of float
            local persistent homology data convolved over the entire image in vector form
        """

        ### Store Dimensions ###
        width = img.shape[0]      
        length = img.shape[1]      
        window_length = self.window_size               
        window_width = self.window_size               

        windows = []
        for x_cord in range(0, width, self.stride):
            for y_cord in range(0, length, self.stride):
                vectorized_data = self.process_window(img, x_cord, y_cord, 
                                                      window_width, window_length) #local persistence
                windows.append(vectorized_data)

        return windows    
        
    