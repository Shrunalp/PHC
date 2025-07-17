import numpy as np
import persim
from persim import plot_diagrams, PersImage, PersistenceImager
from ripser import ripser, lower_star_img
import matplotlib.pyplot as plt
import cv2
import PIL
import gudhi as gd
from gudhi import AlphaComplex
from scipy import ndimage
from gudhi.representations import Silhouette
from gudhi.representations import PersistenceImage
from sklearn.preprocessing import MinMaxScaler
import tqdm
import time


class gCNN:
    
    def __init__(self, cmplx, vector, stride, patch, res, fltr, itr, kernel):
        self.complex = cmplx
        self.vector = vector
        self.stride = stride
        self.patch = patch
        self.res = res
        self.flter = fltr
        self.iter = itr
        self.kernel = kernel


# The purpose of this class is to compute local persistence of an image. 
# Specifcially, it allows the user to compute the persistence of an 
# image by convolving over it. 

# cmplx:  Used to determine the type of simplicial complex to construct with the data
# vec:    Allows the user to convert persistent diagrams to persistent images or 
#         persistent landscapes
# stride: The distance the window space is shifted by on each iteration
# patch:  Determines the size of the window space that performs the convolution
# res:    Fixes the reolution size of the vectorized persistence data 
# fltr:   Value is used to perform thresholding and preprocess images
# iter:   Number of iterations to apply image erosion
# kernel: Size of window used for image erosion



###############################################################


    ### Filter image ###
    def filter(self, img): 
        result = np.where(img <= self.flter, img, 0) 
        return result

    ### Convert img to 2D point cloud ###
    def ptcl2D(self, img):
        coords = np.empty((img.size, 2), dtype=np.int32)
        count = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] >= 0:
                    coords[count, 0] = i
                    coords[count, 1] = j
                    count += 1

        # Return only the filled part of the coords array
        return coords[:count]

    ### Sublevel or Lower Star Filtration Based Persistence ###
    def lower_star(self, img):
        cells_grey = np.asarray(PIL.Image.fromarray(img).convert('L'))
        smoothed = ndimage.uniform_filter(cells_grey.astype(np.float64), size=10)
        smoothed += 0.01 * np.random.randn(*smoothed.shape)
        data = lower_star_img(-smoothed) #persistence
        data[-1][-1] = 1000000 #Remove np.inf
        return data

    ### Extended Persistence Using Upper and Lower Star ###
    def extend_pers(self, img):
        filtration = img.flatten() # Flatten image 
        num_points = len(filtration)
        
        # Create a SimplexTree 
        SimplexTree = gd.SimplexTree()
        
        # Add vertices with their filtration values
        for i in range(num_points):
            SimplexTree.insert([i], filtration[i])
        
        # Add edges
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if i + 1 < rows:
                    SimplexTree.insert([i * cols + j, (i + 1) * cols + j], 
                                        filtration=max(img[i, j], img[i + 1, j]))
                if j + 1 < cols:
                    SimplexTree.insert([i * cols + j, i * cols + j + 1], 
                                        filtration=max(img[i, j], img[i, j + 1]))
                if i + 1 < rows and j + 1 < cols:
                    SimplexTree.insert([i * cols + j, (i + 1) * cols + j + 1], 
                                        filtration=max(img[i, j], img[i + 1, j + 1]))
                    SimplexTree.insert([i * cols + j + 1, (i + 1) * cols + j], 
                                        filtration=max(img[i, j + 1], img[i + 1, j]))
        
        # Compute extended persistence
        SimplexTree.extend_filtration()
        full_data = SimplexTree.extended_persistence(min_persistence=1e-5)
        h1_data = SimplexTree.persistence_intervals_in_dimension(1)
        return [h1_data, full_data]
    
    ### Compute Vectorize Persistence Diagrams ###
    def process_window(self, c, j, i, fl, fw, img):
        kernel = np.ones((self.kernel, self.kernel), np.uint8)
        img = self.filter(img[j:j+fl, i:i+fw])
        img = np.array(cv2.dilate(cv2.convertScaleAbs(img), kernel, iterations=self.iter))
        
        # Compute persistent diagrams
        if self.complex == "alpha":
            point_cloud = np.array(self.ptcl2D(img))
            stree = AlphaComplex(point_cloud).create_simplex_tree() #Create complex
            dgm = stree.persistence() #Generate PD
            data = self.noise_pts(np.array(stree.persistence_intervals_in_dimension(1)))
            diags = [data] #Correct Format

        elif self.complex == "sublevel":
            dgm = self.lower_star(img) #Generate PD
            diags = [dgm] #Correct Format

        elif self.complex == "extended":
            h1_data, full_data = self.extend_pers(img) #Generate PD
            diags = [h1_data] #Correct Formart

        #Vectorize to landscape or image
        if self.vector == "landscape":
            SH = Silhouette(resolution=self.res, weight=lambda x: np.power(x[1]-x[0],1)) #Initalize vectoization
            persistent_data = SH.fit_transform(diags)
        elif self.vector == "img":
            PI = PersistenceImage(resolution=(self.res,self.res), bandwidth=1.0) #Initalize vectoization
            PI.fit(diags)
            persistent_data = PI.transform(diags)[0]

        return persistent_data

    ### Remove Noise in PD ###
    def noise_pts(self, X):
            value_to_remove = np.array([0.25, 0.5]) #Remove redundent noise
            mask = np.any(X != value_to_remove, axis=1)
            new_arr = X[mask]
            return new_arr

    ### Persistence Convolution ###
    def conv_pers(self, img):

        ### Store Dimensions ###
        if len(img.shape) > 2:
            xc = img.shape[2]
        else:
            xc = 1
        xl = img.shape[1]      # Input volume length
        xw = img.shape[0]      # Input volume width
        fl = self.patch               # Filter length
        fw = self.patch               # Filter width
        ol = (xl - fl) // (self.stride + 1)  # Output volume length
        ow = (xw - fw) // (self.stride + 1)  # Output volume width

        windows = []


        for c in range(xc):
            for j in range(0, xw, self.stride):
                for i in range(0, xl, self.stride):
                    window = self.process_window(c, j, i, fl, fw, img) #Convolution step
                    windows.append(window)

        return windows


###############################################################   
