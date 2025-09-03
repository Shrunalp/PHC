"""
A collection of filtration based methods for computing persistence
"""
import numpy as np
from ripser import lower_star_img
import PIL
import gudhi as gd
from gudhi import AlphaComplex
from scipy import ndimage
from .utils import noise_pts


def lower_star(img, smoothing_factor: float = 0.01):

    """
    Parameters
    -----------
    img : np.ndarray of float - size (n,n)
        greyscaled pathology slide, must be depth one or zero
        
    smoothing_factor : float 
        used to smooth greyscaled image to reduce topological noise

    Returns
    --------
    dgm : list of an array of size (n,2)
        persistence diagram of codim one using Alexander Duality to detect cell formation
    """
    
    cells_grey = np.asarray(PIL.Image.fromarray(img).convert('L'))
    smoothed = ndimage.uniform_filter(cells_grey.astype(np.float64), size=10)
    smoothed += smoothing_factor * np.random.randn(*smoothed.shape)
    dgm = lower_star_img(-smoothed) # persistence 
    dgm[-1][-1] = 750 #Replace np.inf
    return [dgm]

def alphacomplex(pointcloud):

    """
    Parameters
    -----------
    pointcloud : np.ndarray of float - size (n,2)
        2D pointcloud used to represent spatial distance between cells in pathology images

    Returns
    --------
    dgm : list of an array of size (n,2)
        dim one persistence diagram generated from the alpha complex
    """

    SimplexTree = AlphaComplex(pointcloud).create_simplex_tree() 
    SimplexTree.persistence() 
    dgm = np.array(SimplexTree.persistence_intervals_in_dimension(1))
    dgm = noise_pts(dgm) #remove noise generated on the pixel level
    dgm = [dgm]
    return dgm

def ext_persistence(img, filtration_function: str = "height_function"):
    
    """
    Parameters
    -----------
    img : np.ndarray of float - size (n,n)
        greyscaled pathology slide, must be depth one or zero

    filtration_function : string
        choice in filtration between pixel_intensity and height_function

    Returns
    --------
    dgm : list of an array of size (n,2)
        dim one ext-persistence diagram generated based on choice of filtration function
    """

    filtration = img.flatten() 
    rows, cols = img.shape
    num_points = rows*cols

    SimplexTree = gd.SimplexTree()
    
    if filtration_function == "pixel_intensity": #filtration on the magnitude of pixels
        for i in range(num_points):
            SimplexTree.insert([i], filtration[i])
        for i in range(rows):
            for j in range(cols):
                if i+1 < rows:
                    SimplexTree.insert([i*cols+j, (i+1)*cols+j], #vertical
                                        filtration=max(img[i, j], img[i+1, j]))
                if j+1 < cols:
                    SimplexTree.insert([i*cols+j, i*cols+j+1], #horizontal
                                        filtration=max(img[i, j], img[i, j+1]))
                if i+1 < rows and j+1 < cols:
                    SimplexTree.insert([i*cols+j, (i+1)*cols+j+1], #top left to bottom right
                                        filtration=max(img[i, j], img[i+1, j+1]))
                    SimplexTree.insert([i*cols+j+1, (i+1)*cols+j], #top right to bottom left
                                        filtration=max(img[i, j+1], img[i+1, j]))
    
    elif filtration_function == "height_function": #filtration on the height of cell boundaries
        for i in range(num_points):
            SimplexTree.insert([i], i%rows)
        for i in range(rows):
            for j in range(cols):
                if i+1 < rows:
                    SimplexTree.insert([i*cols+j, (i+1)*cols+j], filtration=i+1)
                if j+1 < cols:
                    SimplexTree.insert([i*cols+j, i*cols+j+1], filtration=i)
                if i+1 < rows and j+1 < cols:
                    SimplexTree.insert([i*cols+j, (i+1)*cols+j+1], filtration=i+1)
                    SimplexTree.insert([i*cols+j+1, (i+1)*cols+j], filtration=i+1)

    SimplexTree.extend_filtration()
    SimplexTree.extended_persistence(min_persistence=1e-5)
    dgm = SimplexTree.persistence_intervals_in_dimension(1)
    return [dgm]