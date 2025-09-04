"""
A collection of filtration based methods for computing persistence
"""
import numpy as np
import gudhi as gd
from gudhi import CubicalComplex, AlphaComplex
from .utils import pointcloud2D


def alphacomplex(img: np.ndarray, alpha_value: float = None) -> list[np.ndarray]:

    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero
        
    alpha_value : float 
        used to smooth greyscaled image to reduce topological noise

    Returns
    --------
    dgm : list of an array of size (n,2)
        persistence diagram of codim one using Alexander Duality to detect cell formation
    """
    points = pointcloud2D(img)
    alpha_complex = AlphaComplex(points=points)
    
    if alpha_value is None:
        SimplexTree = alpha_complex.create_simplex_tree()
    else:
        SimplexTree = alpha_complex.create_simplex_tree(max_alpha_square=alpha_value**2)
    
    SimplexTree.persistence()
    dgm = SimplexTree.persistence_intervals_in_dimension(1)
    return [dgm]

def cubicalcomplex(img: np.ndarray) -> list[np.ndarray]:

    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero 

    Returns
    --------
    dgm : list of an array of size (n,2)
        dim one persistence diagram generated from the cubical complex using lowerstar filtration
    """

    cubical_complex = CubicalComplex(dimensions=img.shape,
                                       top_dimensional_cells=img.flatten()) 
    cubical_complex.persistence(homology_coeff_field=2, min_persistence=0.01)
    dgm = np.array(cubical_complex.persistence_intervals_in_dimension(1))
    dgm = [dgm]
    return dgm

def ext_persistence(img: np.ndarray, filtration_function: str = "pixel_intensity") -> list[np.ndarray]:
    
    """
    Parameters
    -----------
    img : np.ndarray of float - size (n,n)
        greyscaled pathology slide, must be depth one or zero

    filtration_function : string
        choice in filtration, pixel_intensity is choosen as default 

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