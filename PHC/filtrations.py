"""
A collection of filtration based methods for computing persistence
"""
import numpy as np
import gudhi as gd
from gudhi import CubicalComplex, AlphaComplex
from .utils import pointcloud2D, remove_noisy_pts


def alphacomplex(img: np.ndarray, alpha_value: float = None, dim: int = 1, persistence_type: str = None) -> list[np.ndarray]:

    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero
        
    alpha_value : float 
        used to smooth greyscaled image to reduce topological noise
    
    dim : int 
        persistent homology in dimension n, default set to 1

    persistence_type : str 
        option to compute ordinary or extended persistence, default is ordinary

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
    
    if persistence_type == None:
        SimplexTree.persistence()
    elif persistence_type == "Extended":
        SimplexTree.extend_filtration()
        SimplexTree.extended_persistence(min_persistence=1e-5)
    
    if dim == 0:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)[:-1]
    else:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)
    dgm = remove_noisy_pts(dgm)
    return [dgm]

def lower_star(img: np.ndarray, dim: int = 1, persistence_type: str = None) -> list[np.ndarray]:

    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero
        
    dim : int 
        persistent homology in dimension n, default set to 1

    persistence_type : str 
        option to compute ordinary or extended persistence, default is ordinary

    Returns
    --------
    dgm : list of an array of size (n,2)
        diagram in dim n. lower star filtration computed using barycentric subdivision 
    """

    rows, cols = img.shape
    num_points = rows*cols

    SimplexTree = gd.SimplexTree()


    #Center vertices: 0 to (rows*cols - 1)
    #Corner vertices: rows*cols to (rows*cols + (rows+1)*(cols+1) - 1)
    
    def get_center_id(r, c):
        return r * cols + c
    
    def get_corner_id(r, c):
        return num_points + r * (cols + 1) + c

    #Compute and insert corner vertices
    for r in range(rows + 1):
        for c in range(cols + 1):
            # Identify adjacent pixel values
            adjacent_values = []
            if r > 0 and c > 0:          adjacent_values.append(img[r-1, c-1]) # Top-Left
            if r > 0 and c < cols:       adjacent_values.append(img[r-1, c])   # Top-Right
            if r < rows and c > 0:       adjacent_values.append(img[r, c-1])   # Bottom-Left
            if r < rows and c < cols:    adjacent_values.append(img[r, c])     # Bottom-Right
            
            filt_val = min(adjacent_values) if adjacent_values else 0.0
            
            SimplexTree.insert([get_corner_id(r, c)], filtration=float(filt_val))

    #Insert boundary edges (horizontal and vertical)

    for r in range(rows):
        for c in range(cols + 1):
            u = get_corner_id(r, c)
            v = get_corner_id(r+1, c)
            
            #Determine adjacent pixels (left and right)
            pixel_vals = []
            if c > 0:    pixel_vals.append(img[r, c-1]) #Left pixel
            if c < cols: pixel_vals.append(img[r, c])   #Right pixel
            
            edge_filt = min(pixel_vals) if pixel_vals else 0.0
            SimplexTree.insert([u, v], filtration=float(edge_filt))

    #Horizontal Edges (between rows)
    for r in range(rows + 1):
        for c in range(cols):
            u = get_corner_id(r, c)
            v = get_corner_id(r, c+1)
            
            # Determine adjacent pixels (Up and Down)
            pixel_vals = []
            if r > 0:    pixel_vals.append(img[r-1, c]) # Up pixel
            if r < rows: pixel_vals.append(img[r, c])   # Down pixel
            
            edge_filt = min(pixel_vals) if pixel_vals else 0.0
            SimplexTree.insert([u, v], filtration=float(edge_filt))

    #Process pixels: center, diagonal, and triangles 
    for r in range(rows):
        for c in range(cols):
            pixel_val = float(img[r, c])
            center_id = get_center_id(r, c)
            
            #Insert center vertex
            SimplexTree.insert([center_id], filtration=pixel_val)
            
            # Get the 4 corner IDs for this pixel
            # TL -- TR
            # |      |
            # BL -- BR
            tl = get_corner_id(r, c)
            tr = get_corner_id(r, c+1)
            bl = get_corner_id(r+1, c)
            br = get_corner_id(r+1, c+1)
            
            triangles = [
                [center_id, tl, tr], # Top triangle
                [center_id, tr, br], # Right triangle
                [center_id, br, bl], # Bottom triangle
                [center_id, bl, tl]  # Left triangle
            ]
            
            for tri in triangles:
                SimplexTree.insert(tri, filtration=pixel_val)

    if persistence_type == None:
        SimplexTree.persistence()
    elif persistence_type == "Extended":
        SimplexTree.extend_filtration()
        SimplexTree.extended_persistence(min_persistence=1e-5)
    
    if dim == 0:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)[:-1]
    else:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)
    dgm = remove_noisy_pts(dgm)
    return [dgm]

def adj_complex(img: np.ndarray, dim: int = 1, persistence_type: str = None) -> list[np.ndarray]:
    
    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero

    dim : int 
        persistent homology in dimension n, default set to 1

    persistence_type : str 
        option to compute ordinary or extended persistence, default is ordinary

    Returns
    --------
    dgm : list of an array of size (n,2)
        dim n adjacency complex filtration based diagram
    """

    filtration = img.flatten() 
    rows, cols = img.shape
    num_points = rows*cols

    SimplexTree = gd.SimplexTree()
    
    #Filtration on the magnitude of pixels
    for i in range(num_points):
        SimplexTree.insert([i], filtration[i])
    for i in range(rows):
        for j in range(cols):

            idx = i * cols + j
            
            #Neighbors
            down = (i + 1) * cols + j
            right = i * cols + (j + 1)
            diag_br = (i + 1) * cols + (j + 1) #Bottom-right
            diag_bl = (i + 1) * cols + j       #Bottom-left (used with right)

            #Vertical edges
            if i + 1 < rows:
                SimplexTree.insert([idx, down], filtration=max(img[i, j], img[i+1, j]))
            
            #Horizontal edges
            if j + 1 < cols:
                SimplexTree.insert([idx, right], filtration=max(img[i, j], img[i, j+1]))
            
            #Diagonal edges and triangles
            if i + 1 < rows and j + 1 < cols:
                #Diagonal 1: top-left to bottom-right
                SimplexTree.insert([idx, diag_br], filtration=max(img[i, j], img[i+1, j+1]))
                #Diagonal 2: top-right to bottom-left
                SimplexTree.insert([idx + 1, diag_bl], filtration=max(img[i, j+1], img[i+1, j]))
    
    if persistence_type == None:
        SimplexTree.persistence()
    elif persistence_type == "Extended":
        SimplexTree.extend_filtration()
        SimplexTree.extended_persistence(min_persistence=1e-5)
    
    if dim == 0:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)[:-1]
    else:
        dgm = SimplexTree.persistence_intervals_in_dimension(dim)
    dgm = remove_noisy_pts(dgm)
    return [dgm]

def cubicalcomplex(img: np.ndarray, dim: int = 1) -> list[np.ndarray]:

    """
    Parameters
    -----------
    img : np.ndarray of float 
        greyscaled pathology slide, must be depth one or zero 
    dim : int 
        persistent homology in dimension n, default set to 1

    Returns
    --------
    dgm : list of an array of size (n,2)
        dim one persistence diagram generated from the cubical complex using lowerstar filtration
    """

    cubical_complex = CubicalComplex(dimensions=img.shape,
                                       top_dimensional_cells=img.flatten()) 
    cubical_complex.persistence(homology_coeff_field=2, min_persistence=0.01)
    dgm = np.array(cubical_complex.persistence_intervals_in_dimension(dim))
    return [dgm]
