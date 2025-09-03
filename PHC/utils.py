import numpy as np

def noise_pts(pd: np.ndarray):
        
        """
        Used to simplify persistent diagrams from holes created at the pixel level

        Parameters
        -----------
        pd : np.ndarray of float - size (n,2)
            computed persistence diagram

        Returns
        --------
        denoised_pd: np.ndarray of float - size (m,2)
            Denoised persistence diagram
        """

        value_to_remove = np.array([0.25, 0.5]) #Remove redundent noise
        mask = np.any(pd != value_to_remove, axis=1)
        denoised_pd = pd[mask]
        return denoised_pd

def pointcloud2D(dilated_img: np.ndarray):

    """
    Only needed if filtration used is the alpha complex

    Parameters
    -----------
    img : np.ndarray of int - size (n,n)
        preprocessed pathology slide (i.e., thresholded and dilated)

    Returns
    --------
    pointcloud : np.ndarray of int - size (m,2)
        coordinate positions of nonzero greyscale pixels 
    """

    pointcloud = np.column_stack(np.where(dilated_img >= 0))
    return pointcloud