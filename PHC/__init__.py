"""
Persistent Homology Convolutions: localized persistent homology of images and of cell point
clouds, with tools to cluster, embed and plot the resulting persistence vectors.

Contents
--------
PHC : class
    Sliding-window (or per-cell) persistence, vectorized as persistence images or silhouettes.
preprocess : class
    Thresholds and dilates greyscale slides before persistence is computed.
alphacomplex, alpha_pointcloud, lower_star, adj_complex, cubicalcomplex : function
    Interchangeable filtrations used by PHC on each window.
remove_noisy_pts, pointcloud2D : function
    Helpers shared by the filtrations.
cell_centroids, feature_centroids, read_cell_centroids, read_cell_features : function
    Cell positions from QuPath GeoJSON exports.
window_dissimilarity, agglomerative_clusters, agglomerative_clusters_capped : function
    L2 measures and agglomerative clustering of persistence vectors.
l2_dissimilarity, classical_mds, kruskal_stress, mds_embedding : function
    L2 distance matrices and their (metric) MDS embeddings.
delaunay_adjacency, connect_components, spatial_agglomerative_clusters : function
    Clustering restricted to neighbours in the Delaunay graph of the cell centroids.
cluster_colors, plot_embedding, plot_spatial_graph : function
    matplotlib figures coloured like the QuPath extension.
constrained_linkage_labels, numba_backend_available : function
    Fast numba backend of the spatial clustering; only exported when numba is installed.
"""

from .image_conditioning import preprocess
from .local_ph import PHC
from .filtrations import adj_complex, alpha_pointcloud, alphacomplex, cubicalcomplex, lower_star
from .utils import pointcloud2D, remove_noisy_pts
from .clustering import (agglomerative_clusters, agglomerative_clusters_capped, classical_mds,
                         kruskal_stress, l2_dissimilarity, mds_embedding, window_dissimilarity)
from .plotting import cluster_colors, plot_embedding, plot_spatial_graph
from .spatial import connect_components, delaunay_adjacency, spatial_agglomerative_clusters
from .cells import cell_centroids, feature_centroids, read_cell_centroids, read_cell_features

__all__ = ['PHC', 'preprocess', 'adj_complex', 'alpha_pointcloud', 'alphacomplex',
           'cubicalcomplex', 'lower_star', 'remove_noisy_pts', 'pointcloud2D',
           'agglomerative_clusters', 'window_dissimilarity', 'cell_centroids',
           'read_cell_centroids', 'l2_dissimilarity', 'classical_mds', 'kruskal_stress',
           'mds_embedding', 'cluster_colors', 'plot_embedding', 'agglomerative_clusters_capped',
           'feature_centroids', 'read_cell_features', 'delaunay_adjacency',
           'connect_components', 'spatial_agglomerative_clusters', 'plot_spatial_graph']

try:  # optional numba backend for spatially constrained clustering
    from .constrained_linkage import constrained_linkage_labels, numba_backend_available
    __all__ += ['constrained_linkage_labels', 'numba_backend_available']
except ImportError:
    pass
