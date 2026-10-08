"""
Fast, exact connectivity-constrained agglomerative clustering (Ward and average linkage) in
numba: the same merge tree, distances and labels as scikit-learn's AgglomerativeClustering
with a connectivity matrix, bit for bit, in a fraction of the time.

sklearn's ward_tree / linkage_tree run their merge loop in Python (a heap of every candidate
pair, ~10^7 pops at 5 * 10^4 cells). The kernels here reproduce that loop exactly:

* Ward: sklearn pops the lexicographically smallest valid (inertia, i, j) tuple; every pair
  is pushed once, so keys are unique and any exact priority queue on that key gives the same
  merge sequence, ties included. Ours is two-level (one small heap per owner node plus a
  global heap of owner minima). Inertias use compute_ward_dist's exact arithmetic (centroids
  m2 / m1, squared differences summed in ascending feature order, no fastmath / FMA).
* Average: sklearn's heap compares on weight only, so ties are broken by the heap layout;
  CPython's heapq is therefore replicated step for step with the same push order, and the
  merge rule (n_a * a + n_b * b) / n_out is evaluated in the same form as the installed
  sklearn build (plain or fused multiply-add, detected at runtime).

`numba_backend_available` checks all of this against the installed sklearn once per process;
callers should fall back to sklearn when it returns False.

Contents
--------
constrained_linkage_labels : function
    Cluster labels identical to AgglomerativeClustering(linkage=..., connectivity=...).
numba_backend_available : function
    One-off self-check of the numba kernels against the installed sklearn.
"""

import warnings
from heapq import heappush, heappushpop

import numpy as np
from numba import njit
from numba.core import types as numba_types
from numba.core.errors import NumbaWarning
from numba.extending import intrinsic
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import _hierarchical_fast as sk_hierarchical
from sklearn.cluster._agglomerative import _fix_connectivity
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils._fast_dict import IntFloatDict

CONSTRAINED_LINKAGES = ("ward", "average")
FULL_TREE_MIN_CLUSTERS = 100     # sklearn's compute_full_tree="auto": full tree when
FULL_TREE_FRACTION = 0.02        # n_clusters < max(100, 0.02 * n), else stop early
POOL_SLACK = 1024                # spare entries in every growable numba pool
NEIGHBOUR_BUFFER = 64            # initial size of the per-merge neighbour buffer
WARD_BLOCK = 8                   # neighbours whose distances are summed side by side
CHECK_SEED = 7                   # self-check: random moments and small clustering problems
CHECK_PAIRS = 4000               # self-check: distance pairs compared with compute_ward_dist
CHECK_NODES = 512                # self-check: distinct moment rows
CHECK_FEATURES = 37              # self-check: features per moment row
CHECK_MAX_COUNT = 1000           # self-check: largest cluster size in the moments
CHECK_SCALE = 3.7                # self-check: spread of the moment sums
CHECK_MERGE_KEYS = 4000          # self-check: dictionary entries for the average_merge probe
CHECK_MERGE_SIZES = (37, 1234)   # self-check: cluster sizes n_a, n_b for that probe
CHECK_GRID = (12, 25)            # self-check: grid graph of the small clustering problems
CHECK_DIMENSION = 5              # self-check: features of the small clustering problems
CHECK_CLUSTERS = (2, 5, 150)     # self-check: n_clusters (150 exercises the early stop)


def _jit(**options):

    """
    numba's njit with the on-disk cache when it can be used, so later processes skip
    compilation; falls back to an uncached function when no cache directory is writable.

    Parameters
    ----------
    **options : dict
        Extra njit options, e.g. inline="always".

    Returns
    -------
    decorate : callable
        Decorator turning a Python function into a numba dispatcher.
    """

    def decorate(function):

        """
        Compiles `function` lazily, cached when possible.

        Parameters
        ----------
        function : callable
            Python function in numba's nopython subset.

        Returns
        -------
        compiled : numba.core.registry.CPUDispatcher
            The jitted function.
        """

        try:
            compiled = njit(cache=True, **options)(function)
        except Exception:  # no writable cache locator: compile in memory only
            compiled = njit(**options)(function)
        return compiled

    return decorate


###############################################################################
# Shared numba helpers


@_jit()
def _grow(array: np.ndarray, new_size: int) -> np.ndarray:

    """
    Enlarges a numba pool array, keeping its contents.

    Parameters
    ----------
    array : np.ndarray - size (m,)
        Pool to enlarge.

    new_size : int
        New length, at least m.

    Returns
    -------
    grown : np.ndarray - size (new_size,)
        Copy of `array` followed by uninitialised entries.
    """

    grown = np.empty(new_size, dtype=array.dtype)
    grown[:array.shape[0]] = array
    return grown


@_jit()
def _labels_from_cut(children: np.ndarray, n_leaves: int, cut_nodes: np.ndarray) -> np.ndarray:

    """
    Gives every leaf the index of the cut node above it, as sklearn's _hc_cut does with
    _hc_get_descendent, in one top-down pass over the tree.

    Parameters
    ----------
    children : np.ndarray of int - size (m, 2)
        Children of the m merge nodes (node n_leaves + t has children[t]).

    n_leaves : int
        Number of leaves.

    cut_nodes : np.ndarray of int - size (k,)
        Node ids forming the cut; label t goes to the leaves below cut_nodes[t].

    Returns
    -------
    labels : np.ndarray of int - size (n_leaves,)
        Cluster label of every leaf.
    """

    n_nodes = n_leaves + children.shape[0]
    node_label = np.full(n_nodes, -1, dtype=np.int64)
    for t in range(cut_nodes.shape[0]):
        node_label[cut_nodes[t]] = t
    for node in range(n_nodes - 1, n_leaves - 1, -1):  # parents before their children
        if node_label[node] >= 0:
            node_label[children[node - n_leaves, 0]] = node_label[node]
            node_label[children[node - n_leaves, 1]] = node_label[node]
    labels = node_label[:n_leaves]
    return labels


@_jit()
def _tree_heads(parent: np.ndarray, n_leaves: int) -> np.ndarray:

    """
    Root of every leaf in a partially built tree (sklearn's hc_get_heads on the leaves).

    Parameters
    ----------
    parent : np.ndarray of int - size (n_nodes,)
        Parent of each node, itself for roots.

    n_leaves : int
        Number of leaves.

    Returns
    -------
    heads : np.ndarray of int - size (n_leaves,)
        Root node id above each leaf.
    """

    heads = np.empty(n_leaves, dtype=np.int64)
    for leaf in range(n_leaves):
        node = leaf
        while parent[node] != node:
            node = parent[node]
        heads[leaf] = node
    return heads


###############################################################################
# Ward: priority queues on lexicographic keys


@_jit(inline="always")
def _key_less(v1: float, i1: int, j1: int, v2: float, i2: int, j2: int) -> bool:

    """
    Python tuple order of (inertia, i, j), the order of sklearn's Ward heap.

    Parameters
    ----------
    v1 : float
        Inertia of the first key.

    i1 : int
        Owner (larger) node of the first key.

    j1 : int
        Other node of the first key.

    v2 : float
        Inertia of the second key.

    i2 : int
        Owner node of the second key.

    j2 : int
        Other node of the second key.

    Returns
    -------
    less : bool
        True when the first key sorts before the second.
    """

    less = v1 < v2 or (v1 == v2 and (i1 < i2 or (i1 == i2 and j1 < j2)))
    return less


@_jit()
def _global_sift_down(hv: np.ndarray, hi: np.ndarray, hj: np.ndarray, pos: int,
                      size: int) -> None:

    """
    Restores the global min-heap below `pos` after its key grew or was replaced.

    Parameters
    ----------
    hv : np.ndarray of float - size (cap,)
        Heap inertias.

    hi : np.ndarray of int - size (cap,)
        Heap owner nodes.

    hj : np.ndarray of int - size (cap,)
        Heap other nodes.

    pos : int
        Position to sift down from.

    size : int
        Number of heap entries in use.

    Returns
    -------
    None
    """

    v = hv[pos]
    a = hi[pos]
    b = hj[pos]
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        if child + 1 < size and _key_less(hv[child + 1], hi[child + 1], hj[child + 1],
                                          hv[child], hi[child], hj[child]):
            child += 1
        if not _key_less(hv[child], hi[child], hj[child], v, a, b):
            break
        hv[pos] = hv[child]
        hi[pos] = hi[child]
        hj[pos] = hj[child]
        pos = child
    hv[pos] = v
    hi[pos] = a
    hj[pos] = b


@_jit()
def _global_sift_up(hv: np.ndarray, hi: np.ndarray, hj: np.ndarray, pos: int) -> None:

    """
    Moves a newly appended global-heap entry up to its place.

    Parameters
    ----------
    hv : np.ndarray of float - size (cap,)
        Heap inertias.

    hi : np.ndarray of int - size (cap,)
        Heap owner nodes.

    hj : np.ndarray of int - size (cap,)
        Heap other nodes.

    pos : int
        Position of the new entry.

    Returns
    -------
    None
    """

    v = hv[pos]
    a = hi[pos]
    b = hj[pos]
    while pos > 0:
        up = (pos - 1) >> 1
        if not _key_less(v, a, b, hv[up], hi[up], hj[up]):
            break
        hv[pos] = hv[up]
        hi[pos] = hi[up]
        hj[pos] = hj[up]
        pos = up
    hv[pos] = v
    hi[pos] = a
    hj[pos] = b


@_jit(inline="always")
def _local_less(v1: float, j1: int, v2: float, j2: int) -> bool:

    """
    Order of (inertia, j) inside one owner's heap (the owner i is the same for all).

    Parameters
    ----------
    v1 : float
        Inertia of the first entry.

    j1 : int
        Other node of the first entry.

    v2 : float
        Inertia of the second entry.

    j2 : int
        Other node of the second entry.

    Returns
    -------
    less : bool
        True when the first entry sorts before the second.
    """

    less = v1 < v2 or (v1 == v2 and j1 < j2)
    return less


@_jit()
def _local_sift_down(ev: np.ndarray, ec: np.ndarray, base: int, pos: int, size: int) -> None:

    """
    Restores one owner's min-heap, stored at ev[base:base + size], below `pos`.

    Parameters
    ----------
    ev : np.ndarray of float - size (cap,)
        Entry inertias of all owners.

    ec : np.ndarray of int - size (cap,)
        Entry other nodes of all owners.

    base : int
        Start of this owner's heap in the pool.

    pos : int
        Position (relative to base) to sift down from.

    size : int
        Number of entries in this owner's heap.

    Returns
    -------
    None
    """

    v = ev[base + pos]
    c = ec[base + pos]
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        if child + 1 < size and _local_less(ev[base + child + 1], ec[base + child + 1],
                                            ev[base + child], ec[base + child]):
            child += 1
        if not _local_less(ev[base + child], ec[base + child], v, c):
            break
        ev[base + pos] = ev[base + child]
        ec[base + pos] = ec[base + child]
        pos = child
    ev[base + pos] = v
    ec[base + pos] = c


@_jit()
def _local_pop(ev: np.ndarray, ec: np.ndarray, base: int, size: int) -> int:

    """
    Removes the top entry of one owner's heap.

    Parameters
    ----------
    ev : np.ndarray of float - size (cap,)
        Entry inertias of all owners.

    ec : np.ndarray of int - size (cap,)
        Entry other nodes of all owners.

    base : int
        Start of this owner's heap in the pool.

    size : int
        Number of entries before the pop, at least 1.

    Returns
    -------
    new_size : int
        Number of entries after the pop.
    """

    new_size = size - 1
    if new_size > 0:
        ev[base] = ev[base + new_size]
        ec[base] = ec[base + new_size]
        _local_sift_down(ev, ec, base, 0, new_size)
    return new_size


###############################################################################
# Ward: distances with compute_ward_dist's arithmetic


@_jit(inline="always")
def _ward_dist(centroids: np.ndarray, sa: int, sb: int, ma: float, mb: float,
               d: int) -> float:

    """
    Ward inertia of merging two clusters, bit-identical to sklearn's compute_ward_dist:
    (ma * mb) / (ma + mb) times the squared centroid distance summed in feature order.

    Parameters
    ----------
    centroids : np.ndarray of float - size (n, d)
        Cluster centroids m2 / m1, one row per storage slot.

    sa : int
        Slot of the first cluster.

    sb : int
        Slot of the second cluster.

    ma : float
        Size of the first cluster.

    mb : float
        Size of the second cluster.

    d : int
        Number of features.

    Returns
    -------
    inertia : float
        Ward merge cost.
    """

    weight = (ma * mb) / (ma + mb)
    total = 0.0
    for f in range(d):
        t = centroids[sa, f] - centroids[sb, f]
        total += t * t
    inertia = total * weight
    return inertia


@_jit()
def _square_sums4(centroids: np.ndarray, sk: int, slot: np.ndarray, nb: np.ndarray, q: int,
                  d: int, out: np.ndarray) -> None:

    """
    Squared distances from slot sk to four neighbours, summed side by side for instruction
    level parallelism; each sum is the same ascending-feature sum as in _ward_dist.

    Parameters
    ----------
    centroids : np.ndarray of float - size (n, d)
        Cluster centroids, one row per storage slot.

    sk : int
        Slot of the new cluster.

    slot : np.ndarray of int - size (n_nodes,)
        Storage slot of every node.

    nb : np.ndarray of int - size (m,)
        Neighbour node ids; nb[q:q + 4] are used.

    q : int
        First neighbour position.

    d : int
        Number of features.

    out : np.ndarray of float - size (8,)
        Receives the four sums in out[0:4].

    Returns
    -------
    None
    """

    s0 = slot[nb[q]]
    s1 = slot[nb[q + 1]]
    s2 = slot[nb[q + 2]]
    s3 = slot[nb[q + 3]]
    p0 = p1 = p2 = p3 = 0.0
    for f in range(d):
        ck = centroids[sk, f]
        t0 = ck - centroids[s0, f]
        t1 = ck - centroids[s1, f]
        t2 = ck - centroids[s2, f]
        t3 = ck - centroids[s3, f]
        p0 += t0 * t0
        p1 += t1 * t1
        p2 += t2 * t2
        p3 += t3 * t3
    out[0] = p0
    out[1] = p1
    out[2] = p2
    out[3] = p3


@_jit()
def _square_sums8(centroids: np.ndarray, sk: int, slot: np.ndarray, nb: np.ndarray, q: int,
                  d: int, out: np.ndarray) -> None:

    """
    Squared distances from slot sk to eight neighbours, summed side by side; each sum is the
    same ascending-feature sum as in _ward_dist.

    Parameters
    ----------
    centroids : np.ndarray of float - size (n, d)
        Cluster centroids, one row per storage slot.

    sk : int
        Slot of the new cluster.

    slot : np.ndarray of int - size (n_nodes,)
        Storage slot of every node.

    nb : np.ndarray of int - size (m,)
        Neighbour node ids; nb[q:q + 8] are used.

    q : int
        First neighbour position.

    d : int
        Number of features.

    out : np.ndarray of float - size (8,)
        Receives the eight sums.

    Returns
    -------
    None
    """

    s0 = slot[nb[q]]
    s1 = slot[nb[q + 1]]
    s2 = slot[nb[q + 2]]
    s3 = slot[nb[q + 3]]
    s4 = slot[nb[q + 4]]
    s5 = slot[nb[q + 5]]
    s6 = slot[nb[q + 6]]
    s7 = slot[nb[q + 7]]
    p0 = p1 = p2 = p3 = p4 = p5 = p6 = p7 = 0.0
    for f in range(d):
        ck = centroids[sk, f]
        t0 = ck - centroids[s0, f]
        t1 = ck - centroids[s1, f]
        t2 = ck - centroids[s2, f]
        t3 = ck - centroids[s3, f]
        t4 = ck - centroids[s4, f]
        t5 = ck - centroids[s5, f]
        t6 = ck - centroids[s6, f]
        t7 = ck - centroids[s7, f]
        p0 += t0 * t0
        p1 += t1 * t1
        p2 += t2 * t2
        p3 += t3 * t3
        p4 += t4 * t4
        p5 += t5 * t5
        p6 += t6 * t6
        p7 += t7 * t7
    out[0] = p0
    out[1] = p1
    out[2] = p2
    out[3] = p3
    out[4] = p4
    out[5] = p5
    out[6] = p6
    out[7] = p7


###############################################################################
# Ward merge loop


@_jit()
def _ward_merges(
        vectors: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Connectivity-constrained Ward merge loop, the same merges in the same order as sklearn's
    ward_tree. Candidate pairs are owned by their larger node id (the node whose creation
    pushed them); each owner keeps them in a local heap on (inertia, j) and a global heap
    holds each owner's current minimum, so popping the global minimum is sklearn's pop.

    Parameters
    ----------
    vectors : np.ndarray of float64 - size (n, d)
        C-contiguous samples.

    indptr : np.ndarray of int64 - size (n + 1,)
        CSR row pointers of the symmetric, connected connectivity graph.

    indices : np.ndarray of int64 - size (nnz,)
        CSR column indices of that graph.

    n_nodes : int
        2n - 1 for the full tree, 2n - n_clusters to stop early.

    Returns
    -------
    tree : tuple[np.ndarray, np.ndarray, np.ndarray]
        (children, inertias, parent): children is an np.ndarray of int - size (n_nodes - n,
        2) with the (i, j) merged at each step (i the owner, before sklearn reverses them);
        inertias is an np.ndarray of float - size (n_nodes - n,) with each merge's inertia;
        parent is an np.ndarray of int - size (n_nodes,).
    """

    ### Moments: sizes, sums and centroids (a merged node reuses child i's slot) ###
    n, d = vectors.shape
    sizes = np.zeros(n_nodes)
    sizes[:n] = 1.0
    sums = vectors.copy()
    centroids = vectors.copy()  # X / 1.0 == X exactly
    slot = np.empty(n_nodes, dtype=np.int64)
    slot[:n] = np.arange(n)
    parent = np.arange(n_nodes)
    alive = np.ones(n_nodes, dtype=np.bool_)
    mark = np.full(n_nodes, -1, dtype=np.int64)
    children = np.empty((n_nodes - n, 2), dtype=np.int64)
    inertias = np.empty(n_nodes - n)

    ### Exact neighbour lists of the current clusters, contiguous per node ###
    # A list never grows after it is written: a merge only replaces or removes entries in
    # the neighbours' lists, so the pool is append-only.
    nnz = indptr[n]
    pool_size = nnz + POOL_SLACK
    neighbours = np.empty(pool_size, dtype=np.int64)
    list_start = np.zeros(n_nodes, dtype=np.int64)
    list_len = np.zeros(n_nodes, dtype=np.int64)
    used = 0
    for r in range(n):
        list_start[r] = used
        for p in range(indptr[r], indptr[r + 1]):
            if indices[p] != r:  # self loops never yield a neighbour in sklearn either
                neighbours[used] = indices[p]
                used += 1
        list_len[r] = used - list_start[r]

    ### Owner heaps (entries (j, inertia) owned by i > j) and the global heap ###
    entry_size = nnz + POOL_SLACK
    ev = np.empty(entry_size)
    ec = np.empty(entry_size, dtype=np.int64)
    owner_start = np.zeros(n_nodes, dtype=np.int64)
    owner_len = np.zeros(n_nodes, dtype=np.int64)
    n_entries = 0
    hv = np.empty(n_nodes)
    hi = np.empty(n_nodes, dtype=np.int64)
    hj = np.empty(n_nodes, dtype=np.int64)
    heap_len = 0
    for r in range(n):
        owner_start[r] = n_entries
        for p in range(indptr[r], indptr[r + 1]):
            c = indices[p]
            if c < r:  # sklearn keeps the upper triangle, (row, col) with col < row
                ev[n_entries] = _ward_dist(centroids, r, c, 1.0, 1.0, d)
                ec[n_entries] = c
                n_entries += 1
        size = n_entries - owner_start[r]
        owner_len[r] = size
        for pos in range(size // 2 - 1, -1, -1):
            _local_sift_down(ev, ec, owner_start[r], pos, size)
        if size > 0:
            hv[heap_len] = ev[owner_start[r]]
            hi[heap_len] = r
            hj[heap_len] = ec[owner_start[r]]
            heap_len += 1
    for pos in range(heap_len // 2 - 1, -1, -1):
        _global_sift_down(hv, hi, hj, pos, heap_len)

    nb = np.empty(NEIGHBOUR_BUFFER, dtype=np.int64)
    block = np.empty(WARD_BLOCK)
    for k in range(n, n_nodes):

        ### Next merge: smallest (inertia, i, j) with both ends alive ###
        while True:
            if heap_len == 0:
                raise ValueError("The connectivity graph ran out of pairs before the tree "
                                 "was complete.")
            inertia = hv[0]
            i = hi[0]
            j = hj[0]
            if alive[i] and alive[j]:
                heap_len -= 1
                if heap_len > 0:
                    hv[0] = hv[heap_len]
                    hi[0] = hi[heap_len]
                    hj[0] = hj[heap_len]
                    _global_sift_down(hv, hi, hj, 0, heap_len)
                break
            if alive[i]:  # owner alive, column dead: advance the owner past dead columns
                base = owner_start[i]
                size = _local_pop(ev, ec, base, owner_len[i])
                while size > 0 and not alive[ec[base]]:
                    size = _local_pop(ev, ec, base, size)
                owner_len[i] = size
                if size > 0:
                    hv[0] = ev[base]
                    hj[0] = ec[base]
                    _global_sift_down(hv, hi, hj, 0, heap_len)
                    continue
            heap_len -= 1  # owner dead (or exhausted): drop it with all its entries
            if heap_len > 0:
                hv[0] = hv[heap_len]
                hi[0] = hi[heap_len]
                hj[0] = hj[heap_len]
                _global_sift_down(hv, hi, hj, 0, heap_len)
        parent[i] = k
        parent[j] = k
        children[k - n, 0] = i
        children[k - n, 1] = j
        inertias[k - n] = inertia
        alive[i] = False
        alive[j] = False

        ### Moments of k ###
        si = slot[i]
        sj = slot[j]
        sizes[k] = sizes[i] + sizes[j]
        mk = sizes[k]
        for f in range(d):
            sums[si, f] = sums[si, f] + sums[sj, f]
            centroids[si, f] = sums[si, f] / mk
        slot[k] = si

        ### Neighbours of k = N(i) | N(j) minus {i, j} ###
        mark[i] = k
        mark[j] = k
        count = 0
        need = list_len[i] + list_len[j]
        if need > nb.shape[0]:
            nb = _grow(nb, max(2 * nb.shape[0], need))
        for source in (i, j):
            for p in range(list_start[source], list_start[source] + list_len[source]):
                r = neighbours[p]
                if mark[r] != k:
                    mark[r] = k
                    nb[count] = r
                    count += 1
        for e in range(count):  # in each neighbour's list: first of i / j -> k, drop second
            c = nb[e]
            start = list_start[c]
            length = list_len[c]
            replaced = False
            p = start
            while p < start + length:
                if neighbours[p] == i or neighbours[p] == j:
                    if not replaced:
                        neighbours[p] = k
                        replaced = True
                        p += 1
                    else:
                        length -= 1
                        neighbours[p] = neighbours[start + length]
                        break
                else:
                    p += 1
            list_len[c] = length
        if used + count > pool_size:
            pool_size = max(2 * pool_size, used + count)
            neighbours = _grow(neighbours, pool_size)
        list_start[k] = used
        list_len[k] = count
        neighbours[used:used + count] = nb[:count]
        used += count

        ### Inertias (k, c) into k's owner heap ###
        if n_entries + count > entry_size:
            entry_size = max(2 * entry_size, n_entries + count)
            ev = _grow(ev, entry_size)
            ec = _grow(ec, entry_size)
        base = n_entries
        owner_start[k] = base
        q = 0
        while q < count:
            if q + 8 <= count:
                _square_sums8(centroids, si, slot, nb, q, d, block)
                n_block = 8
            elif q + 4 <= count:
                _square_sums4(centroids, si, slot, nb, q, d, block)
                n_block = 4
            else:
                sc = slot[nb[q]]
                total = 0.0
                for f in range(d):
                    t = centroids[si, f] - centroids[sc, f]
                    total += t * t
                block[0] = total
                n_block = 1
            for e in range(n_block):
                mc = sizes[nb[q + e]]
                ev[n_entries] = block[e] * ((mk * mc) / (mk + mc))
                ec[n_entries] = nb[q + e]
                n_entries += 1
            q += n_block
        owner_len[k] = count
        for pos in range(count // 2 - 1, -1, -1):
            _local_sift_down(ev, ec, base, pos, count)
        if count > 0:
            hv[heap_len] = ev[base]
            hi[heap_len] = k
            hj[heap_len] = ec[base]
            _global_sift_up(hv, hi, hj, heap_len)
            heap_len += 1
    tree = (children, inertias, parent)
    return tree


###############################################################################
# Average linkage: CPython heapq on weights, sorted neighbour maps


@intrinsic
def _fma(typingctx, a, b, c):

    """
    Fused multiply-add a * b + c with a single rounding (LLVM llvm.fma), used when the
    installed sklearn's average_merge was compiled with FMA contraction.

    Parameters
    ----------
    typingctx : numba.core.typing.context.Context
        numba typing context (supplied by numba).

    a : numba.core.types.Float
        Type of the first factor.

    b : numba.core.types.Float
        Type of the second factor.

    c : numba.core.types.Float
        Type of the addend.

    Returns
    -------
    definition : tuple
        (signature, codegen) as numba intrinsics require.
    """

    signature = numba_types.float64(numba_types.float64, numba_types.float64,
                                    numba_types.float64)

    def codegen(context, builder, sig, args):

        """
        Emits the LLVM fma instruction.

        Parameters
        ----------
        context : numba.core.base.BaseContext
            numba target context.

        builder : llvmlite.ir.IRBuilder
            LLVM IR builder.

        sig : numba.core.typing.templates.Signature
            The intrinsic's signature.

        args : tuple
            LLVM values of a, b and c.

        Returns
        -------
        result : llvmlite.ir.Value
            a * b + c, rounded once.
        """

        result = builder.fma(args[0], args[1], args[2])
        return result

    definition = (signature, codegen)
    return definition


@_jit(inline="always")
def _heapq_siftdown(hw: np.ndarray, ha: np.ndarray, hb: np.ndarray, start: int,
                    pos: int) -> None:

    """
    CPython heapq._siftdown on weights only (WeightedEdge's comparison).

    Parameters
    ----------
    hw : np.ndarray of float - size (cap,)
        Heap weights.

    ha : np.ndarray of int - size (cap,)
        Heap edge ends a.

    hb : np.ndarray of int - size (cap,)
        Heap edge ends b.

    start : int
        Position the sift stops at.

    pos : int
        Position of the item to move up.

    Returns
    -------
    None
    """

    w = hw[pos]
    a = ha[pos]
    b = hb[pos]
    while pos > start:
        up = (pos - 1) >> 1
        if not w < hw[up]:
            break
        hw[pos] = hw[up]
        ha[pos] = ha[up]
        hb[pos] = hb[up]
        pos = up
    hw[pos] = w
    ha[pos] = a
    hb[pos] = b


@_jit()
def _heapq_siftup(hw: np.ndarray, ha: np.ndarray, hb: np.ndarray, pos: int, end: int) -> None:

    """
    CPython heapq._siftup (bottom-up: down to a leaf, then back up) on weights only.

    Parameters
    ----------
    hw : np.ndarray of float - size (cap,)
        Heap weights.

    ha : np.ndarray of int - size (cap,)
        Heap edge ends a.

    hb : np.ndarray of int - size (cap,)
        Heap edge ends b.

    pos : int
        Position of the item to move down.

    end : int
        Number of heap entries in use.

    Returns
    -------
    None
    """

    start = pos
    w = hw[pos]
    a = ha[pos]
    b = hb[pos]
    child = 2 * pos + 1
    while child < end:
        if child + 1 < end and not hw[child] < hw[child + 1]:
            child += 1
        hw[pos] = hw[child]
        ha[pos] = ha[child]
        hb[pos] = hb[child]
        pos = child
        child = 2 * pos + 1
    hw[pos] = w
    ha[pos] = a
    hb[pos] = b
    _heapq_siftdown(hw, ha, hb, start, pos)


@_jit()
def _append_entry(key: np.ndarray, val: np.ndarray, nxt: np.ndarray, head: np.ndarray,
                  tail: np.ndarray, node: int, slot: int, k: int, value: float) -> None:

    """
    Appends (k, value) at the tail of a node's key-sorted neighbour map (k is larger than
    every key already there, so the map stays sorted, like IntFloatDict.append).

    Parameters
    ----------
    key : np.ndarray of int - size (cap,)
        Entry keys of all maps.

    val : np.ndarray of float - size (cap,)
        Entry values of all maps.

    nxt : np.ndarray of int - size (cap,)
        Next entry of each entry, -1 at the tail.

    head : np.ndarray of int - size (n_nodes,)
        First entry of each map, -1 when empty.

    tail : np.ndarray of int - size (n_nodes,)
        Last entry of each map, -1 when empty.

    node : int
        Map to append to.

    slot : int
        Free pool position to use.

    k : int
        Key to append.

    value : float
        Edge weight to store.

    Returns
    -------
    None
    """

    key[slot] = k
    val[slot] = value
    nxt[slot] = -1
    if tail[node] == -1:
        head[node] = slot
    else:
        nxt[tail[node]] = slot
    tail[node] = slot


@_jit()
def _average_merges(
        n: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        n_nodes: int,
        use_fma: bool
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Connectivity-constrained average-linkage merge loop, the same merges in the same order
    as sklearn's linkage_tree(linkage="average"): CPython's heapq replicated on weights, and
    average_merge's sorted-map merge with its exact arithmetic.

    Parameters
    ----------
    n : int
        Number of samples.

    indptr : np.ndarray of int64 - size (n + 1,)
        Row pointers of sklearn's LIL connectivity (diagonal removed), rows in LIL order.

    indices : np.ndarray of int64 - size (nnz,)
        Column of every entry, ascending within each row.

    weights : np.ndarray of float64 - size (nnz,)
        paired_distances weight of every entry.

    n_nodes : int
        2n - 1 for the full tree, 2n - n_clusters to stop early.

    use_fma : bool
        Evaluate (n_a * a + n_b * b) / n_out as fma(n_a, a, n_b * b) / n_out.

    Returns
    -------
    tree : tuple[np.ndarray, np.ndarray, np.ndarray]
        (children, weights_merged, parent): children is an np.ndarray of int - size
        (n_nodes - n, 2) with the (a, b) of each popped edge (before sklearn reverses them);
        weights_merged is an np.ndarray of float - size (n_nodes - n,); parent is an
        np.ndarray of int - size (n_nodes,).
    """

    ### Neighbour maps (IntFloatDict) as key-sorted linked lists ###
    nnz = indptr[n]
    pool_size = nnz + POOL_SLACK
    key = np.empty(pool_size, dtype=np.int64)
    val = np.empty(pool_size)
    nxt = np.empty(pool_size, dtype=np.int64)
    head = np.full(n_nodes, -1, dtype=np.int64)
    tail = np.full(n_nodes, -1, dtype=np.int64)
    used = 0
    for r in range(n):
        for p in range(indptr[r], indptr[r + 1]):
            _append_entry(key, val, nxt, head, tail, r, used, indices[p], weights[p])
            used += 1

    ### Heap in sklearn's list order, then heapify ###
    heap_size = nnz + POOL_SLACK
    hw = np.empty(heap_size)
    ha = np.empty(heap_size, dtype=np.int64)
    hb = np.empty(heap_size, dtype=np.int64)
    heap_len = 0
    for r in range(n):
        for p in range(indptr[r], indptr[r + 1]):
            if indices[p] < r:
                hw[heap_len] = weights[p]
                ha[heap_len] = r
                hb[heap_len] = indices[p]
                heap_len += 1
    for pos in range(heap_len // 2 - 1, -1, -1):
        _heapq_siftup(hw, ha, hb, pos, heap_len)

    parent = np.arange(n_nodes)
    counts = np.zeros(n_nodes, dtype=np.int64)  # sklearn's used_node: size, 0 once merged
    counts[:n] = 1
    children = np.empty((n_nodes - n, 2), dtype=np.int64)
    weights_merged = np.empty(n_nodes - n)
    out_key = np.empty(NEIGHBOUR_BUFFER, dtype=np.int64)
    out_val = np.empty(NEIGHBOUR_BUFFER)

    for k in range(n, n_nodes):

        ### heappop until both ends are alive ###
        while True:
            if heap_len == 0:
                raise ValueError("The connectivity graph ran out of pairs before the tree "
                                 "was complete.")
            w = hw[0]
            i = ha[0]
            j = hb[0]
            heap_len -= 1
            if heap_len > 0:
                hw[0] = hw[heap_len]
                ha[0] = ha[heap_len]
                hb[0] = hb[heap_len]
                _heapq_siftup(hw, ha, hb, 0, heap_len)
            if counts[i] != 0 and counts[j] != 0:
                break
        weights_merged[k - n] = w
        parent[i] = k
        parent[j] = k
        children[k - n, 0] = i
        children[k - n, 1] = j
        n_i = counts[i]
        n_j = counts[j]
        counts[k] = n_i + n_j
        counts[i] = 0
        counts[j] = 0
        fa = float(n_i)
        fb = float(n_j)
        n_out = float(n_i + n_j)

        ### average_merge(A[i], A[j]) as a sorted merge-join, dead keys masked ###
        count = 0
        pa = head[i]
        pb = head[j]
        while pa != -1 or pb != -1:
            if pb == -1 or (pa != -1 and key[pa] < key[pb]):
                kk = key[pa]
                vv = val[pa]
                pa = nxt[pa]
            elif pa == -1 or key[pb] < key[pa]:
                kk = key[pb]
                vv = val[pb]
                pb = nxt[pb]
            else:
                kk = key[pa]
                if use_fma:
                    vv = _fma(fa, val[pa], fb * val[pb]) / n_out
                else:
                    vv = (fa * val[pa] + fb * val[pb]) / n_out
                pa = nxt[pa]
                pb = nxt[pb]
            if counts[kk] == 0:
                continue
            if count == out_key.shape[0]:
                out_key = _grow(out_key, 2 * count)
                out_val = _grow(out_val, 2 * count)
            out_key[count] = kk
            out_val[count] = vv
            count += 1

        ### A[col].append(k, d), A[k] = merged map, heappush in ascending col ###
        if used + 2 * count > pool_size:
            pool_size = max(2 * pool_size, used + 2 * count)
            key = _grow(key, pool_size)
            val = _grow(val, pool_size)
            nxt = _grow(nxt, pool_size)
        if heap_len + count > heap_size:
            heap_size = max(2 * heap_size, heap_len + count)
            hw = _grow(hw, heap_size)
            ha = _grow(ha, heap_size)
            hb = _grow(hb, heap_size)
        for e in range(count):
            _append_entry(key, val, nxt, head, tail, out_key[e], used, k, out_val[e])
            used += 1
            _append_entry(key, val, nxt, head, tail, k, used, out_key[e], out_val[e])
            used += 1
            hw[heap_len] = out_val[e]
            ha[heap_len] = k
            hb[heap_len] = out_key[e]
            heap_len += 1
            _heapq_siftdown(hw, ha, hb, 0, heap_len - 1)
        head[i] = -1  # A[i] = A[j] = 0
        head[j] = -1
    tree = (children, weights_merged, parent)
    return tree


###############################################################################
# Python wrappers


def _ward_connectivity(vectors: np.ndarray, connectivity: sparse.spmatrix
                       ) -> tuple[np.ndarray, np.ndarray]:

    """
    CSR arrays with the entries of the LIL graph sklearn's ward_tree builds
    (connectivity + connectivity.T, completed by sklearn itself if not connected).

    Parameters
    ----------
    vectors : np.ndarray of float - size (n, d)
        Samples (only used when sklearn has to complete the graph).

    connectivity : scipy.sparse.spmatrix - size (n, n)
        Connectivity matrix as passed to AgglomerativeClustering.

    Returns
    -------
    graph : tuple[np.ndarray, np.ndarray]
        (indptr, indices) of int64 - sizes (n + 1,) and (nnz,).
    """

    symmetric = sparse.csr_matrix(connectivity + connectivity.T)
    n_components, _ = connected_components(symmetric)
    if n_components > 1:  # sklearn completes the graph (and warns), exactly as it would
        symmetric = _fix_connectivity(vectors, connectivity, affinity="euclidean")[0].tocsr()
    graph = (np.ascontiguousarray(symmetric.indptr, dtype=np.int64),
             np.ascontiguousarray(symmetric.indices, dtype=np.int64))
    return graph


def _average_connectivity(vectors: np.ndarray, connectivity: sparse.spmatrix
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Weighted graph exactly as sklearn's linkage_tree prepares it: fixed connectivity,
    diagonal removed, paired Euclidean distances as weights, LIL row order.

    Parameters
    ----------
    vectors : np.ndarray of float - size (n, d)
        Samples.

    connectivity : scipy.sparse.spmatrix - size (n, n)
        Connectivity matrix as passed to AgglomerativeClustering.

    Returns
    -------
    graph : tuple[np.ndarray, np.ndarray, np.ndarray]
        (indptr, indices, weights): int64 - size (n + 1,), int64 - size (nnz,) and
        float64 - size (nnz,).
    """

    n = len(vectors)
    fixed = _fix_connectivity(vectors, connectivity, affinity="euclidean")[0].tocoo()
    off_diagonal = fixed.row != fixed.col
    fixed.row = fixed.row[off_diagonal]
    fixed.col = fixed.col[off_diagonal]
    fixed.data = paired_distances(vectors[fixed.row], vectors[fixed.col], metric="euclidean")
    lil = fixed.tolil()
    lengths = np.fromiter((len(row) for row in lil.rows), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.fromiter((c for row in lil.rows for c in row), dtype=np.int64,
                          count=indptr[-1])
    weights = np.fromiter((v for row in lil.data for v in row), dtype=np.float64,
                          count=indptr[-1])
    graph = (indptr, indices, weights)
    return graph


def _cut_tree(n_clusters: int, children: np.ndarray, n_leaves: int) -> np.ndarray:

    """
    sklearn's _hc_cut with the same heap operations, so labels are numbered identically.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to cut into.

    children : np.ndarray of int - size (n_leaves - 1, 2)
        Full merge tree in sklearn's order.

    n_leaves : int
        Number of leaves.

    Returns
    -------
    labels : np.ndarray of int - size (n_leaves,)
        Cluster label of every leaf.
    """

    nodes = [-(max(children[-1]) + 1)]
    for _ in range(n_clusters - 1):
        these_children = children[-nodes[0] - n_leaves]
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    cut_nodes = np.array([-node for node in nodes], dtype=np.int64)
    labels = _labels_from_cut(children, n_leaves, cut_nodes).astype(np.intp)
    return labels


def _merge_tree(vectors: np.ndarray, connectivity: sparse.spmatrix, n_nodes: int,
                linkage: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Builds the (possibly partial) merge tree with the numba kernel of one linkage.

    Parameters
    ----------
    vectors : np.ndarray of float64 - size (n, d)
        C-contiguous samples.

    connectivity : scipy.sparse.spmatrix - size (n, n)
        Connectivity matrix as passed to AgglomerativeClustering.

    n_nodes : int
        2n - 1 for the full tree, 2n - n_clusters to stop early.

    linkage : str
        One of "ward" or "average".

    Returns
    -------
    tree : tuple[np.ndarray, np.ndarray, np.ndarray]
        (children, distances, parent): children in sklearn's order - size (n_nodes - n, 2),
        sklearn's distances_ - size (n_nodes - n,), parent - size (n_nodes,).

    Raises
    ------
    ValueError
        If `linkage` is not "ward" or "average".
    """

    if linkage == "ward":
        indptr, indices = _ward_connectivity(vectors, connectivity)
        children, inertias, parent = _ward_merges(vectors, indptr, indices, n_nodes)
        distances = np.sqrt(2.0 * inertias)  # ward_tree's return_distance scaling
    elif linkage == "average":
        indptr, indices, weights = _average_connectivity(vectors, connectivity)
        children, distances, parent = _average_merges(len(vectors), indptr, indices, weights,
                                                      n_nodes, _average_uses_fma())
    else:
        raise ValueError(f"Unknown linkage {linkage!r}; expected 'ward' or 'average'.")
    tree = (np.ascontiguousarray(children[:, ::-1]), distances, parent)
    return tree


def constrained_linkage_labels(
        vectors: np.ndarray,
        connectivity: sparse.spmatrix,
        n_clusters: int,
        linkage: str = "ward"
        ) -> np.ndarray:

    """
    Drop-in replacement for AgglomerativeClustering(n_clusters, linkage=linkage,
    connectivity=connectivity, metric="euclidean").fit_predict(vectors), giving identical
    labels much faster. Call `numba_backend_available` once first and use sklearn if it
    returns False.

    Parameters
    ----------
    vectors : np.ndarray of float - size (n, d)
        Samples, n >= 2.

    connectivity : scipy.sparse.spmatrix - size (n, n)
        Connectivity matrix (made symmetric, and completed if not connected, as in sklearn).

    n_clusters : int
        Number of clusters, 1 <= n_clusters <= n.

    linkage : str
        One of "ward" or "average", default "ward".

    Returns
    -------
    labels : np.ndarray of int - size (n,)
        Cluster label of every sample, numbered as sklearn numbers them.

    Raises
    ------
    ValueError
        If `linkage` is not "ward" or "average", or `n_clusters` is outside [1, n].
    """

    if linkage not in CONSTRAINED_LINKAGES:
        raise ValueError(f"Unknown linkage {linkage!r}; expected 'ward' or 'average'.")
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    n = len(vectors)
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}.")

    ### sklearn's compute_full_tree="auto" ###
    full_tree = n_clusters < max(FULL_TREE_MIN_CLUSTERS, FULL_TREE_FRACTION * n)
    n_nodes = 2 * n - 1 if full_tree else 2 * n - n_clusters
    children, _, parent = _merge_tree(vectors, connectivity, n_nodes, linkage)

    if full_tree:
        labels = _cut_tree(n_clusters, children, n)
    else:
        heads = _tree_heads(parent, n)
        labels = np.searchsorted(np.unique(heads), heads)
    return labels


###############################################################################
# Self-checks against the installed sklearn


_AVERAGE_FMA = None
_BACKEND_OK = None


def _average_uses_fma() -> bool:

    """
    Finds out how the installed sklearn evaluates average_merge's weighted mean: plain
    (n_a * a + n_b * b) / n_out, or with a fused multiply-add (clang's default contraction on
    arm64 builds). Probed once, against exact rational arithmetic.

    Returns
    -------
    use_fma : bool
        True for fma(n_a, a, n_b * b) / n_out.

    Raises
    ------
    RuntimeError
        If sklearn matches neither form.
    """

    global _AVERAGE_FMA
    if _AVERAGE_FMA is None:
        from fractions import Fraction

        rng = np.random.default_rng(CHECK_SEED)
        keys = np.arange(CHECK_MERGE_KEYS, dtype=np.intp)
        va, vb = rng.random(CHECK_MERGE_KEYS), rng.random(CHECK_MERGE_KEYS)
        n_a, n_b = CHECK_MERGE_SIZES
        merged = sk_hierarchical.average_merge(IntFloatDict(keys, va), IntFloatDict(keys, vb),
                                               np.ones(CHECK_MERGE_KEYS, dtype=np.intp),
                                               n_a, n_b)
        sk_values = np.array([value for _, value in merged])
        n_out = float(n_a + n_b)
        plain = (n_a * va + n_b * vb) / n_out
        fused = np.array([float(Fraction(float(n_a)) * Fraction(a) + Fraction(float(n_b) * b))
                          for a, b in zip(va, vb)]) / n_out
        if np.array_equal(sk_values, plain):
            _AVERAGE_FMA = False
        elif np.array_equal(sk_values, fused):
            _AVERAGE_FMA = True
        else:
            raise RuntimeError("sklearn's average_merge matches neither plain nor fused "
                               "multiply-add arithmetic.")
    use_fma = _AVERAGE_FMA
    return use_fma


def _ward_kernels_match() -> bool:

    """
    Compares the Ward distance kernels (single, 4- and 8-wide) bit for bit with sklearn's
    compute_ward_dist on moments with non-trivial cluster sizes.

    Returns
    -------
    match : bool
        True when every value is identical.
    """

    rng = np.random.default_rng(CHECK_SEED)
    sizes = rng.integers(1, CHECK_MAX_COUNT, CHECK_NODES).astype(np.float64)
    sums = rng.standard_normal((CHECK_NODES, CHECK_FEATURES)) * sizes[:, None] * CHECK_SCALE
    rows = rng.integers(0, CHECK_NODES, CHECK_PAIRS).astype(np.intp)
    cols = rng.integers(0, CHECK_NODES, CHECK_PAIRS).astype(np.intp)
    sk_values = np.empty(CHECK_PAIRS)
    sk_hierarchical.compute_ward_dist(sizes, sums, rows, cols, sk_values)

    centroids = sums / sizes[:, None]
    slot = np.arange(CHECK_NODES, dtype=np.int64)
    values = np.array([_ward_dist(centroids, r, c, sizes[r], sizes[c], CHECK_FEATURES)
                       for r, c in zip(rows, cols)])
    match = bool(np.array_equal(sk_values, values))
    block = np.empty(WARD_BLOCK)
    for start in range(0, CHECK_PAIRS - WARD_BLOCK + 1, WARD_BLOCK):
        nb = cols[start:start + WARD_BLOCK].astype(np.int64)
        single = np.array([_ward_dist(centroids, rows[start], c, 1.0, 1.0, CHECK_FEATURES)
                           for c in nb]) * 2.0  # weight 1/2 for unit sizes: exact rescale
        _square_sums8(centroids, rows[start], slot, nb, 0, CHECK_FEATURES, block)
        match = match and np.array_equal(block, single)
        _square_sums4(centroids, rows[start], slot, nb, 0, CHECK_FEATURES, block)
        match = match and np.array_equal(block[:4], single[:4])
    return match


def _small_problems_match() -> bool:

    """
    Clusters small grid-graph problems (continuous data and data full of exact ties) with
    both linkages and compares the labels with sklearn's, including an early-stopped tree.

    Returns
    -------
    match : bool
        True when every labelling is identical.
    """

    rows, cols = CHECK_GRID
    n = rows * cols
    grid = np.arange(n).reshape(rows, cols)
    edges = np.r_[np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()],
                  np.c_[grid[:-1].ravel(), grid[1:].ravel()]]
    connectivity = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                     shape=(n, n)).tocsr()
    connectivity = connectivity + connectivity.T
    rng = np.random.default_rng(CHECK_SEED)
    problems = (rng.random((n, CHECK_DIMENSION)),
                rng.integers(0, 2, (n, CHECK_DIMENSION)).astype(float))  # many ties
    match = True
    for vectors in problems:
        for linkage in CONSTRAINED_LINKAGES:
            for n_clusters in CHECK_CLUSTERS:
                sk_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                                    connectivity=connectivity
                                                    ).fit_predict(vectors)
                labels = constrained_linkage_labels(vectors, connectivity, n_clusters, linkage)
                match = match and np.array_equal(sk_labels, labels)
    return match


def numba_backend_available() -> bool:

    """
    Decides once per process whether `constrained_linkage_labels` can stand in for sklearn:
    the Ward kernels must match compute_ward_dist bit for bit, average_merge's arithmetic
    must be identified, and small problems must cluster identically. Compiles (or loads
    from the cache) every kernel on the way. Any failure means "use sklearn".

    Returns
    -------
    available : bool
        True when the numba backend reproduces the installed sklearn.
    """

    global _BACKEND_OK
    if _BACKEND_OK is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumbaWarning)  # e.g. cache could not be saved
                _average_uses_fma()  # raises if sklearn's arithmetic cannot be identified
                _BACKEND_OK = bool(_ward_kernels_match() and _small_problems_match())
        except Exception:
            _BACKEND_OK = False
    available = _BACKEND_OK
    return available
