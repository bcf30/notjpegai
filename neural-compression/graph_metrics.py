"""
Learned Geometric Boundary Topology (LGBT) metric module.

Quantifies structural distortion in neural image compression by comparing
graph representations of image skeletons. LGBT measures the ratio of false
edges — structural elements present in the reconstructed image but not in
the original — to total edges in the reconstruction.

Pipeline: Image → Skeleton → Graph → LGBT comparison
"""

from typing import Any, Dict, Tuple, Union

import cv2
from cv2 import ximgproc
import networkx as nx
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.spatial import cKDTree


def get_skeleton(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract structural skeleton from an image.

    Converts an image to a one-pixel-wide binary skeleton via:
    grayscale conversion → Gaussian blur → Canny edge detection
    (with Otsu-derived thresholds) → morphological thinning.

    Args:
        image: PIL Image or HWC uint8 NumPy array [0, 255].

    Returns:
        Binary skeleton as 2D NumPy array (0=background, 1=skeleton).

    Raises:
        TypeError: If input is not a PIL Image or NumPy array.
    """
    # Accept PIL Image or NumPy array; reject everything else.
    if isinstance(image, Image.Image):
        arr = np.array(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Expected PIL Image or NumPy array, got {type(image)}")

    # --- 2.1 Convert to grayscale if RGB (3-channel) ---
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # --- 2.2 Apply Gaussian blur (σ=1.0) for noise suppression ---
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # --- 2.3 Compute Otsu threshold on gradient magnitude ---
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    mag_max = magnitude.max()
    if mag_max == 0:
        # Flat image — no edges at all.
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    magnitude_uint8 = (magnitude / mag_max * 255).astype(np.uint8)
    high_thresh_val, _ = cv2.threshold(magnitude_uint8, 0, 255, cv2.THRESH_OTSU)
    low_thresh = high_thresh_val * 0.5

    # --- 2.4 Apply Canny edge detection with Otsu-derived thresholds ---
    edges = cv2.Canny(blurred, low_thresh, high_thresh_val)

    # --- 2.5 Perform morphological thinning to one-pixel width ---
    thinned = ximgproc.thinning(edges)

    # --- 2.6 Return binary skeleton (0=background, 1=skeleton) ---
    skeleton = (thinned > 0).astype(np.uint8)
    return skeleton


def skeleton_to_graph(
    skeleton: np.ndarray,
    max_link_distance: float = 50.0,
) -> nx.Graph:
    """Convert binary skeleton to sparse graph.

    Identifies all skeleton pixels as nodes, builds a KD-tree from their
    positions, queries the single nearest neighbor for each node, and adds
    an edge only if the distance is below *max_link_distance*.

    Args:
        skeleton: Binary 2D array (0=background, 1=skeleton).
        max_link_distance: Maximum Euclidean distance (pixels) for linking
            two nodes with an edge.

    Returns:
        networkx Graph with nodes containing a ``'pos'`` attribute
        storing ``(y, x)`` coordinates.
    """
    # --- 3.1 Identify ALL skeleton pixels as nodes (y, x) ---
    ys, xs = np.where(skeleton == 1)
    positions = np.column_stack((ys, xs)) if len(ys) > 0 else np.empty((0, 2))

    # --- 3.2 Create networkx Graph with 'pos' attribute ---
    G = nx.Graph()
    for i, (y, x) in enumerate(positions):
        G.add_node(i, pos=(int(y), int(x)))

    # Edge case: fewer than 2 nodes → no edges possible
    if len(positions) < 2:
        return G

    # --- 3.3 Build cKDTree from node positions ---
    tree = cKDTree(positions)

    # --- 3.4 Query single nearest neighbor for each node ---
    # k=2 because the closest match (distance 0) is the node itself
    dists, indices = tree.query(positions, k=2)

    # --- 3.5 Add edge only if distance < max_link_distance ---
    for i in range(len(positions)):
        nn_dist = dists[i, 1]
        nn_idx = indices[i, 1]
        if nn_dist < max_link_distance:
            G.add_edge(i, int(nn_idx))

    return G


def calculate_lgbt(
    graph_orig: nx.Graph,
    graph_recon: nx.Graph,
    tolerance: float = 5.0,
) -> Dict[str, Any]:
    """Calculate Learned Geometric Boundary Topology between two graphs.

    Compares edge sets using fuzzy matching via a KD-tree built from the
    original graph's node positions.  An edge in the reconstructed graph
    is *true* when both of its endpoints are within *tolerance* pixels of
    nodes in the original graph; otherwise it is *false*.

    Args:
        graph_orig: Graph from the original image.
        graph_recon: Graph from the reconstructed image.
        tolerance: Pixel tolerance for fuzzy edge matching.

    Returns:
        Dict with keys:
            - ``'lgbt'``: float — ratio of false edges to total edges.
            - ``'false_edges'``: int — edges in recon but not in orig.
            - ``'true_edges'``: int — edges present in both graphs.
            - ``'orig_nodes'``: int — node count in original graph.
            - ``'recon_nodes'``: int — node count in reconstructed graph.
    """
    orig_nodes = graph_orig.number_of_nodes()
    recon_nodes = graph_recon.number_of_nodes()

    # --- 4.6 Handle edge cases: empty graphs, <2 nodes ---
    recon_edges = list(graph_recon.edges())
    if not recon_edges or recon_nodes < 2:
        return {
            "lgbt": 0.0,
            "false_edges": 0,
            "true_edges": 0,
            "orig_nodes": orig_nodes,
            "recon_nodes": recon_nodes,
        }

    # --- 4.1 Extract edge endpoint coordinates from both graphs ---
    orig_positions = np.array(
        [graph_orig.nodes[n]["pos"] for n in graph_orig.nodes()]
    ) if orig_nodes > 0 else np.empty((0, 2))

    # --- 4.2 Build KD-tree from original graph node positions ---
    # If original graph is empty, all recon edges are false.
    if orig_nodes == 0:
        total = len(recon_edges)
        return {
            "lgbt": 1.0 if total > 0 else 0.0,
            "false_edges": total,
            "true_edges": 0,
            "orig_nodes": 0,
            "recon_nodes": recon_nodes,
        }

    orig_tree = cKDTree(orig_positions)

    # --- 4.3 / 4.4 For each recon edge, classify as true or false ---
    true_edges = 0
    false_edges = 0

    for u, v in recon_edges:
        pos_u = np.array(graph_recon.nodes[u]["pos"], dtype=float)
        pos_v = np.array(graph_recon.nodes[v]["pos"], dtype=float)

        # Dynamic tolerance: max(base tolerance, 10% of edge length)
        edge_length = np.linalg.norm(pos_u - pos_v)
        dyn_tolerance = max(tolerance, edge_length * 0.1)

        # Query KD-tree for each endpoint
        dist_u, _ = orig_tree.query(pos_u)
        dist_v, _ = orig_tree.query(pos_v)

        if dist_u <= dyn_tolerance and dist_v <= dyn_tolerance:
            true_edges += 1
        else:
            false_edges += 1

    # --- 4.5 Compute LGBT = false_edges / (true_edges + false_edges) ---
    total = true_edges + false_edges
    lgbt = false_edges / total if total > 0 else 0.0

    # --- 4.7 Return result dict ---
    return {
        "lgbt": lgbt,
        "false_edges": false_edges,
        "true_edges": true_edges,
        "orig_nodes": orig_nodes,
        "recon_nodes": recon_nodes,
    }


def evaluate_structural_integrity(
    pil_orig: Union[Image.Image, np.ndarray],
    pil_recon: Union[Image.Image, np.ndarray],
    tolerance: float = 5.0,
    max_link_distance: float = 50.0,
) -> Dict[str, Any]:
    """Evaluate structural integrity between original and reconstructed images.

    High-level wrapper that runs the full LGBT pipeline: skeleton extraction,
    graph construction, and LGBT calculation.

    Args:
        pil_orig: Original image (PIL Image or HWC uint8 NumPy array).
        pil_recon: Reconstructed image (PIL Image or HWC uint8 NumPy array).
        tolerance: Pixel tolerance for fuzzy edge matching.
        max_link_distance: Maximum distance for edge linking in graph
            construction.

    Returns:
        Dict with LGBT results (same schema as :func:`calculate_lgbt`).

    Raises:
        TypeError: If inputs are not PIL Images or NumPy arrays.
        ValueError: If images have different dimensions.
    """
    # --- 5.2 Validate input types ---
    for name, img in [("orig", pil_orig), ("recon", pil_recon)]:
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(
                f"Expected PIL Image or NumPy array, got {type(img)}"
            )

    # --- 5.1 Convert to arrays and validate dimensions match ---
    def _to_array(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.array(img)
        return img

    arr_orig = _to_array(pil_orig)
    arr_recon = _to_array(pil_recon)

    h1, w1 = arr_orig.shape[:2]
    h2, w2 = arr_recon.shape[:2]
    if (h1, w1) != (h2, w2):
        raise ValueError(
            f"Image dimensions must match: orig {h1}x{w1}, recon {h2}x{w2}"
        )

    # --- 5.3 Convert both images to skeletons ---
    skel_orig = get_skeleton(pil_orig)
    skel_recon = get_skeleton(pil_recon)

    # --- 5.4 Convert skeletons to graphs ---
    graph_orig = skeleton_to_graph(skel_orig, max_link_distance=max_link_distance)
    graph_recon = skeleton_to_graph(skel_recon, max_link_distance=max_link_distance)

    # --- 5.5 Call calculate_lgbt and return results ---
    return calculate_lgbt(graph_orig, graph_recon, tolerance=tolerance)
