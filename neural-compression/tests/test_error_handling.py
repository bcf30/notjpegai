"""Tests for Task 6: Dimension validation and error handling in graph_metrics.

Verifies:
- 6.1 TypeError for non-PIL/non-NumPy input
- 6.2 ValueError for dimension mismatch
- 6.3 Handle empty/black images gracefully (return LGBT=0.0)
- 6.4 Handle graphs with <2 nodes (return LGBT=0.0)
"""

import networkx as nx
import numpy as np
import pytest
from PIL import Image

from graph_metrics import (
    calculate_lgbt,
    evaluate_structural_integrity,
    get_skeleton,
    skeleton_to_graph,
)


# --- 6.1 TypeError for non-PIL/non-NumPy input ---

class TestTypeErrorForInvalidInput:
    """get_skeleton and evaluate_structural_integrity raise TypeError
    when given something other than a PIL Image or NumPy array."""

    def test_get_skeleton_rejects_string(self):
        with pytest.raises(TypeError, match="Expected PIL Image or NumPy array"):
            get_skeleton("not_an_image")

    def test_get_skeleton_rejects_list(self):
        with pytest.raises(TypeError, match="Expected PIL Image or NumPy array"):
            get_skeleton([[1, 2], [3, 4]])

    def test_get_skeleton_rejects_int(self):
        with pytest.raises(TypeError, match="Expected PIL Image or NumPy array"):
            get_skeleton(42)

    def test_evaluate_rejects_non_image_orig(self):
        valid = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(TypeError, match="Expected PIL Image or NumPy array"):
            evaluate_structural_integrity("bad", valid)

    def test_evaluate_rejects_non_image_recon(self):
        valid = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(TypeError, match="Expected PIL Image or NumPy array"):
            evaluate_structural_integrity(valid, 123)


# --- 6.2 ValueError for dimension mismatch ---

class TestValueErrorForDimensionMismatch:
    """evaluate_structural_integrity raises ValueError when image
    dimensions don't match."""

    def test_different_height(self):
        orig = np.zeros((32, 64, 3), dtype=np.uint8)
        recon = np.zeros((48, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image dimensions must match"):
            evaluate_structural_integrity(orig, recon)

    def test_different_width(self):
        orig = np.zeros((32, 64, 3), dtype=np.uint8)
        recon = np.zeros((32, 48, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image dimensions must match"):
            evaluate_structural_integrity(orig, recon)

    def test_pil_dimension_mismatch(self):
        orig = Image.new("RGB", (64, 32))
        recon = Image.new("RGB", (48, 32))
        with pytest.raises(ValueError, match="Image dimensions must match"):
            evaluate_structural_integrity(orig, recon)


# --- 6.3 Handle empty/black images gracefully (return LGBT=0.0) ---

class TestBlackImageGraceful:
    """Black / empty images should produce LGBT=0.0 without exceptions."""

    def test_black_numpy_images(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        result = evaluate_structural_integrity(black, black)
        assert result["lgbt"] == 0.0

    def test_black_pil_images(self):
        black = Image.new("RGB", (32, 32), color=(0, 0, 0))
        result = evaluate_structural_integrity(black, black)
        assert result["lgbt"] == 0.0

    def test_skeleton_of_black_image_is_empty(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        skel = get_skeleton(black)
        assert skel.sum() == 0

    def test_graph_from_empty_skeleton(self):
        empty_skel = np.zeros((32, 32), dtype=np.uint8)
        g = skeleton_to_graph(empty_skel)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0


# --- 6.4 Handle graphs with <2 nodes (return LGBT=0.0) ---

class TestGraphWithFewNodes:
    """calculate_lgbt returns LGBT=0.0 when the reconstructed graph
    has fewer than 2 nodes."""

    def test_both_empty_graphs(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        result = calculate_lgbt(g1, g2)
        assert result["lgbt"] == 0.0
        assert result["false_edges"] == 0
        assert result["true_edges"] == 0

    def test_recon_single_node(self):
        g_orig = nx.Graph()
        g_orig.add_node(0, pos=(10, 10))
        g_recon = nx.Graph()
        g_recon.add_node(0, pos=(10, 10))
        result = calculate_lgbt(g_orig, g_recon)
        assert result["lgbt"] == 0.0

    def test_recon_zero_nodes_orig_has_nodes(self):
        g_orig = nx.Graph()
        g_orig.add_node(0, pos=(5, 5))
        g_orig.add_node(1, pos=(10, 10))
        g_orig.add_edge(0, 1)
        g_recon = nx.Graph()
        result = calculate_lgbt(g_orig, g_recon)
        assert result["lgbt"] == 0.0
