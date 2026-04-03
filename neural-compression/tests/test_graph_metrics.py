"""
Minimal tests for the LGBT (Learned Geometric Boundary Topology) metric module.

Four tests covering:
1. Skeleton-to-graph round trip
2. Identical images produce LGBT = 0.0
3. Empty/black images handled gracefully
4. Dimension mismatch raises ValueError
"""

import sys
import os

import numpy as np
import pytest
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_metrics import (
    get_skeleton,
    skeleton_to_graph,
    evaluate_structural_integrity,
)


class TestSkeletonToGraphRoundTrip:
    """Test 7.1: Synthetic cross skeleton, verify nodes and edges."""

    def test_cross_skeleton_produces_nodes_and_edges(self):
        """Create a simple cross-shaped skeleton and verify graph extraction."""
        # Create a 21x21 binary image with a cross skeleton
        # Vertical line: column 10, rows 5-15
        # Horizontal line: row 10, columns 5-15
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        skeleton[5:16, 10] = 1  # vertical line
        skeleton[10, 5:16] = 1  # horizontal line

        # Convert to graph
        graph = skeleton_to_graph(skeleton, max_link_distance=50.0)

        # Verify nodes exist (should have many nodes from all skeleton pixels)
        assert graph.number_of_nodes() > 0, "Graph should have nodes"

        # Verify edges exist (nearest-neighbor linking should create edges)
        assert graph.number_of_edges() > 0, "Graph should have edges"

        # Verify all nodes have 'pos' attribute
        for node_id in graph.nodes():
            pos = graph.nodes[node_id].get('pos')
            assert pos is not None, f"Node {node_id} should have 'pos' attribute"
            assert len(pos) == 2, f"Position should be (y, x) tuple"
            y, x = pos
            assert 0 <= y < 21 and 0 <= x < 21, f"Position {pos} should be within image bounds"


class TestIdenticalImagesLGBTZero:
    """Test 7.2: Same image twice, assert LGBT == 0.0."""

    def test_identical_images_produce_zero_lgbt(self):
        """When comparing an image with itself, LGBT should be 0.0."""
        # Create a simple test image with some structure
        # A white rectangle on black background
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:44, 20:44] = 255  # white square

        # Compare with itself
        result = evaluate_structural_integrity(img, img)

        # LGBT should be 0.0 (no false edges)
        assert result['lgbt'] == 0.0, f"LGBT should be 0.0 for identical images, got {result['lgbt']}"
        assert result['false_edges'] == 0, "False edges should be 0 for identical images"
        assert result['true_edges'] >= 0, "True edges should be non-negative"


class TestEmptyImageLGBTZero:
    """Test 7.3: Black image, assert LGBT == 0.0, no exceptions."""

    def test_empty_black_image_returns_zero_lgbt(self):
        """A completely black image should produce LGBT = 0.0 without exceptions."""
        # Create a completely black image
        black_img = np.zeros((64, 64), dtype=np.uint8)

        # Should not raise any exceptions
        result = evaluate_structural_integrity(black_img, black_img)

        # LGBT should be 0.0 (no edges in either graph)
        assert result['lgbt'] == 0.0, f"LGBT should be 0.0 for empty images, got {result['lgbt']}"
        assert result['false_edges'] == 0, "False edges should be 0 for empty images"
        assert result['true_edges'] == 0, "True edges should be 0 for empty images"
        assert result['orig_nodes'] == 0, "Original graph should have 0 nodes"
        assert result['recon_nodes'] == 0, "Reconstructed graph should have 0 nodes"


class TestDimensionMismatchRaises:
    """Test 7.4: Different shaped images, assert ValueError."""

    def test_dimension_mismatch_raises_value_error(self):
        """Images with different dimensions should raise ValueError."""
        # Create two images with different dimensions
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((32, 32), dtype=np.uint8)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            evaluate_structural_integrity(img1, img2)

        # Verify error message mentions dimension mismatch
        error_msg = str(exc_info.value)
        assert "dimension" in error_msg.lower() or "match" in error_msg.lower(), \
            f"Error message should mention dimension mismatch: {error_msg}"

    def test_dimension_mismatch_different_width(self):
        """Images with different widths should raise ValueError."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 48), dtype=np.uint8)

        with pytest.raises(ValueError):
            evaluate_structural_integrity(img1, img2)

    def test_dimension_mismatch_different_height(self):
        """Images with different heights should raise ValueError."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((48, 64), dtype=np.uint8)

        with pytest.raises(ValueError):
            evaluate_structural_integrity(img1, img2)


class TestInputTypeHandling:
    """Additional tests for input type handling."""

    def test_pil_image_input(self):
        """PIL Images should be accepted as input."""
        # Create PIL images
        img = Image.new('RGB', (64, 64), color='white')

        # Should not raise
        result = evaluate_structural_integrity(img, img)
        assert 'lgbt' in result
        assert isinstance(result['lgbt'], float)

    def test_mixed_input_types(self):
        """Mixed PIL and NumPy inputs should work."""
        pil_img = Image.new('L', (64, 64), color=128)
        np_img = np.full((64, 64), 128, dtype=np.uint8)

        # Should not raise
        result = evaluate_structural_integrity(pil_img, np_img)
        assert 'lgbt' in result