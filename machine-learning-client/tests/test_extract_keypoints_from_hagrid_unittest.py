"""Unit tests for extract_keypoints_from_hagrid module."""

# pylint: disable=import-error,too-few-public-methods

import unittest
import numpy as np

from src import extract_keypoints_from_hagrid as hagrid_mod


class FakeLandmark:
    """Single x,y,z landmark."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    """21-keypoint hand structure (MediaPipe-like)."""

    def __init__(self, n_points: int = 21):
        self.landmark = [
            FakeLandmark(x=i * 0.1, y=i * 0.01, z=i * 0.001) for i in range(n_points)
        ]


class TestExtractLandmarkVector(unittest.TestCase):
    """Tests for extract_landmark_vector."""

    def test_vector_shape_and_dtype(self):
        """Output shape = NUM_LANDMARKS * DIM_PER_LM, dtype = float32."""
        fake = FakeHandLandmarks()
        vec = hagrid_mod.extract_landmark_vector(fake)

        expected_dim = hagrid_mod.NUM_LANDMARKS * hagrid_mod.DIM_PER_LM

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (expected_dim,))
        self.assertEqual(vec.dtype, np.float32)
