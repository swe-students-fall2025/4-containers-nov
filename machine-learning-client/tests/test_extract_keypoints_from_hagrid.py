"""Unit tests for extract_keypoints_from_hagrid module."""

# pylint: disable=import-error,too-few-public-methods

import unittest
import numpy as np

from unittest import mock
from pathlib import Path

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


class TestCollectImagePaths(unittest.TestCase):
    """Tests for collect_image_paths (lines 36-68)."""

    @mock.patch("src.extract_keypoints_from_hagrid.random.shuffle")
    @mock.patch("pathlib.Path.rglob")
    @mock.patch("pathlib.Path.exists")
    def test_collect_paths_logic(self, mock_exists, mock_rglob, mock_shuffle):
        """Test collection, filtering, and shuffling of image paths."""

        mock_exists.return_value = True

        mock_paths = [
            Path("fake/palm/1.jpg"),
            Path("fake/palm/2.png"),
            Path("fake/palm/3.txt"),
            Path("fake/palm/4.JPG"),
            Path("fake/palm/archive.zip"),
        ]
        mock_rglob.return_value = mock_paths

        samples = hagrid_mod.collect_image_paths(Path("fake/"), ["palm"], 10)

        self.assertEqual(len(samples), 3)

        self.assertIn((Path("fake/palm/1.jpg"), 0), samples)
        self.assertIn((Path("fake/palm/2.png"), 0), samples)
        self.assertIn((Path("fake/palm/4.JPG"), 0), samples)

        self.assertEqual(mock_shuffle.call_count, 2)

    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_collect_paths_dir_not_found(self, mock_exists):
        """Test that it raises FileNotFoundError if class dir doesn't exist."""

        with self.assertRaises(FileNotFoundError):
            hagrid_mod.collect_image_paths(Path("fake/"), ["palm"], 10)
