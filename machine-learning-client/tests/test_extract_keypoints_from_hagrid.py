"""Unit tests for extract_keypoints_from_hagrid module."""

# pylint: disable=import-error,too-few-public-methods
import tempfile
import shutil
import unittest
from unittest import mock
from pathlib import Path

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


class TestCollectImagePaths(unittest.TestCase):
    """Tests for collect_image_paths (lines 36-68)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir)
        (self.root_path / "palm").mkdir()
        (self.root_path / "palm" / "img1.jpg").touch()
        (self.root_path / "palm" / "img2.png").touch()
        (self.root_path / "fist").mkdir()
        (self.root_path / "fist" / "img3.jpeg").touch()

        (self.root_path / "empty_class").mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_happy_path_collects_files(self):
        classes = ["palm", "fist"]
        samples = hagrid_mod.collect_image_paths(
            self.root_path, classes, max_per_class=100
        )

        self.assertEqual(len(samples), 3)

        labels = {label for (path, label) in samples}
        self.assertEqual(labels, {0, 1})

    def test_max_per_class_limit(self):
        classes = ["palm"]
        samples = hagrid_mod.collect_image_paths(
            self.root_path, classes, max_per_class=1
        )

        self.assertEqual(len(samples), 1)

    def test_missing_class_folder_raises_error(self):
        classes = ["stop"]

        with self.assertRaises(FileNotFoundError):
            hagrid_mod.collect_image_paths(self.root_path, classes, max_per_class=100)

    def test_empty_class_folder_raises_error(self):
        classes = ["empty_class"]

        with self.assertRaises(RuntimeError):
            hagrid_mod.collect_image_paths(self.root_path, classes, max_per_class=100)

    @mock.patch("pathlib.Path.is_file", return_value=True)
    @mock.patch("src.extract_keypoints_from_hagrid.random.shuffle")
    @mock.patch("pathlib.Path.rglob")
    @mock.patch("pathlib.Path.exists")
    def test_collect_paths_logic(
        self, mock_exists, mock_rglob, mock_shuffle, _mock_is_file
    ):  # pylint: disable=unused-argument
        """Test collection, filtering, and shuffling of image paths."""

        mock_exists.return_value = True

        mock_paths = [
            Path("fake/palm/1.jpg"),
            Path("fake/palm/2.png"),
            Path("fake/palm/3.txt"),
            Path("fake/palm/4.JPG"),
            Path("fake.zip"),
        ]
        mock_rglob.return_value = mock_paths

        samples = hagrid_mod.collect_image_paths(Path("fake/"), ["palm"], 10)

        self.assertEqual(len(samples), 3)

        self.assertIn((Path("fake/palm/1.jpg"), 0), samples)
        self.assertIn((Path("fake/palm/2.png"), 0), samples)
        self.assertIn((Path("fake/palm/4.JPG"), 0), samples)

        self.assertEqual(mock_shuffle.call_count, 2)

    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_collect_paths_dir_not_found(
        self, _mock_exists
    ):  # pylint: disable=unused-argument
        """Test that it raises FileNotFoundError if class dir doesn't exist."""

        with self.assertRaises(FileNotFoundError):
            hagrid_mod.collect_image_paths(Path("fake/"), ["palm"], 10)


class TestMainFunction(unittest.TestCase):
    """Tests for the main() function logic, using heavy mocking."""

    @patch("src.extract_keypoints_from_hagrid.collect_image_paths")
    @patch("src.extract_keypoints_from_hagrid.cv2.imread")
    @patch("src.extract_keypoints_from_hagrid.mp_hands.Hands")
    @patch("src.extract_keypoints_from_hagrid.np.save")
    @patch("src.extract_keypoints_from_hagrid.Path.mkdir")
    @patch("builtins.open")
    def test_main_handles_image_read_failure(
        self,
        mock_open,
        mock_mkdir,
        mock_np_save,
        mock_hands_class,
        mock_imread,
        mock_collect,
    ):

        mock_collect.return_value = [(Path("fake/img.jpg"), 0)]

        mock_imread.return_value = None

        mock_hands_instance = mock_hands_class.return_value.__enter__.return_value
        mock_hands_instance.process.return_value = None

        with self.assertRaises(RuntimeError) as context:
            hagrid_mod.main()

        self.assertIn("No samples with detected hands", str(context.exception))

        mock_imread.assert_called_with("fake/img.jpg")

    @patch("src.extract_keypoints_from_hagrid.collect_image_paths")
    @patch("src.extract_keypoints_from_hagrid.cv2.imread")
    @patch("src.extract_keypoints_from_hagrid.mp_hands.Hands")
    @patch("src.extract_keypoints_from_hagrid.np.save")
    @patch("src.extract_keypoints_from_hagrid.Path.mkdir")
    @patch("builtins.open")
    def test_main_handles_no_hand_detected(
        self,
        mock_open,
        mock_mkdir,
        mock_np_save,
        mock_hands_class,
        mock_imread,
        mock_collect,
    ):

        mock_collect.return_value = [(Path("fake/img.jpg"), 0)]

        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_hands_instance = mock_hands_class.return_value.__enter__.return_value

        mock_process_result = unittest.mock.MagicMock()
        mock_process_result.multi_hand_landmarks = None
        mock_hands_instance.process.return_value = mock_process_result

        with self.assertRaises(RuntimeError) as context:
            hagrid_mod.main()

        self.assertIn("No samples with detected hands", str(context.exception))
