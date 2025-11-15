import unittest
from unittest import mock

import numpy as np
import torch

# Import module under test
from src import live_mediapipe_mlp as lm


# ---- Fake landmark structures (match MediaPipe format) ----
class FakeLandmark:
    """Single landmark with x,y,z values."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    """21 MediaPipe hand landmarks."""

    def __init__(self, n_points: int = 21):
        self.landmark = [
            FakeLandmark(x=i * 0.1, y=i * 0.01, z=i * 0.001)
            for i in range(n_points)
        ]


# ---- MLP model tests ----
class TestGestureMLP(unittest.TestCase):
    def test_forward_shape(self):
        """Output shape: (batch_size, num_classes)."""
        input_dim = 63
        num_classes = 8
        model = lm.GestureMLP(input_dim=input_dim, num_classes=num_classes)

        x = torch.randn(4, input_dim)
        out = model(x)

        self.assertEqual(out.shape, (4, num_classes))

    def test_forward_batch_one(self):
        """Model works with batch_size=1."""
        input_dim = 63
        num_classes = 8
        model = lm.GestureMLP(input_dim=input_dim, num_classes=num_classes)

        x = torch.randn(1, input_dim)
        out = model(x)

        self.assertEqual(out.shape, (1, num_classes))


# ---- Keypoint extraction tests ----
class TestExtractKeypoints(unittest.TestCase):
    def test_extract_keypoints_shape_dtype(self):
        """Returned vector: shape (63,), dtype float32."""
        fake_hand = FakeHandLandmarks()
        vec = lm.extract_keypoints(fake_hand)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (63,))
        self.assertEqual(vec.dtype, np.float32)


# ---- Model loading tests ----
class TestLoadModel(unittest.TestCase):
    @mock.patch("src.live_mediapipe_mlp.torch.load")
    def test_load_model_from_checkpoint(self, mock_torch_load):
        """load_model() reads checkpoint fields and restores model."""
        input_dim = 63
        num_classes = 3
        class_names = ["palm", "fist", "like"]

        tmp = lm.GestureMLP(input_dim, num_classes)
        state = tmp.state_dict()

        mock_torch_load.return_value = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "model_state_dict": state,
        }

        model, loaded = lm.load_model()

        self.assertIsInstance(model, lm.GestureMLP)
        self.assertEqual(loaded, class_names)

        x = torch.randn(2, input_dim)
        out = model(x)
        self.assertEqual(out.shape, (2, num_classes))


# ---- Mongo insert logic tests ----
class TestMongoInsertLogic(unittest.TestCase):
    def test_insert_event_structure(self):
        """Fake collection should capture inserted event."""
        inserted = []

        class FakeCollection:
            def insert_one(self, event):
                inserted.append(event)

        with mock.patch.object(lm, "gesture_collection", FakeCollection(), create=True):
            event = {
                "timestamp": "2025-11-14T12:00:00Z",
                "gesture": "palm",
                "confidence": 0.91,
                "handedness": "Right",
            }
            lm.gesture_collection.insert_one(event)

        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0]["gesture"], "palm")
        self.assertGreaterEqual(inserted[0]["confidence"], 0.9)


class TestInitDb(unittest.TestCase):
    @mock.patch("src.live_mediapipe_mlp.MongoClient")
    def test_init_db_sets_globals_once(self, mock_mongo_client):
        """init_db should initialize Mongo only once and correctly set global variables."""

        # Use the MagicMock client created by the patch
        fake_client = mock_mongo_client.return_value
        fake_db = mock.MagicMock()
        fake_collection = mock.MagicMock()

        # client["handsense"] -> db
        def client_getitem(name):
            if name == "handsense":
                return fake_db
            return None

        fake_client.__getitem__.side_effect = client_getitem

        # db["gesture_events"] -> collection
        def db_getitem(name):
            if name == "gesture_events":
                return fake_collection
            return None

        fake_db.__getitem__.side_effect = db_getitem

        # Reset global variables
        lm.mongo_client = None
        lm.mongo_db = None
        lm.gesture_collection = None

        # First call to init_db() should initialize globals
        lm.init_db()

        self.assertIs(lm.mongo_client, fake_client)
        self.assertIs(lm.mongo_db, fake_db)
        self.assertIs(lm.gesture_collection, fake_collection)
        fake_client.admin.command.assert_called_once_with("ping")

        # Second call should not create new connections
        lm.init_db()
        mock_mongo_client.assert_called_once()


class TestMainLoopOnce(unittest.TestCase):
    @mock.patch("src.live_mediapipe_mlp.init_db")
    @mock.patch("src.live_mediapipe_mlp.cv2")
    @mock.patch("src.live_mediapipe_mlp.mp")
    @mock.patch("src.live_mediapipe_mlp.load_model")
    def test_main_runs_single_iteration(
        self,
        mock_load_model,
        mock_mp,
        mock_cv2,
        mock_init_db,
    ):
        """Run main() for exactly one loop iteration using fully mocked camera, MediaPipe, and DB."""

        # ----- Fake model and class names -----
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(63, 3)

            def forward(self, x):
                return self.fc(x)

        fake_model = FakeModel()
        fake_classes = ["palm", "fist", "like"]
        mock_load_model.return_value = (fake_model, fake_classes)

        # Fake torch.device -> "cpu" and softmax -> high confidence
        with mock.patch("src.live_mediapipe_mlp.torch.device", return_value="cpu"), \
             mock.patch("src.live_mediapipe_mlp.torch.softmax") as mock_softmax:

            # Always return a distribution with high confidence for class 0
            mock_softmax.side_effect = lambda logits, dim: torch.tensor([[0.9, 0.05, 0.05]])

            # ----- Fake MediaPipe hands result -----
            class FakeLandmark:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z

            class FakeHandLandmarks:
                def __init__(self):
                    self.landmark = [FakeLandmark(0.1, 0.2, 0.3) for _ in range(21)]

            class FakeHandInfo:
                class Classification:
                    def __init__(self):
                        self.label = "Right"

                def __init__(self):
                    self.classification = [self.Classification()]

            class FakeResults:
                def __init__(self):
                    self.multi_hand_landmarks = [FakeHandLandmarks()]
                    self.multi_handedness = [FakeHandInfo()]

            class FakeHandsContext:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False

                def process(self, img_rgb):
                    return FakeResults()

            fake_hands_module = mock.Mock()
            fake_hands_module.Hands.return_value = FakeHandsContext()
            fake_hands_module.HAND_CONNECTIONS = object()
            mock_mp.solutions.hands = fake_hands_module

            mock_mp.solutions.drawing_utils = mock.Mock()

            # ----- Fake cv2 camera -----
            class FakeCapture:
                def __init__(self):
                    self.called = 0

                def isOpened(self):
                    return True

                def read(self):
                    # First call returns a frame; second ends loop
                    if self.called == 0:
                        self.called += 1
                        return True, mock.Mock()
                    return False, None

                def release(self):
                    return None

            mock_cv2.VideoCapture.return_value = FakeCapture()
            mock_cv2.flip.side_effect = lambda frame, _: frame
            mock_cv2.cvtColor.side_effect = lambda frame, code: frame
            mock_cv2.putText.side_effect = lambda *args, **kwargs: None
            mock_cv2.imshow.side_effect = lambda *args, **kwargs: None
            mock_cv2.destroyAllWindows.side_effect = lambda: None
            mock_cv2.waitKey.return_value = ord("q")

            # ----- Fake Mongo collection -----
            inserted = []

            class FakeCollection:
                def insert_one(self, event):
                    inserted.append(event)

            lm.gesture_collection = FakeCollection()

            # ----- Run main() -----
            lm.main()

            # Verify at least one gesture event was logged
            self.assertGreaterEqual(len(inserted), 1)
            self.assertIn(inserted[0]["gesture"], fake_classes)

