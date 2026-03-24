import unittest

import numpy as np

from src.wave_anomaly.metrics import StreamingPixelMetrics, connected_components, object_metrics


class MetricsTestCase(unittest.TestCase):
    def test_streaming_metrics(self) -> None:
        metrics = StreamingPixelMetrics(np.array([0.5], dtype=np.float32))
        y_true = np.array([[[[1, 0], [1, 0]]]], dtype=np.float32)
        y_prob = np.array([[[[0.9, 0.2], [0.8, 0.1]]]], dtype=np.float32)
        metrics.update(y_true, y_prob)
        summary = metrics.summary_at(0.5)
        self.assertAlmostEqual(summary["precision"], 1.0)
        self.assertAlmostEqual(summary["recall"], 1.0)

    def test_connected_components(self) -> None:
        mask = np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        components = connected_components(mask, connectivity=4)
        self.assertEqual(len(components), 2)

    def test_object_metrics(self) -> None:
        y_true = np.zeros((4, 4), dtype=np.float32)
        y_prob = np.zeros((4, 4), dtype=np.float32)
        y_true[1:3, 1:3] = 1.0
        y_prob[1:3, 1:3] = 0.8
        metrics = object_metrics(y_true, y_prob, threshold=0.5, min_area=1)
        self.assertAlmostEqual(metrics["object_csi"], 1.0)


if __name__ == "__main__":
    unittest.main()
