import unittest


class ModelSmokeTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from src.wave_anomaly.model import DualBranchConvLSTMUNet

        model = DualBranchConvLSTMUNet()
        x_wind = torch.randn(2, 8, 3, 64, 64)
        x_wave = torch.randn(2, 8, 4, 64, 64)
        y = model(x_wind, x_wave)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
