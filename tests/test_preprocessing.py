import unittest


class PreprocessingHelpersTest(unittest.TestCase):
    def test_default_config_exists(self) -> None:
        from pathlib import Path

        config_path = Path("configs/default.yaml")
        self.assertTrue(config_path.exists())


if __name__ == "__main__":
    unittest.main()
