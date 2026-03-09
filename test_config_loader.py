import unittest

from quant.config.loader import get_module_configuration, load_configuration


class ConfigLoaderTests(unittest.TestCase):
    def test_load_configuration_merges_modules_and_universe(self):
        config = load_configuration()
        self.assertIn("universe", config)
        self.assertIn("tickers", config["universe"])
        self.assertGreater(len(config["universe"]["tickers"]), 0)

        modules = get_module_configuration(config)
        self.assertIsInstance(modules, dict)
        self.assertIn("technical_indicators", modules)


if __name__ == "__main__":
    unittest.main()
