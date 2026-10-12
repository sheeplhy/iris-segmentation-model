from __future__ import annotations

import unittest

from iris_system.gui.app import _cli_option


class GuiArgumentTests(unittest.TestCase):
    def test_training_keys_use_argparse_kebab_case(self):
        self.assertEqual(_cli_option("gradient_accumulation"), "--gradient-accumulation")
        self.assertEqual(_cli_option("encoder_lr_factor"), "--encoder-lr-factor")

    def test_loss_keys_use_argparse_kebab_case(self):
        self.assertEqual(_cli_option("iris", loss=True), "--loss-iris")
        self.assertEqual(_cli_option("eye_region", loss=True), "--loss-eye-region")
        self.assertEqual(_cli_option("boundary", loss=True), "--loss-boundary")


if __name__ == "__main__":
    unittest.main()
