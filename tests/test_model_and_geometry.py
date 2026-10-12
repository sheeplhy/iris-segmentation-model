from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from iris_system.data.transforms import letterbox_array, random_training_view, undo_letterbox
from iris_system.models import build_model
from iris_system.inference import IrisPredictor
from iris_system.training.losses import multitask_loss
from iris_system.training.trainer import train_model
from iris_system.data.records import SampleRecord


class GeometryTests(unittest.TestCase):
    def test_letterbox_inverse_restores_original_shape(self):
        image = np.zeros((37, 101), dtype=np.float32)
        image[8:30, 20:80] = 1.0
        padded, meta = letterbox_array(image, (56, 84), interpolation=cv2.INTER_LINEAR)
        restored = undo_letterbox(padded, meta, interpolation=cv2.INTER_LINEAR)
        self.assertEqual(restored.shape, image.shape)
        self.assertGreater(float(restored[10:28, 25:75].mean()), 0.9)

    def test_positive_local_view_preserves_native_detail(self):
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        iris = np.zeros((400, 800), dtype=np.float32)
        iris[180:220, 380:420] = 1.0
        targets = {
            "iris": iris,
            "eye_region": iris.copy(),
            "pupil": np.zeros_like(iris),
            "boundary": np.zeros_like(iris),
        }
        cropped, cropped_targets, view = random_training_view(
            image,
            targets,
            (448, 672),
            global_probability=0.0,
            positive_probability=1.0,
            context_min=2.0,
            context_max=2.0,
            center_jitter=0.0,
        )
        self.assertEqual(view, "positive_local")
        self.assertEqual(cropped.shape[:2], cropped_targets["iris"].shape)
        self.assertGreater(float(cropped_targets["iris"].mean()), float(iris.mean()))


class ModelTests(unittest.TestCase):
    def test_predictor_returns_iris_and_sclera_at_original_size(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, value):
                batch, _, height, width = value.shape
                logits = torch.zeros(batch, 3, height, width, device=value.device)
                logits[:, 0] = 2.0
                logits[:, 1] = -2.0
                return {"logits": logits, "boundary_logits": logits[:, :1]}

        predictor = IrisPredictor(
            ConstantModel(),
            {"model": {"input_height": 56, "input_width": 84}, "inference": {"threshold": 0.5}},
            device=torch.device("cpu"),
        )
        result = predictor.predict(np.zeros((37, 101, 3), dtype=np.uint8), preset="fast")
        self.assertEqual(result.mask.shape, (37, 101))
        self.assertEqual(result.eye_region_mask.shape, (37, 101))
        self.assertEqual(result.sclera_mask.shape, (37, 101))
        self.assertTrue(np.all(result.mask == 255))
        # Inference projects eye_region to contain the primary iris prediction.
        self.assertTrue(np.all(result.eye_region_mask == 255))
        self.assertTrue(np.all(result.sclera_mask == 0))

    def test_dinov2_model_shapes(self):
        model = build_model({"model": {"pretrained": False, "decoder_channels": 32}}, pretrained=False)
        output = model(torch.randn(1, 3, 56, 84))
        self.assertEqual(tuple(output["logits"].shape), (1, 3, 56, 84))
        self.assertEqual(tuple(output["boundary_logits"].shape), (1, 1, 56, 84))

    def test_missing_auxiliary_labels_do_not_backpropagate(self):
        body = torch.zeros(2, 3, 8, 8, requires_grad=True)
        boundary = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros(2, 4, 8, 8)
        valid = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
        loss, _ = multitask_loss(
            {"logits": body, "boundary_logits": boundary},
            target,
            valid,
            torch.ones(2),
            {"iris": 1.0, "eye_region": 1.0, "pupil": 1.0,
             "occlusion_boundary": 1.0, "hierarchy": 1.0},
        )
        loss.backward()
        self.assertGreater(float(body.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(body.grad[:, 1:].abs().sum()), 0.0)
        self.assertEqual(float(boundary.grad.abs().sum()), 0.0)


class TrainingSelectionTests(unittest.TestCase):
    def test_without_validation_exports_last_epoch_and_does_not_create_best(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Conv2d(3, 4, 1)
                self.decoder = torch.nn.Conv2d(4, 4, 1)
                self.body_head = torch.nn.Conv2d(4, 3, 1)
                self.boundary_head = torch.nn.Conv2d(4, 1, 1)

            def set_encoder_trainable(self, trainable):
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = trainable

            def forward(self, value):
                features = self.decoder(self.encoder(value))
                return {
                    "logits": self.body_head(features),
                    "boundary_logits": self.boundary_head(features),
                }

        temporary = Path(tempfile.mkdtemp())
        image_path = temporary / "eye.png"
        mask_path = temporary / "mask.png"
        cv2.imwrite(str(image_path), np.full((14, 14, 3), 100, dtype=np.uint8))
        mask = np.zeros((14, 14), dtype=np.uint8)
        mask[4:10, 4:10] = 255
        cv2.imwrite(str(mask_path), mask)
        record = SampleRecord(
            sample_id="train:eye",
            image_path=str(image_path),
            mask_path=str(mask_path),
            dataset="train",
            domain="train",
            source_format="binary_mask",
            width=14,
            height=14,
            available_labels=["iris"],
        )
        config = {
            "model": {"input_height": 14, "input_width": 14, "pretrained": False},
            "training": {
                "seed": 66,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "num_workers": 0,
                "learning_rate": 1e-3,
                "encoder_lr_factor": 1.0,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "warmup_epochs": 0,
                "freeze_encoder_epochs": 0,
                "early_stopping_patience": 1,
                "amp": False,
                "quality_weights": {"unknown": 1.0},
            },
            "loss": {
                "iris": 1.0,
                "eye_region": 1.0,
                "pupil": 0.0,
                "occlusion_boundary": 0.0,
                "hierarchy": 0.0,
            },
        }
        run_dir = temporary / "run"
        with patch("iris_system.training.trainer.build_model", return_value=TinyModel()):
            selected = train_model(
                train_records=[record],
                val_records=[],
                config=config,
                run_dir=run_dir,
                mode="rebuild",
                device=torch.device("cpu"),
            )
        self.assertEqual(selected, run_dir / "last.pth")
        self.assertTrue(selected.is_file())
        self.assertFalse((run_dir / "best.pth").exists())
        payload = torch.load(selected, map_location="cpu", weights_only=False)
        self.assertEqual(payload["meta"]["epoch"], 2)
        self.assertNotIn("val_dice", payload["meta"]["metrics"])


if __name__ == "__main__":
    unittest.main()
