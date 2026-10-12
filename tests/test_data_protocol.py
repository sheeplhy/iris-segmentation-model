from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from iris_system.data.manifest import compile_dataset
from iris_system.data.records import DatasetSchema, SampleRecord
from iris_system.data.targets import build_targets
from iris_system.data.validation import validate_records
from iris_system.evaluation.evaluator import contamination_report, semantic_mask


def _image(path: Path, size=(80, 60)) -> None:
    Image.new("RGB", size, (90, 70, 50)).save(path)


class CvatProtocolTests(unittest.TestCase):
    def _dataset(self, shapes: str) -> Path:
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "JPEGImages"
        images.mkdir()
        _image(images / "annotated.png")
        _image(images / "unannotated.png")
        xml = f'''<annotations>
          <image id="0" name="annotated.png" width="80" height="60">
            <polygon label="iris" points="20,15;60,15;60,45;20,45">
              <attribute name="if clear">clear</attribute>
            </polygon>
            {shapes}
          </image>
          <image id="1" name="unannotated.png" width="80" height="60" />
        </annotations>'''
        (temporary / "annotations.xml").write_text(xml, encoding="utf-8")
        return temporary

    def test_unannotated_images_are_ignored(self):
        records = compile_dataset(self._dataset(""), compute_hashes=False)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].quality, "clear")
        self.assertEqual(records[0].boundary_status, "unavailable")

    def test_complete_point_means_valid_empty_boundary(self):
        shape = '''<points label="occlusion_boundary" points="3,4">
          <attribute name="complete">true</attribute>
        </points>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.boundary_status, "complete")
        self.assertTrue(valid["boundary"])
        self.assertEqual(float(targets["boundary"].sum()), 0.0)

    def test_complete_polyline_geometry_is_ignored(self):
        shape = '''<polyline label="occlusion_boundary" points="3,4;20,15;40,8">
          <attribute name="status">complete</attribute>
        </polyline>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.boundary_status, "complete")
        self.assertEqual(record.boundary_polylines, [])
        self.assertTrue(valid["boundary"])
        self.assertEqual(float(targets["boundary"].sum()), 0.0)

    def test_boundary_polyline_generates_soft_target(self):
        shape = '''<polyline label="occlusion_boundary" points="20,15;40,12;60,15">
          <attribute name="boundary">true</attribute>
          <attribute name="position">upper</attribute>
        </polyline>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.boundary_status, "boundary")
        self.assertTrue(valid["boundary"])
        self.assertGreater(float(targets["boundary"].sum()), 0.0)
        self.assertEqual(record.boundary_polylines[0]["position"], "upper")

    def test_boundary_value_may_use_an_existing_custom_attribute_name(self):
        shape = '''<polyline label="occlusion_boundary" points="20,45;40,48;60,45">
          <attribute name="annotation_state">boundary</attribute>
          <attribute name="position">lower</attribute>
        </polyline>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        self.assertEqual(record.boundary_status, "boundary")
        self.assertEqual(record.boundary_polylines[0]["position"], "lower")

    def test_if_clear_is_read_only_from_iris(self):
        shape = '''<polygon label="sclera" points="8,8;72,8;72,52;8,52">
          <attribute name="if clear">blur</attribute>
        </polygon>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        self.assertEqual(record.quality, "clear")

    def test_unavailable_iris_polygon_is_only_an_attribute_carrier(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "JPEGImages"
        images.mkdir()
        _image(images / "unavailable.png")
        (temporary / "annotations.xml").write_text(
            '''<annotations><image id="0" name="unavailable.png" width="80" height="60">
              <polygon label="iris" points="5,5;75,5;75,55;5,55">
                <attribute name="if clear">unavailable</attribute>
              </polygon>
              <polygon label="pupil" points="30,20;50,20;50,40;30,40" />
              <polygon label="sclera" points="2,2;78,2;78,58;2,58" />
              <polyline label="occlusion_boundary" points="10,10;70,10">
                <attribute name="status">boundary</attribute>
              </polyline>
            </image></annotations>''',
            encoding="utf-8",
        )

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertTrue(record.unavailable)
        self.assertEqual(record.quality, "unavailable")
        self.assertEqual(record.iris_polygons, [])
        self.assertEqual(record.pupil_polygons, [])
        self.assertEqual(record.sclera_polygons, [])
        self.assertEqual(record.boundary_polylines, [])
        self.assertEqual(record.boundary_status, "unavailable")
        self.assertEqual(record.available_labels, ["iris"])
        self.assertTrue(valid["iris"])
        self.assertFalse(valid["eye_region"])
        self.assertFalse(valid["pupil"])
        self.assertFalse(valid["boundary"])
        self.assertEqual(float(targets["iris"].sum()), 0.0)
        self.assertEqual(float(targets["eye_region"].sum()), 0.0)
        self.assertEqual(float(targets["pupil"].sum()), 0.0)
        self.assertEqual(float(targets["boundary"].sum()), 0.0)

    def test_boundary_polyline_does_not_require_position(self):
        shape = '''<polyline label="occlusion_boundary" points="20,15;40,12;60,15">
          <attribute name="status">boundary</attribute>
        </polyline>'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.boundary_status, "boundary")
        self.assertNotIn("position", record.boundary_polylines[0])
        self.assertTrue(valid["boundary"])
        self.assertGreater(float(targets["boundary"].sum()), 0.0)

    def test_cvat_sclera_polygon_is_optional_auxiliary_supervision(self):
        shape = '''<polygon label="sclera" points="8,8;72,8;72,52;8,52" />'''
        record = compile_dataset(self._dataset(shape), compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertTrue(valid["iris"])
        self.assertTrue(valid["eye_region"])
        self.assertFalse(valid["pupil"])
        self.assertGreater(float(targets["eye_region"].sum()), float(targets["iris"].sum()))


class ExternalMaskTests(unittest.TestCase):
    def test_unknown_masks_encoding_reports_dataset_folder_and_file(self):
        temporary = Path(tempfile.mkdtemp(prefix="unknown_encoding_"))
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.png", size=(32, 24))
        cv2.imwrite(str(masks / "eye.png"), np.full((24, 32, 3), (56, 34, 12), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, r"unknown_encoding_.*folder 'Masks'.*eye.png"):
            compile_dataset(temporary, compute_hashes=False)

    def test_cvat_segmentation_class_is_checked_even_though_xml_is_authoritative(self):
        temporary = Path(tempfile.mkdtemp(prefix="cvat_bad_export_"))
        images = temporary / "JPEGImages"
        masks = temporary / "SegmentationClass"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.png", size=(32, 24))
        (temporary / "annotations.xml").write_text(
            '''<annotations><image id="0" name="eye.png" width="32" height="24">
            <polygon label="iris" points="10,6;22,6;22,18;10,18" />
            </image></annotations>''', encoding="utf-8"
        )
        cv2.imwrite(str(masks / "eye.png"), np.full((24, 32, 3), 127, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, r"folder 'SegmentationClass'.*eye.png"):
            compile_dataset(temporary, compute_hashes=False)

    def test_fill_rejects_non_black_white_values(self):
        temporary = Path(tempfile.mkdtemp(prefix="bad_fill_"))
        images = temporary / "Images"
        fills = temporary / "Fill"
        images.mkdir()
        fills.mkdir()
        _image(images / "eye.png", size=(32, 24))
        cv2.imwrite(str(fills / "eye.png"), np.full((24, 32), 128, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, r"folder 'Fill'.*eye.png.*expected black/white"):
            compile_dataset(temporary, compute_hashes=False)

    def test_fill_only_sample_trains_sclera_without_fake_empty_iris(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        fills = temporary / "Fill"
        images.mkdir()
        fills.mkdir()
        _image(images / "eye.jpg", size=(32, 24))
        fill = np.zeros((24, 32), dtype=np.uint8)
        fill[6:18, 4:28] = 255
        cv2.imwrite(str(fills / "eye.png"), fill)

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertIsNone(record.mask_path)
        self.assertIsNotNone(record.fill_path)
        self.assertFalse(valid["iris"])
        self.assertTrue(valid["eye_region"])
        self.assertEqual(int(targets["eye_region"].sum()), 288)
        self.assertFalse(any(issue.level == "error" for issue in validate_records([record])))

    def test_fill_and_binary_mask_are_independent_supervision(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        fills = temporary / "Fill"
        images.mkdir()
        masks.mkdir()
        fills.mkdir()
        _image(images / "eye.jpg", size=(32, 24))
        iris = np.zeros((24, 32), dtype=np.uint8)
        iris[8:16, 12:20] = 255
        sclera = np.zeros((24, 32), dtype=np.uint8)
        sclera[6:18, 4:28] = 255
        cv2.imwrite(str(masks / "eye.png"), iris)
        cv2.imwrite(str(fills / "eye.png"), sclera)

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertTrue(valid["iris"] and valid["eye_region"])
        self.assertEqual(int(targets["iris"].sum()), 64)
        self.assertEqual(int(targets["eye_region"].sum()), 288)

    def test_existing_empty_mask_is_a_valid_negative_sample(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "closed.jpg", size=(32, 24))
        cv2.imwrite(str(masks / "closed.png"), np.zeros((24, 32), dtype=np.uint8))

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertTrue(valid["iris"])
        self.assertEqual(float(targets["iris"].sum()), 0.0)
        self.assertFalse(any(issue.level == "error" for issue in validate_records([record])))

    def test_rgb_structure_union_and_auxiliary_targets(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.jpg", size=(32, 24))
        mask = np.zeros((24, 32, 3), dtype=np.uint8)
        mask[5:19, 5:27] = (255, 0, 0)
        mask[7:17, 8:24] = (0, 255, 0)
        mask[10:14, 13:19] = (0, 0, 255)
        cv2.imwrite(str(masks / "eye.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        schema = DatasetSchema(
            name="rgb",
            format="color_mask",
            colors={"background": (0, 0, 0), "sclera": (255, 0, 0),
                    "iris": (0, 255, 0), "pupil": (0, 0, 255)},
            primary_union=["iris", "pupil"],
        )
        record = compile_dataset(temporary, schema=schema, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertTrue(valid["iris"] and valid["eye_region"] and valid["pupil"])
        self.assertEqual(int(targets["iris"].sum()), 160)
        self.assertEqual(int(targets["pupil"].sum()), 24)
        self.assertEqual(int(targets["eye_region"].sum()), 308)
        self.assertFalse(valid["boundary"])

    def test_near_black_one_hot_rgb_is_detected_without_a_schema(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.png", size=(32, 24))
        mask = np.zeros((24, 32, 3), dtype=np.uint8)
        mask[...] = (1, 0, 0)  # background
        mask[7:17, 3:29] = (0, 1, 0)  # visible sclera
        mask[7:17, 11:21] = (0, 0, 1)  # iris disk, pupil already included
        cv2.imwrite(str(masks / "eye.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.schema["name"], "one_hot_background_sclera_iris")
        self.assertEqual(record.schema["primary_union"], ["iris"])
        self.assertTrue(valid["iris"] and valid["eye_region"])
        self.assertFalse(valid["pupil"])
        self.assertEqual(int(targets["iris"].sum()), 100)
        self.assertEqual(int(targets["eye_region"].sum()), 260)

    def test_near_black_indexed_classes_are_decoded(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.png", size=(32, 24))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[6:18, 3:29] = 1  # sclera
        mask[7:17, 10:22] = 2  # iris
        mask[10:14, 14:18] = 3  # pupil
        cv2.imwrite(str(masks / "eye.png"), mask)

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.source_format, "indexed_mask")
        self.assertEqual(record.schema["name"], "indexed_background_sclera_iris_pupil")
        self.assertTrue(valid["iris"] and valid["eye_region"] and valid["pupil"])
        self.assertEqual(int(targets["iris"].sum()), 120)
        self.assertEqual(int(targets["pupil"].sum()), 16)
        self.assertEqual(int(targets["eye_region"].sum()), 312)

    def test_periorbital_zero_to_five_indexed_classes_are_decoded(self):
        temporary = Path(tempfile.mkdtemp())
        images = temporary / "Images"
        masks = temporary / "Masks"
        images.mkdir()
        masks.mkdir()
        _image(images / "eye.png", size=(32, 24))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[1:3, 4:28] = 1       # eyebrows -> task background
        mask[6:18, 3:29] = 2     # sclera
        mask[8:16, 11:21] = 3    # iris with pupil already merged
        mask[10:14, 3:5] = 4     # caruncle -> task background
        mask[20:22, 5:27] = 5    # eyelids -> task background
        cv2.imwrite(str(masks / "eye.png"), mask)

        record = compile_dataset(temporary, compute_hashes=False)[0]
        targets, valid = build_targets(record)
        self.assertEqual(record.source_format, "indexed_mask")
        self.assertEqual(record.schema["name"], "indexed_periorbital_0_5")
        self.assertEqual(record.schema["primary_union"], ["iris"])
        self.assertTrue(valid["iris"] and valid["eye_region"])
        self.assertFalse(valid["pupil"])
        self.assertEqual(int(targets["iris"].sum()), 80)
        self.assertEqual(int(targets["eye_region"].sum()), 304)
        self.assertEqual(float(targets["iris"][1:3].sum()), 0.0)
        self.assertEqual(float(targets["eye_region"][20:22].sum()), 0.0)


class ContaminationTests(unittest.TestCase):
    def test_reports_near_duplicate_perceptual_hash(self):
        record = SampleRecord(
            sample_id="benchmark:new_name",
            image_path="unused.png",
            dataset="benchmark",
            domain="benchmark",
            source_format="binary_mask",
            width=1,
            height=1,
            perceptual_hash="0" * 63 + "1",
        )
        meta = {
            "training_samples": {
                "train:old_name": {
                    "sha256": "different",
                    "perceptual_hash": "0" * 64,
                }
            }
        }
        report = contamination_report([record], meta)
        self.assertEqual(len(report), 1)
        self.assertIn("near-duplicate", report[0]["reason"])


class RenderingTests(unittest.TestCase):
    def test_combined_mask_uses_red_sclera_and_green_iris(self):
        iris = np.zeros((3, 4), dtype=np.uint8)
        sclera = np.zeros_like(iris)
        sclera[1, 1:4] = 255
        iris[1, 2] = 255
        output = semantic_mask(iris_mask=iris, sclera_mask=sclera)
        self.assertEqual(tuple(output[1, 1]), (255, 0, 0))
        self.assertEqual(tuple(output[1, 2]), (0, 255, 0))


if __name__ == "__main__":
    unittest.main()
