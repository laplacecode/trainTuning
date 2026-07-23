from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.dataset_audit import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def test_audit_counts_objects_negatives_and_split_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split in ("train", "val"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)

            (root / "train" / "images" / "positive.jpg").write_bytes(b"image")
            (root / "train" / "labels" / "positive.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            (root / "train" / "images" / "ok.jpg").write_bytes(b"image")
            (root / "train" / "labels" / "ok.txt").write_text("", encoding="utf-8")
            (root / "val" / "images" / "positive.jpg").write_bytes(b"image")
            (root / "val" / "labels" / "positive.txt").write_text(
                "1 0.4 0.4 0.1 0.1\n",
                encoding="utf-8",
            )
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "path: .",
                        "train: train/images",
                        "val: val/images",
                        "names: [left, right]",
                    ]
                ),
                encoding="utf-8",
            )

            result = audit_dataset(
                str(yaml_path),
                task="detect",
                scene_type="fixed_industrial",
            )

        self.assertEqual("warning", result["status"])
        self.assertEqual(2, result["splits"]["train"]["images"])
        self.assertEqual(1, result["splits"]["train"]["negative_images"])
        self.assertEqual(2, result["class_count"])
        self.assertTrue(any("数据泄漏" in warning for warning in result["warnings"]))

    def test_fixed_industrial_scene_warns_when_negatives_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split in ("train", "val"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
                (root / split / "images" / f"{split}.jpg").write_bytes(b"image")
                (root / split / "labels" / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: .\ntrain: train/images\nval: val/images\nnames: [defect]\n",
                encoding="utf-8",
            )

            result = audit_dataset(
                str(yaml_path),
                task="detect",
                scene_type="fixed_industrial",
            )

        self.assertTrue(any("负样本" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
