from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from engine.recommender import recommend
from engine.worker import build_training_config


ROOT = Path(__file__).resolve().parents[1]


class RecommenderTests(unittest.TestCase):
    def test_low_memory_small_objects_clamps_image_size(self) -> None:
        result = recommend({
            "gpu_memory_gb": 4,
            "image_count": 900,
            "class_count": 5,
            "task": "detect",
            "object_size": "small",
            "training_goal": "accuracy",
        })

        self.assertEqual("yolov8n.pt", result["model"])
        self.assertEqual(512, result["config"]["imgsz"])
        self.assertTrue(any("显存" in warning for warning in result["warnings"]))

    def test_speed_goal_uses_no_larger_model_than_accuracy_goal(self) -> None:
        common = {
            "gpu_memory_gb": 16,
            "image_count": 8000,
            "class_count": 12,
            "task": "detect",
            "model_family": "yolo11",
            "model_variant": "auto",
        }
        speed = recommend({**common, "training_goal": "speed"})
        accuracy = recommend({**common, "training_goal": "accuracy"})
        ranks = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}

        self.assertLessEqual(
            ranks[speed["model"].removeprefix("yolo11")[0]],
            ranks[accuracy["model"].removeprefix("yolo11")[0]],
        )
        self.assertLess(speed["config"]["epochs"], accuracy["config"]["epochs"])

    def test_segment_adds_copy_paste(self) -> None:
        result = recommend({
            "gpu_memory_gb": 12,
            "image_count": 3500,
            "class_count": 8,
            "task": "segment",
        })
        self.assertIn("copy_paste", result["config"])

    def test_fixed_direction_sensitive_scene_disables_invalid_augmentations(self) -> None:
        result = recommend({
            "gpu_memory_gb": 8,
            "image_count": 500,
            "class_count": 2,
            "task": "detect",
            "scene_type": "fixed_industrial",
            "direction_sensitive": True,
        })

        self.assertEqual(0.0, result["config"]["fliplr"])
        self.assertEqual(0.0, result["config"]["degrees"])
        self.assertEqual(0.0, result["config"]["mosaic"])
        self.assertLessEqual(result["config"]["scale"], 0.15)

    def test_existing_weights_uses_conservative_finetune(self) -> None:
        result = recommend({
            "initial_weights_path": r"D:\models\existing_best.pt",
            "image_count": 61,
            "class_count": 32,
            "scene_type": "fixed_industrial",
        })

        self.assertEqual(r"D:\models\existing_best.pt", result["model"])
        self.assertEqual("AdamW", result["config"]["optimizer"])
        self.assertEqual(0.0003, result["config"]["lr0"])
        self.assertEqual(16, result["config"]["nbs"])

    def test_worker_uses_confirmed_recommendation_config(self) -> None:
        _, config = build_training_config({
            "request": {
                "dataset_path": "coco8.yaml",
                "gpu_memory_gb": 8,
                "image_count": 2000,
                "class_count": 10,
            },
            "config": {
                "imgsz": 736,
                "lr0": 0.0007,
            },
        })

        self.assertEqual(736, config["imgsz"])
        self.assertEqual(0.0007, config["lr0"])

    def test_worker_emits_json_protocol(self) -> None:
        request = {
            "id": "protocol-test",
            "action": "recommend",
            "payload": {
                "gpu_memory_gb": 8,
                "image_count": 2000,
                "class_count": 10,
            },
        }
        child_environment = os.environ.copy()
        child_environment["PYTHONIOENCODING"] = "utf-8"
        child_environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "engine" / "worker.py")],
            input=json.dumps(request, ensure_ascii=False) + "\n",
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            cwd=ROOT,
            env=child_environment,
        )
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual("status", events[0]["event"])
        self.assertEqual("recommendation", events[-1]["event"])
        self.assertEqual("protocol-test", events[-1]["id"])

    def test_worker_training_supports_interactive_adjustment_response(self) -> None:
        request = {
            "id": "adaptive-protocol-test",
            "action": "train",
            "payload": {
                "request": {
                    "dataset_path": "coco8.yaml",
                    "gpu_memory_gb": 8,
                    "image_count": 2000,
                    "class_count": 10,
                },
                "force_simulation": True,
                "adjustment_mode": "confirm",
            },
        }
        response = {
            "action": "adjustment_response",
            "accepted": False,
        }
        child_environment = os.environ.copy()
        child_environment["PYTHONIOENCODING"] = "utf-8"
        child_environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "engine" / "worker.py")],
            input="\n".join(
                [
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                ]
            ) + "\n",
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            cwd=ROOT,
            env=child_environment,
            timeout=10,
        )

        events = [json.loads(line) for line in completed.stdout.splitlines()]
        event_names = [event["event"] for event in events]
        self.assertIn("progress", event_names)
        self.assertIn("adjustment_proposed", event_names)
        self.assertIn("adjustment_skipped", event_names)
        self.assertIn("training_summary", event_names)
        self.assertEqual("completed", event_names[-1])


if __name__ == "__main__":
    unittest.main()
