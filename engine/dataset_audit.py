from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def audit_dataset(
    dataset_path: str,
    *,
    task: str = "detect",
    scene_type: str = "general",
) -> dict[str, Any]:
    """Inspect a local YOLO dataset without modifying it."""

    if task == "classify":
        return {
            "status": "skipped",
            "summary": "分类数据集采用目录类别结构，当前版本暂不执行 YOLO 标签审计。",
            "warnings": ["分类任务仍需单独检查 train/val/test 的类别目录和样本泄漏。"],
        }

    yaml_path = Path(dataset_path).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (Path.cwd() / yaml_path).resolve()
    if not yaml_path.is_file():
        if "/" not in dataset_path and "\\" not in dataset_path:
            return {
                "status": "skipped",
                "summary": f"{dataset_path} 将由 Ultralytics 解析，未执行本地数据审计。",
                "warnings": [],
            }
        return {
            "status": "missing",
            "summary": f"找不到数据集配置：{yaml_path}",
            "warnings": ["训练前必须选择存在的 YOLO 数据集 YAML。"],
        }

    try:
        import yaml
    except ImportError:
        return {
            "status": "unavailable",
            "summary": "未安装 PyYAML，已跳过本地数据审计。",
            "warnings": ["安装 engine/requirements.txt 后可启用训练前数据审计。"],
        }

    try:
        with yaml_path.open("r", encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exception:
        return {
            "status": "invalid",
            "summary": f"无法读取数据集 YAML：{exception}",
            "warnings": ["数据集 YAML 无法解析，不能安全启动训练。"],
        }

    class_count = _class_count(config.get("names"))
    dataset_root = _dataset_root(yaml_path, config.get("path"))
    splits: dict[str, dict[str, Any]] = {}
    all_warnings: list[str] = []
    split_stems: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        configured = config.get(split)
        if not configured:
            continue
        split_result = _audit_split(
            split,
            configured,
            dataset_root=dataset_root,
            yaml_path=yaml_path,
            class_count=class_count,
            task=task,
        )
        splits[split] = split_result
        split_stems[split] = set(split_result.pop("_stems", []))
        all_warnings.extend(split_result["warnings"])

    train = splits.get("train", {})
    validation = splits.get("val", {})
    train_images = int(train.get("images", 0))
    val_images = int(validation.get("images", 0))
    negatives = int(train.get("negative_images", 0)) + int(
        validation.get("negative_images", 0)
    )
    objects = int(train.get("objects", 0)) + int(validation.get("objects", 0))

    overlap = sorted(split_stems.get("train", set()) & split_stems.get("val", set()))
    if overlap:
        all_warnings.append(
            f"train/val 有 {len(overlap)} 个同名样本；请按工件或采集批次检查数据泄漏。"
        )
    if not validation:
        all_warnings.append("未配置 val 划分，无法依据独立验证指标调优。")
    elif val_images < 20:
        all_warnings.append(
            f"验证集只有 {val_images} 张图片，单轮指标波动可能很大。"
        )
    if task in {"detect", "segment", "obb"} and scene_type == "fixed_industrial" and negatives == 0:
        all_warnings.append(
            "固定工业场景未发现显式负样本（空标签图片）；误报率无法得到可靠训练与验证。"
        )
    if objects == 0 and train_images:
        all_warnings.append("未读到任何有效标注目标，请检查 labels 路径和标注格式。")

    duplicate_warnings = list(dict.fromkeys(all_warnings))
    status = "warning" if duplicate_warnings else "ok"
    summary = (
        f"数据审计：train {train_images} 张，val {val_images} 张，"
        f"目标 {objects} 个，显式负样本 {negatives} 张。"
    )
    return {
        "status": status,
        "summary": summary,
        "yaml_path": str(yaml_path),
        "class_count": class_count,
        "splits": splits,
        "warnings": duplicate_warnings,
    }


def _audit_split(
    split: str,
    configured: Any,
    *,
    dataset_root: Path,
    yaml_path: Path,
    class_count: int | None,
    task: str,
) -> dict[str, Any]:
    entries = configured if isinstance(configured, list) else [configured]
    images: list[Path] = []
    warnings: list[str] = []
    for entry in entries:
        resolved = _resolve_entry(entry, dataset_root, yaml_path)
        if resolved is None:
            warnings.append(f"{split} 路径无效：{entry}")
            continue
        if resolved.is_dir():
            images.extend(
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif resolved.is_file() and resolved.suffix.lower() == ".txt":
            images.extend(_images_from_manifest(resolved, dataset_root))
        else:
            warnings.append(f"{split} 路径不存在或格式不受支持：{resolved}")

    images = sorted(set(path.resolve() for path in images))
    counts: Counter[int] = Counter()
    missing_labels = 0
    empty_labels = 0
    invalid_labels = 0
    objects = 0
    for image_path in images:
        label_path = _label_path(image_path)
        if not label_path.is_file():
            missing_labels += 1
            continue
        try:
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError):
            invalid_labels += 1
            continue
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            fields = line.split()
            expected_minimum = 5
            if len(fields) < expected_minimum:
                invalid_labels += 1
                continue
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                invalid_labels += 1
                continue
            if class_id < 0 or (class_count is not None and class_id >= class_count):
                invalid_labels += 1
                continue
            if not _coordinates_are_valid(task, coordinates):
                invalid_labels += 1
                continue
            counts[class_id] += 1
            objects += 1

    if missing_labels:
        warnings.append(
            f"{split} 有 {missing_labels} 张图片缺少同名标签；若是负样本，建议创建显式空标签。"
        )
    if invalid_labels:
        warnings.append(f"{split} 发现 {invalid_labels} 行不可用标注。")
    nonzero_counts = [count for count in counts.values() if count > 0]
    if len(nonzero_counts) >= 2 and max(nonzero_counts) / min(nonzero_counts) >= 10:
        warnings.append(
            f"{split} 类别实例最大/最小比达到 {max(nonzero_counts) / min(nonzero_counts):.1f}。"
        )

    return {
        "images": len(images),
        "objects": objects,
        "negative_images": empty_labels,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "class_counts": {str(key): value for key, value in sorted(counts.items())},
        "warnings": warnings,
        "_stems": [path.stem for path in images],
    }


def _dataset_root(yaml_path: Path, configured_root: Any) -> Path:
    if not configured_root:
        return yaml_path.parent
    root = Path(str(configured_root)).expanduser()
    if root.is_absolute():
        return root
    beside_yaml = (yaml_path.parent / root).resolve()
    if beside_yaml.exists():
        return beside_yaml
    return (Path.cwd() / root).resolve()


def _resolve_entry(entry: Any, dataset_root: Path, yaml_path: Path) -> Path | None:
    if not isinstance(entry, (str, Path)):
        return None
    path = Path(str(entry)).expanduser()
    if path.is_absolute():
        return path
    under_root = (dataset_root / path).resolve()
    if under_root.exists():
        return under_root
    return (yaml_path.parent / path).resolve()


def _images_from_manifest(manifest: Path, dataset_root: Path) -> list[Path]:
    result: list[Path] = []
    try:
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return result
    for line in lines:
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = dataset_root / path
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            result.append(path.resolve())
    return result


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _class_count(names: Any) -> int | None:
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    return None


def _coordinates_are_valid(task: str, coordinates: list[float]) -> bool:
    if task == "pose":
        if len(coordinates) < 4 or not all(0.0 <= value <= 1.0 for value in coordinates[:4]):
            return False
        keypoints = coordinates[4:]
        for index in range(0, len(keypoints), 3):
            group = keypoints[index:index + 3]
            if len(group) < 2 or not all(0.0 <= value <= 1.0 for value in group[:2]):
                return False
            if len(group) == 3 and not 0.0 <= group[2] <= 2.0:
                return False
        return True
    return all(0.0 <= value <= 1.0 for value in coordinates)
