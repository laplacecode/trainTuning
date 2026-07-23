from __future__ import annotations

import json
import math
import os
import shlex
from dataclasses import dataclass
from typing import Any


TASK_MEMORY_FACTOR = {
    "detect": 1.0,
    "segment": 1.38,
    "pose": 1.25,
    "classify": 0.62,
    "obb": 1.18,
}

VARIANT_MEMORY = {
    "n": 3.0,
    "s": 5.5,
    "m": 9.0,
    "l": 14.0,
    "x": 21.0,
}

VARIANTS = ["n", "s", "m", "l", "x"]


@dataclass(slots=True)
class Parameter:
    key: str
    label: str
    value: Any
    category: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.value, bool):
            display = str(self.value).lower()
        else:
            display = str(self.value)
        return {
            "key": self.key,
            "label": self.label,
            "value": display,
            "category": self.category,
            "reason": self.reason,
        }


def recommend(request: dict[str, Any]) -> dict[str, Any]:
    image_count = _positive_int(request.get("image_count"), 2000)
    class_count = _positive_int(request.get("class_count"), 10)
    gpu_memory = _positive_float(request.get("gpu_memory_gb"), 8.0)
    gpu_count = _positive_int(request.get("gpu_count"), 1)
    task = _choice(request.get("task"), TASK_MEMORY_FACTOR, "detect")
    family = str(request.get("model_family") or "yolov8").lower().replace("-", "")
    family = family if family in {"yolo26", "yolo11", "yolov8"} else "yolov8"
    requested_variant = str(request.get("model_variant") or "n").lower()
    object_size = _choice(request.get("object_size"), {"small", "mixed", "large"}, "mixed")
    goal = _choice(request.get("training_goal"), {"balanced", "accuracy", "speed"}, "balanced")
    class_balance = _choice(request.get("class_balance"), {"balanced", "mild", "severe"}, "balanced")
    label_quality = _choice(request.get("label_quality"), {"high", "medium", "low"}, "high")
    dataset_path = str(request.get("dataset_path") or "data/dataset.yaml")

    variant = requested_variant if requested_variant in VARIANTS else _choose_variant(gpu_memory, task, goal)
    model_name = f"{family}{variant}.pt"
    imgsz = _choose_image_size(gpu_memory, task, object_size, goal)
    epochs = _choose_epochs(image_count, goal, label_quality)
    patience = max(20, min(60, round(epochs * 0.22)))
    workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    if os.name == "nt":
        workers = min(workers, 4)

    mosaic = 1.0 if object_size == "small" else 0.7 if object_size == "mixed" else 0.4
    if image_count < 1000:
        mosaic = min(1.0, mosaic + 0.15)
    if label_quality == "low":
        mosaic = max(0.3, mosaic - 0.2)

    mixup = 0.0
    if image_count < 2500:
        mixup = 0.08
    if class_balance == "severe":
        mixup = max(mixup, 0.12)
    if label_quality == "low":
        mixup = 0.0

    optimizer = "auto"
    close_mosaic = 15 if object_size == "small" else 10
    batch: int | float = 0.70 if gpu_memory >= 4 else 0.55
    cache: bool | str = "disk" if image_count <= 8000 else False

    parameters = [
        Parameter("imgsz", "输入尺寸", imgsz, "基础",
                  _imgsz_reason(object_size, gpu_memory)),
        Parameter("batch", "Batch", batch, "基础",
                  "按可用显存比例自动估算，减少手工试错和 OOM 风险。"),
        Parameter("epochs", "训练轮数", epochs, "训练",
                  _epochs_reason(image_count)),
        Parameter("patience", "早停耐心值", patience, "训练",
                  "在充分收敛与避免无效训练之间取平衡。"),
        Parameter("optimizer", "优化器", optimizer, "优化",
                  "交给当前 Ultralytics 根据模型和迭代规模选择优化器。"),
        Parameter("cos_lr", "余弦学习率", goal != "speed", "优化",
                  "精度或均衡目标使用平滑退火；速度优先关闭以简化短训。"),
        Parameter("mosaic", "Mosaic", round(mosaic, 2), "增强",
                  "结合目标尺寸、数据量和标注质量设置增强强度。"),
        Parameter("mixup", "MixUp", round(mixup, 2), "增强",
                  "小数据或类别失衡时提供轻量正则化。"),
        Parameter("close_mosaic", "关闭 Mosaic", close_mosaic, "增强",
                  "训练末期关闭强增强以稳定定位精度。"),
        Parameter("amp", "混合精度", True, "性能",
                  "NVIDIA GPU 上通常能降低显存占用并提升吞吐。"),
        Parameter("cache", "数据缓存", cache, "性能",
                  "中小数据集使用磁盘缓存，避免占满内存。"),
        Parameter("workers", "加载进程", workers, "性能",
                  "根据当前 CPU 核心数估算，并限制 Windows 多进程开销。"),
        Parameter("pretrained", "预训练权重", True, "模型",
                  "自定义数据训练优先采用迁移学习。"),
        Parameter("device", "训练设备", _device_value(gpu_count), "硬件",
                  "使用指定数量的本机 GPU。"),
    ]

    if task == "segment":
        parameters.append(Parameter(
            "copy_paste", "Copy-Paste", 0.2 if image_count < 5000 else 0.1, "增强",
            "实例分割任务可通过实例级粘贴增强稀有目标。"))

    config = {parameter.key: parameter.value for parameter in parameters}
    config.update({
        "model": model_name,
        "data": dataset_path,
        "task": task,
    })

    warnings = _warnings(
        image_count=image_count,
        class_count=class_count,
        gpu_memory=gpu_memory,
        task=task,
        object_size=object_size,
        class_balance=class_balance,
        label_quality=label_quality,
        model_name=model_name,
        imgsz=imgsz,
    )
    confidence = _confidence(image_count, label_quality, class_balance, len(warnings))
    summary = _summary(model_name, imgsz, epochs, goal, object_size, gpu_memory)
    command = _build_command(task, config)

    return {
        "model": model_name,
        "summary": summary,
        "confidence": confidence,
        "command": command,
        "parameters": [parameter.to_dict() for parameter in parameters],
        "warnings": warnings,
        "config": config,
    }


def _choose_variant(gpu_memory: float, task: str, goal: str) -> str:
    usable = gpu_memory * 0.88 / TASK_MEMORY_FACTOR[task]
    fitting = [variant for variant in VARIANTS if VARIANT_MEMORY[variant] <= usable]
    index = VARIANTS.index(fitting[-1]) if fitting else 0
    if goal == "accuracy" and index < len(VARIANTS) - 1:
        candidate = VARIANTS[index + 1]
        if VARIANT_MEMORY[candidate] <= usable * 1.15:
            index += 1
    elif goal == "speed" and index > 0:
        index -= 1
    return VARIANTS[index]


def _choose_image_size(gpu_memory: float, task: str, object_size: str, goal: str) -> int:
    size = 640
    if object_size == "small":
        size = 960
    elif object_size == "large" and goal == "speed":
        size = 512
    elif goal == "accuracy":
        size = 800

    effective_memory = gpu_memory / TASK_MEMORY_FACTOR[task]
    if effective_memory < 5:
        size = min(size, 512)
    elif effective_memory < 8:
        size = min(size, 640)
    elif effective_memory < 12:
        size = min(size, 800)
    return int(math.ceil(size / 32) * 32)


def _choose_epochs(image_count: int, goal: str, label_quality: str) -> int:
    if image_count < 500:
        epochs = 260
    elif image_count < 2000:
        epochs = 200
    elif image_count < 10000:
        epochs = 140
    elif image_count < 50000:
        epochs = 100
    else:
        epochs = 80
    if goal == "accuracy":
        epochs = round(epochs * 1.2)
    elif goal == "speed":
        epochs = round(epochs * 0.7)
    if label_quality == "low":
        epochs = round(epochs * 0.85)
    return max(50, epochs)


def _warnings(**values: Any) -> list[str]:
    warnings: list[str] = []
    if values["image_count"] < 500:
        warnings.append("数据量较小，建议使用交叉验证或重复划分验证稳定性。")
    if values["image_count"] / values["class_count"] < 80:
        warnings.append("平均每类样本偏少，优先补充长尾类别而不是继续增大模型。")
    if values["class_balance"] == "severe":
        warnings.append("类别严重失衡，建议先检查每类实例数并针对稀有类别采样。")
    if values["label_quality"] == "low":
        warnings.append("标注质量较低；清洗漏标和错标通常比扩大参数搜索更有效。")
    if values["object_size"] == "small" and values["gpu_memory"] < 8:
        warnings.append("小目标需要较高输入分辨率，但当前显存会限制 batch 或模型规模。")
    if values["task"] == "segment" and values["gpu_memory"] < 6:
        warnings.append("实例分割显存需求较高，发生 OOM 时优先降低 batch 利用率。")
    if not warnings:
        warnings.append("当前条件未发现明显高风险项，建议先进行 10–20 轮冒烟训练。")
    return warnings


def _confidence(image_count: int, label_quality: str, class_balance: str, warning_count: int) -> int:
    score = 84
    if image_count < 500:
        score -= 12
    elif image_count < 2000:
        score -= 5
    if label_quality == "medium":
        score -= 5
    elif label_quality == "low":
        score -= 14
    if class_balance == "mild":
        score -= 3
    elif class_balance == "severe":
        score -= 8
    score -= max(0, warning_count - 2)
    return max(45, min(92, score))


def _summary(model: str, imgsz: int, epochs: int, goal: str, object_size: str, memory: float) -> str:
    goal_text = {"balanced": "精度与速度均衡", "accuracy": "精度优先", "speed": "训练效率优先"}[goal]
    size_text = {"small": "小目标", "mixed": "混合尺寸目标", "large": "大目标"}[object_size]
    return (
        f"基于 {memory:g} GB 显存和{size_text}场景，建议以 {model}、"
        f"{imgsz}px 输入训练 {epochs} 轮作为第一组基线；策略偏向{goal_text}。"
    )


def _imgsz_reason(object_size: str, memory: float) -> str:
    if object_size == "small":
        return f"小目标需要更多像素细节，并根据 {memory:g} GB 显存限制上限。"
    return f"结合目标尺寸与 {memory:g} GB 显存选择稳定的 32 倍数分辨率。"


def _epochs_reason(image_count: int) -> str:
    if image_count < 2000:
        return "数据较少，增加轮数并依赖早停观察充分收敛。"
    if image_count > 10000:
        return "数据较多，每轮覆盖充分，无需堆叠过多 epoch。"
    return "中等数据规模采用稳健的训练轮数，并由早停控制过拟合。"


def _device_value(gpu_count: int) -> str:
    if gpu_count <= 1:
        return "0"
    return ",".join(str(index) for index in range(gpu_count))


def _build_command(task: str, config: dict[str, Any]) -> str:
    order = [
        "model", "data", "epochs", "imgsz", "batch", "optimizer", "patience",
        "cos_lr", "mosaic", "mixup", "close_mosaic", "amp", "cache", "workers", "device",
    ]
    parts = ["yolo", task, "train"]
    for key in order:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool):
            encoded = str(value).lower()
        else:
            encoded = str(value)
        if key == "data":
            encoded = f'"{encoded}"'
        parts.append(f"{key}={encoded}")
    return " ".join(parts)


def _choice(value: Any, choices: Any, default: str) -> str:
    normalized = str(value or default).lower()
    return normalized if normalized in choices else default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    print(json.dumps(recommend({}), ensure_ascii=False, indent=2))
