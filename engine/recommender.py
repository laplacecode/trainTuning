from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
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
    scene_type = _choice(
        request.get("scene_type"),
        {"fixed_industrial", "general"},
        "general",
    )
    direction_sensitive = bool(request.get("direction_sensitive", False))
    initial_weights = str(request.get("initial_weights_path") or "").strip()

    variant = requested_variant if requested_variant in VARIANTS else _choose_variant(gpu_memory, task, goal)
    model_name = initial_weights or f"{family}{variant}.pt"
    imgsz = _choose_image_size(gpu_memory, task, object_size, goal)
    epochs = _choose_epochs(image_count, goal, label_quality)
    patience = max(20, min(60, round(epochs * 0.22)))
    workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    if os.name == "nt":
        workers = min(workers, 4)

    mosaic = 1.0 if object_size == "small" else 0.7 if object_size == "mixed" else 0.4
    if scene_type == "fixed_industrial":
        mosaic = 0.0 if direction_sensitive else 0.2
    elif image_count < 1000:
        mosaic = min(1.0, mosaic + 0.15)
    if label_quality == "low":
        mosaic = max(0.3, mosaic - 0.2)
        if scene_type == "fixed_industrial":
            mosaic = 0.0

    mixup = 0.0
    if image_count < 2500 and scene_type != "fixed_industrial":
        mixup = 0.08
    if class_balance == "severe" and scene_type != "fixed_industrial":
        mixup = max(mixup, 0.12)
    if label_quality == "low":
        mixup = 0.0

    optimizer = "AdamW" if initial_weights else "auto"
    lr0 = 0.0003 if initial_weights else 0.01
    lrf = 0.05 if initial_weights else 0.01
    nbs = 16 if initial_weights or image_count < 1000 else 64
    close_mosaic = 0 if mosaic == 0 else 15 if object_size == "small" else 10
    batch: int | float = 0.70 if gpu_memory >= 4 else 0.55
    cache: bool | str = "disk" if image_count <= 8000 else False
    fixed_scene = scene_type == "fixed_industrial"
    hsv_h = 0.003 if fixed_scene else 0.015
    hsv_s = 0.15 if fixed_scene else 0.7
    hsv_v = 0.20 if fixed_scene else 0.4
    translate = 0.03 if fixed_scene else 0.1
    scale = 0.15 if fixed_scene else 0.5
    degrees = 0.0 if fixed_scene or direction_sensitive else 5.0
    fliplr = 0.0 if direction_sensitive or fixed_scene else 0.5
    flipud = 0.0

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
                  "已有权重微调使用 AdamW 和小学习率；从通用预训练起步则自动选择。"),
        Parameter("lr0", "初始学习率", lr0, "优化",
                  "已有任务权重采用小步长微调，避免破坏已学习特征。"
                  if initial_weights else "通用预训练起步使用标准初始学习率。"),
        Parameter("lrf", "最终学习率比例", lrf, "优化",
                  "配合余弦退火平滑收敛，并保留训练后期的细调能力。"),
        Parameter("nbs", "标称 Batch", nbs, "优化",
                  "小数据或已有权重微调时增加每轮有效参数更新次数。"),
        Parameter("cos_lr", "余弦学习率", goal != "speed", "优化",
                  "精度或均衡目标使用平滑退火；速度优先关闭以简化短训。"),
        Parameter("mosaic", "Mosaic", round(mosaic, 2), "增强",
                  "结合目标尺寸、数据量和标注质量设置增强强度。"),
        Parameter("mixup", "MixUp", round(mixup, 2), "增强",
                  "小数据或类别失衡时提供轻量正则化。"),
        Parameter("close_mosaic", "关闭 Mosaic", close_mosaic, "增强",
                  "训练末期关闭强增强以稳定定位精度。"),
        Parameter("hsv_h", "色相扰动", hsv_h, "增强",
                  "固定工业相机使用弱颜色增强，减少不符合现场的颜色失真。"),
        Parameter("hsv_s", "饱和度扰动", hsv_s, "增强",
                  "根据采集场景限制颜色变化幅度。"),
        Parameter("hsv_v", "明度扰动", hsv_v, "增强",
                  "保留合理光照扰动，同时避免偏离真实工位。"),
        Parameter("degrees", "旋转增强", degrees, "增强",
                  "固定工位或方向敏感任务不生成语义错误的旋转样本。"),
        Parameter("translate", "平移增强", translate, "增强",
                  "固定工位只保留轻量位置扰动。"),
        Parameter("scale", "缩放增强", scale, "增强",
                  "固定相机限制缩放范围，避免生成不真实拍摄距离。"),
        Parameter("fliplr", "水平翻转", fliplr, "增强",
                  "左右/方向具有语义时必须关闭水平翻转。"),
        Parameter("flipud", "垂直翻转", flipud, "增强",
                  "默认关闭垂直翻转，避免制造不真实工位。"),
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
        Parameter("seed", "随机种子", 0, "复现",
                  "候选实验固定随机性，确保参数对比公平。"),
        Parameter("deterministic", "确定性训练", True, "复现",
                  "优先保证实验可复现，便于可靠比较候选方案。"),
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
        scene_type=scene_type,
        direction_sensitive=direction_sensitive,
        initial_weights=initial_weights,
    )
    confidence = _confidence(image_count, label_quality, class_balance, len(warnings))
    summary = _summary(
        model_name,
        imgsz,
        epochs,
        goal,
        object_size,
        gpu_memory,
        initial_weights=bool(initial_weights),
        fixed_scene=fixed_scene,
    )
    command = _build_command(task, config)
    experiments = _experiment_plan(
        config,
        initial_weights=bool(initial_weights),
        fixed_scene=fixed_scene,
        gpu_memory=gpu_memory,
    )

    return {
        "model": model_name,
        "summary": summary,
        "confidence": confidence,
        "command": command,
        "parameters": [parameter.to_dict() for parameter in parameters],
        "warnings": warnings,
        "config": config,
        "experiments": experiments,
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
    if values["scene_type"] == "fixed_industrial":
        warnings.append("固定工业场景应按工件或采集批次划分数据，避免连续帧泄漏到验证集。")
    if values["direction_sensitive"]:
        warnings.append("方向具有类别语义，已关闭水平翻转和旋转增强，防止生成错误标签。")
    if values["initial_weights"]:
        warnings.append("检测到已有任务权重，将保留原检测头并采用小学习率微调；请保留原模型作为基线。")
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


def _summary(
    model: str,
    imgsz: int,
    epochs: int,
    goal: str,
    object_size: str,
    memory: float,
    *,
    initial_weights: bool,
    fixed_scene: bool,
) -> str:
    goal_text = {"balanced": "精度与速度均衡", "accuracy": "精度优先", "speed": "训练效率优先"}[goal]
    size_text = {"small": "小目标", "mixed": "混合尺寸目标", "large": "大目标"}[object_size]
    model_display = Path(model).name if initial_weights else model
    strategy = "已有权重小学习率微调" if initial_weights else "通用预训练权重训练"
    scene = "固定工业工位弱增强" if fixed_scene else "通用场景增强"
    return (
        f"基于 {memory:g} GB 显存和{size_text}场景，建议以 {model_display}、"
        f"{imgsz}px 输入训练 {epochs} 轮作为第一组基线；采用{strategy}、{scene}，"
        f"策略偏向{goal_text}。"
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
        "lr0", "lrf", "nbs", "cos_lr", "mosaic", "mixup", "close_mosaic",
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr",
        "flipud", "amp", "cache", "workers", "device", "seed", "deterministic",
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
        if key in {"data", "model"}:
            encoded = f'"{encoded}"'
        parts.append(f"{key}={encoded}")
    return " ".join(parts)


def _experiment_plan(
    config: dict[str, Any],
    *,
    initial_weights: bool,
    fixed_scene: bool,
    gpu_memory: float,
) -> list[dict[str, Any]]:
    experiments = [
        {
            "name": "Baseline",
            "change": "不改参数",
            "reason": "先得到可复现基线；后续候选保持数据划分和随机种子一致。",
            "config_overrides": {},
        }
    ]

    if initial_weights:
        experiments.append(
            {
                "name": "较小学习率",
                "change": f"仅 lr0: {config['lr0']} → {max(float(config['lr0']) / 3, 1e-5):g}",
                "reason": "验证已有任务权重是否需要更保守的微调步长。",
                "config_overrides": {
                    "lr0": max(float(config["lr0"]) / 3, 1e-5),
                },
            }
        )
    else:
        current_size = int(config["imgsz"])
        candidate_size = current_size + 160 if gpu_memory >= 10 else max(320, current_size - 128)
        candidate_size = int(math.ceil(candidate_size / 32) * 32)
        experiments.append(
            {
                "name": "分辨率对照",
                "change": f"仅 imgsz: {current_size} → {candidate_size}",
                "reason": "以实测 mAP50-95 和速度判断分辨率，不假设越大越好。",
                "config_overrides": {"imgsz": candidate_size},
            }
        )

    if fixed_scene:
        current_mosaic = float(config["mosaic"])
        candidate_mosaic = 0.0 if current_mosaic > 0 else 0.2
        experiments.append(
            {
                "name": "增强强度对照",
                "change": f"仅 mosaic: {current_mosaic:g} → {candidate_mosaic:g}",
                "reason": "固定工位需要用独立实验确认 Mosaic 是否破坏真实空间关系。",
                "config_overrides": {"mosaic": candidate_mosaic},
            }
        )
    else:
        experiments.append(
            {
                "name": "模型规模对照",
                "change": "仅更换相邻规模模型",
                "reason": "比较速度与 mAP50-95，避免多个超参数同时变化。",
                "config_overrides": {},
            }
        )
    return experiments


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
