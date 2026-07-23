from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from recommender import recommend


def emit(request_id: str, event: str, **values: Any) -> None:
    payload = {"id": request_id, "event": event, **values}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def handle_recommend(request_id: str, payload: dict[str, Any]) -> None:
    emit(request_id, "status", message="正在运行本地参数规则引擎。")
    result = recommend(payload)
    emit(request_id, "recommendation", result=result)


def handle_health(request_id: str) -> None:
    capabilities = {
        "python": sys.version.split()[0],
        "ultralytics": module_available("ultralytics"),
        "torch": module_available("torch"),
        "protocol": 1,
    }
    emit(request_id, "completed", message="Python Worker 运行正常。", data=capabilities)


def handle_train(request_id: str, payload: dict[str, Any]) -> None:
    tuning_request = payload.get("request") or {}
    recommendation = recommend(tuning_request)
    config = recommendation["config"].copy()
    dry_run_if_unavailable = bool(payload.get("dry_run_if_unavailable", True))

    if not module_available("ultralytics"):
        if not dry_run_if_unavailable:
            emit(request_id, "error", message="当前 Python 环境未安装 ultralytics。")
            return
        simulate_training(request_id, recommendation)
        return

    run_ultralytics(request_id, config)


def run_ultralytics(request_id: str, config: dict[str, Any]) -> None:
    from ultralytics import YOLO

    model_name = str(config.pop("model"))
    task = config.pop("task", None)
    config["project"] = str(Path.cwd() / "runs")
    config["name"] = f"tuner-{time.strftime('%Y%m%d-%H%M%S')}"

    emit(request_id, "status", message=f"正在加载 {model_name}。")
    model = YOLO(model_name, task=task)

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        epochs = int(getattr(trainer, "epochs", config.get("epochs", 100)))
        metrics = getattr(trainer, "metrics", {}) or {}
        fitness = getattr(trainer, "fitness", None)
        message = f"训练中，fitness={fitness:.4f}" if isinstance(fitness, (int, float)) else "训练中"
        emit(
            request_id,
            "progress",
            progress=round(epoch / max(1, epochs) * 100, 2),
            epoch=epoch,
            epochs=epochs,
            message=message,
            data=serializable_metrics(metrics),
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    emit(request_id, "status", message="Ultralytics 训练已启动。")
    results = model.train(**config)
    save_dir = str(getattr(results, "save_dir", config["project"]))
    emit(request_id, "completed", message=f"训练完成，结果保存在 {save_dir}。", progress=100)


def simulate_training(request_id: str, recommendation: dict[str, Any]) -> None:
    epochs = 12
    emit(
        request_id,
        "status",
        message="未检测到 Ultralytics，正在进行 12 轮界面与通信模拟。",
    )
    for epoch in range(1, epochs + 1):
        time.sleep(0.18)
        progress = round(epoch / epochs * 100, 2)
        simulated_map = 0.18 + 0.62 * (1 - pow(0.78, epoch))
        simulated_loss = 2.4 * pow(0.84, epoch) + 0.35
        emit(
            request_id,
            "progress",
            progress=progress,
            epoch=epoch,
            epochs=epochs,
            message=f"模拟 mAP50={simulated_map:.3f}, loss={simulated_loss:.3f}",
            data={"metrics/mAP50(B)": round(simulated_map, 4), "train/loss": round(simulated_loss, 4)},
        )
    emit(
        request_id,
        "completed",
        message=f"模拟训练完成；真实训练将使用 {recommendation['model']}。",
        progress=100,
    )


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        elif hasattr(value, "item"):
            try:
                result[str(key)] = value.item()
            except (TypeError, ValueError):
                pass
    return result


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        return 0

    request: dict[str, Any] = {}
    try:
        request = json.loads(line)
        request_id = str(request.get("id") or "")
        action = str(request.get("action") or "")
        payload = request.get("payload") or {}

        if action == "recommend":
            handle_recommend(request_id, payload)
        elif action == "train":
            handle_train(request_id, payload)
        elif action == "health":
            handle_health(request_id)
        else:
            emit(request_id, "error", message=f"不支持的动作：{action}")
            return 2
        return 0
    except Exception as exception:
        request_id = str(request.get("id") or "")
        emit(request_id, "error", message=f"{type(exception).__name__}: {exception}")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
