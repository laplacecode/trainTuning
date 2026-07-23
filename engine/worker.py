from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from .dataset_audit import audit_dataset
    from .recommender import recommend
    from .training_advisor import TrainingAdvisor
except ImportError:
    from dataset_audit import audit_dataset
    from recommender import recommend
    from training_advisor import TrainingAdvisor


def emit(request_id: str, event: str, **values: Any) -> None:
    payload = {"id": request_id, "event": event, **values}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def prepare_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = payload.copy()
    dataset_path = str(enriched.get("dataset_path") or "data/dataset.yaml")
    task = str(enriched.get("task") or "detect")
    scene_type = str(enriched.get("scene_type") or "general")
    audit = audit_dataset(dataset_path, task=task, scene_type=scene_type)

    if audit.get("status") in {"ok", "warning"}:
        train = (audit.get("splits") or {}).get("train") or {}
        if train.get("images"):
            enriched["image_count"] = train["images"]
        if audit.get("class_count"):
            enriched["class_count"] = audit["class_count"]

    result = recommend(enriched)
    result["dataset_audit"] = audit
    audit_warnings = audit.get("warnings") or []
    result["warnings"] = list(dict.fromkeys([*result["warnings"], *audit_warnings]))
    if audit.get("status") in {"missing", "invalid", "warning"}:
        result["confidence"] = max(40, int(result["confidence"]) - 8)
    return result


def handle_recommend(request_id: str, payload: dict[str, Any]) -> None:
    emit(request_id, "status", message="正在审计数据集并生成可复现的参数基线。")
    result = prepare_recommendation(payload)
    emit(request_id, "recommendation", result=result)


def handle_health(request_id: str) -> None:
    capabilities = {
        "python": sys.version.split()[0],
        "ultralytics": module_available("ultralytics"),
        "torch": module_available("torch"),
        "adaptive_training": True,
        "dataset_audit": True,
        "protocol": 2,
    }
    emit(request_id, "completed", message="Python Worker 运行正常。", data=capabilities)


def build_training_config(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tuning_request = payload.get("request") or {}
    recommendation = prepare_recommendation(tuning_request)
    config = recommendation["config"].copy()
    provided_config = payload.get("config")
    if isinstance(provided_config, dict):
        config.update(
            {
                str(key): value
                for key, value in provided_config.items()
                if value is not None
            }
        )
    return recommendation, config


def handle_train(request_id: str, payload: dict[str, Any]) -> None:
    recommendation, config = build_training_config(payload)
    dry_run_if_unavailable = bool(payload.get("dry_run_if_unavailable", True))
    force_simulation = bool(payload.get("force_simulation", False))
    adjustment_mode = str(payload.get("adjustment_mode") or "confirm")
    if adjustment_mode not in {"confirm", "auto", "off"}:
        adjustment_mode = "confirm"

    if force_simulation or not module_available("ultralytics"):
        if not dry_run_if_unavailable:
            emit(request_id, "error", message="当前 Python 环境未安装 ultralytics。")
            return
        simulate_training(
            request_id,
            recommendation,
            config,
            adjustment_mode=adjustment_mode,
        )
        return

    run_ultralytics(
        request_id,
        config,
        adjustment_mode=adjustment_mode,
    )


def run_ultralytics(
    request_id: str,
    config: dict[str, Any],
    *,
    adjustment_mode: str,
) -> None:
    from ultralytics import YOLO

    effective_config = config.copy()
    model_name = str(effective_config.pop("model"))
    task = effective_config.pop("task", None)
    effective_config["project"] = str(Path.cwd() / "runs")
    effective_config["name"] = f"tuner-{time.strftime('%Y%m%d-%H%M%S')}"
    effective_config["exist_ok"] = False
    effective_config["plots"] = True
    advisor = TrainingAdvisor()

    emit(request_id, "status", message=f"正在加载 {Path(model_name).name}。")
    model = YOLO(model_name, task=task)

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        epochs = int(getattr(trainer, "epochs", effective_config.get("epochs", 100)))
        metrics = collect_trainer_metrics(trainer)
        current_lr = current_learning_rate(trainer)
        if current_lr is not None:
            metrics["lr"] = current_lr
        fitness = metrics.get("fitness")
        message = (
            f"训练中，fitness={fitness:.4f}"
            if isinstance(fitness, (int, float))
            else "训练中，正在收集验证指标"
        )
        emit(
            request_id,
            "progress",
            progress=round(epoch / max(1, epochs) * 100, 2),
            epoch=epoch,
            epochs=epochs,
            message=message,
            data=metrics,
        )

        proposal = advisor.observe(
            epoch=epoch,
            epochs=epochs,
            metrics=metrics,
            current_lr=current_lr,
        )
        if proposal is None or adjustment_mode == "off":
            return
        accepted = (
            True
            if adjustment_mode == "auto"
            else wait_for_adjustment_response(request_id, proposal)
        )
        applied = apply_adjustment(trainer, proposal) if accepted else {}
        advisor.record_decision(proposal, accepted=accepted, applied=applied)
        emit(
            request_id,
            "adjustment_applied" if accepted else "adjustment_skipped",
            message=(
                f"已应用：{proposal['title']}"
                if accepted
                else f"已跳过：{proposal['title']}"
            ),
            adjustment={**proposal, "applied": applied},
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    emit(request_id, "status", message="Ultralytics 训练已启动；将以验证集指标判断调整。")
    model.train(**effective_config)

    trainer = getattr(model, "trainer", None)
    save_dir = Path(
        str(getattr(trainer, "save_dir", effective_config["project"]))
    ).resolve()
    summary = advisor.summary()
    summary["save_dir"] = str(save_dir)
    summary["best_weights"] = str(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    summary["last_weights"] = str(getattr(trainer, "last", save_dir / "weights" / "last.pt"))
    write_tuning_history(
        save_dir,
        model_name=model_name,
        config={
            **effective_config,
            "model": model_name,
            "task": task,
        },
        summary=summary,
    )
    emit(
        request_id,
        "training_summary",
        message=training_summary_message(summary),
        data=summary,
    )
    emit(
        request_id,
        "completed",
        message=f"训练完成，结果保存在 {save_dir}；部署优先使用 best.pt。",
        progress=100,
    )


def simulate_training(
    request_id: str,
    recommendation: dict[str, Any],
    config: dict[str, Any],
    *,
    adjustment_mode: str,
) -> None:
    epochs = 12
    advisor = TrainingAdvisor(
        min_epochs=8,
        plateau_patience=3,
        early_stop_patience=8,
        cooldown_epochs=3,
    )
    current_lr = float(config.get("lr0", 0.01))
    emit(
        request_id,
        "status",
        message="未检测到 Ultralytics，正在进行 12 轮界面、指标与人工确认模拟。",
    )
    for epoch in range(1, epochs + 1):
        time.sleep(0.08)
        progress = round(epoch / epochs * 100, 2)
        learning_epoch = min(epoch, 6)
        simulated_map = 0.18 + 0.52 * (1 - pow(0.62, learning_epoch))
        if epoch > 6:
            simulated_map += (epoch - 6) * 0.0001
        simulated_loss = 2.4 * pow(0.84, epoch) + 0.35
        metrics = {
            "metrics/precision(B)": round(min(0.92, simulated_map + 0.08), 4),
            "metrics/recall(B)": round(max(0.0, simulated_map - 0.04), 4),
            "metrics/mAP50(B)": round(simulated_map + 0.10, 4),
            "metrics/mAP50-95(B)": round(simulated_map, 4),
            "train/box_loss": round(simulated_loss, 4),
            "val/box_loss": round(simulated_loss * (1.05 + epoch * 0.012), 4),
            "fitness": round(simulated_map, 4),
            "lr": current_lr,
        }
        emit(
            request_id,
            "progress",
            progress=progress,
            epoch=epoch,
            epochs=epochs,
            message=f"模拟 mAP50-95={simulated_map:.3f}, loss={simulated_loss:.3f}",
            data=metrics,
        )
        proposal = advisor.observe(
            epoch=epoch,
            epochs=epochs,
            metrics=metrics,
            current_lr=current_lr,
        )
        if proposal is None or adjustment_mode == "off":
            continue
        accepted = (
            True
            if adjustment_mode == "auto"
            else wait_for_adjustment_response(request_id, proposal)
        )
        applied: dict[str, Any] = {}
        if accepted and proposal["action"] == "reduce_lr":
            current_lr = float(proposal["proposed_value"])
            applied = {"learning_rate": current_lr}
        elif accepted and proposal["action"] == "stop_early":
            applied = {"stopped": True}
        advisor.record_decision(proposal, accepted=accepted, applied=applied)
        emit(
            request_id,
            "adjustment_applied" if accepted else "adjustment_skipped",
            message=(
                f"已应用：{proposal['title']}"
                if accepted
                else f"已跳过：{proposal['title']}"
            ),
            adjustment={**proposal, "applied": applied},
        )
        if applied.get("stopped"):
            break

    summary = advisor.summary()
    summary["save_dir"] = "模拟训练不生成权重"
    emit(
        request_id,
        "training_summary",
        message=training_summary_message(summary),
        data=summary,
    )
    emit(
        request_id,
        "completed",
        message=f"模拟训练完成；真实训练将使用 {Path(recommendation['model']).name}。",
        progress=100,
    )


def wait_for_adjustment_response(
    request_id: str,
    proposal: dict[str, Any],
) -> bool:
    emit(
        request_id,
        "adjustment_proposed",
        message=proposal["title"],
        adjustment=proposal,
    )
    while True:
        line = sys.stdin.readline()
        if not line:
            return False
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get("action") != "adjustment_response":
            continue
        proposal_id = str(response.get("proposal_id") or "")
        if proposal_id and proposal_id != proposal["proposal_id"]:
            continue
        return bool(response.get("accepted", False))


def apply_adjustment(trainer: Any, proposal: dict[str, Any]) -> dict[str, Any]:
    action = proposal["action"]
    if action == "stop_early":
        trainer.stop = True
        return {"stopped": True}
    if action != "reduce_lr":
        return {}

    optimizer = getattr(trainer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", []) if optimizer else []
    if not param_groups:
        return {"warning": "优化器尚未提供可调整的参数组。"}

    current = current_learning_rate(trainer)
    target = float(proposal["proposed_value"])
    factor = target / current if current and current > 0 else 0.5
    applied_values: list[float] = []
    for group in param_groups:
        group["lr"] = max(float(group.get("lr", target)) * factor, 1e-7)
        if "initial_lr" in group:
            group["initial_lr"] = max(float(group["initial_lr"]) * factor, 1e-7)
        applied_values.append(group["lr"])

    scheduler = getattr(trainer, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [
            max(float(value) * factor, 1e-7) for value in scheduler.base_lrs
        ]
    return {
        "learning_rates": applied_values,
        "factor": factor,
    }


def collect_trainer_metrics(trainer: Any) -> dict[str, Any]:
    metrics = serializable_metrics(getattr(trainer, "metrics", {}) or {})
    fitness = getattr(trainer, "fitness", None)
    if isinstance(fitness, (int, float)):
        metrics["fitness"] = float(fitness)

    label_loss_items = getattr(trainer, "label_loss_items", None)
    train_loss = getattr(trainer, "tloss", None)
    if callable(label_loss_items) and train_loss is not None:
        try:
            metrics.update(serializable_metrics(label_loss_items(train_loss, prefix="train")))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return metrics


def current_learning_rate(trainer: Any) -> float | None:
    optimizer = getattr(trainer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", []) if optimizer else []
    values = [
        float(group["lr"])
        for group in param_groups
        if isinstance(group.get("lr"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def write_tuning_history(
    save_dir: Path,
    *,
    model_name: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": model_name,
            "config": config,
            "summary": summary,
        }
        (save_dir / "train_tuning_history.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exception:
        print(f"无法写入调优记录：{exception}", file=sys.stderr, flush=True)


def training_summary_message(summary: dict[str, Any]) -> str:
    best_epoch = summary.get("best_epoch")
    best_score = summary.get("best_score")
    if best_epoch and isinstance(best_score, (int, float)):
        return (
            f"训练复盘：最佳验证指标 {best_score:.4f} 出现在 Epoch {best_epoch}；"
            "请使用 best.pt，并以单变量方式安排下一组实验。"
        )
    return "训练复盘：未获得可比较的验证指标，请检查 val 配置和数据集。"


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
