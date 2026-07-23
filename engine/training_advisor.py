from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4


SCORE_KEYS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95",
    "fitness",
    "metrics/mAP50(B)",
    "metrics/mAP50(M)",
    "metrics/mAP50",
)


@dataclass(slots=True)
class MetricSnapshot:
    epoch: int
    score: float
    metrics: dict[str, float]


class TrainingAdvisor:
    """Analyze validation trends and propose only low-risk live adjustments."""

    def __init__(
        self,
        *,
        min_epochs: int = 10,
        plateau_patience: int = 6,
        early_stop_patience: int = 20,
        min_delta: float = 0.002,
        cooldown_epochs: int = 5,
    ) -> None:
        self.min_epochs = min_epochs
        self.plateau_patience = plateau_patience
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.cooldown_epochs = cooldown_epochs
        self.history: list[MetricSnapshot] = []
        self.decisions: list[dict[str, Any]] = []
        self.best_score = float("-inf")
        self.best_epoch = 0
        self._last_proposal_epoch = 0
        self._reduction_count = 0
        self._skipped_actions: set[str] = set()

    def observe(
        self,
        *,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
        current_lr: float | None,
    ) -> dict[str, Any] | None:
        numeric_metrics = _numeric_metrics(metrics)
        score = _score_from_metrics(numeric_metrics)
        if score is None:
            return None

        self.history.append(MetricSnapshot(epoch, score, numeric_metrics))
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch

        stale_epochs = epoch - self.best_epoch
        if epoch < self.min_epochs:
            return None
        if epoch - self._last_proposal_epoch < self.cooldown_epochs:
            return None

        if (
            stale_epochs >= self.early_stop_patience
            and "stop_early" not in self._skipped_actions
            and epoch >= max(self.min_epochs, round(epochs * 0.35))
        ):
            return self._proposal(
                epoch=epoch,
                action="stop_early",
                title="验证指标长期没有改善，建议提前停止",
                reason=(
                    f"主要验证指标已连续 {stale_epochs} 轮没有显著超过"
                    f"第 {self.best_epoch} 轮的最佳值。继续训练更可能增加过拟合和时间成本。"
                ),
                evidence=[
                    f"当前主指标：{score:.4f}",
                    f"最佳主指标：{self.best_score:.4f}（Epoch {self.best_epoch}）",
                    f"无显著改善：{stale_epochs} Epoch",
                    "停止后保留 Ultralytics 生成的 best.pt，不以 last.pt 覆盖。",
                ],
                current_value=epoch,
                proposed_value=self.best_epoch,
                confidence=min(95, 70 + stale_epochs),
            )

        if (
            stale_epochs >= self.plateau_patience
            and self._reduction_count < 2
            and "reduce_lr" not in self._skipped_actions
            and current_lr is not None
            and current_lr > 2e-6
            and epoch < round(epochs * 0.9)
        ):
            return self._proposal(
                epoch=epoch,
                action="reduce_lr",
                title="验证指标进入平台期，建议降低当前学习率",
                reason=(
                    f"主指标已连续 {stale_epochs} 轮没有显著改善。"
                    "降低学习率可以在保留当前权重的同时进行更细的局部搜索。"
                ),
                evidence=[
                    f"当前主指标：{score:.4f}",
                    f"最佳主指标：{self.best_score:.4f}（Epoch {self.best_epoch}）",
                    f"当前学习率：{current_lr:.6g}",
                    "只调整优化器学习率，不改变模型、batch、分辨率或数据增强。",
                ],
                current_value=current_lr,
                proposed_value=max(current_lr * 0.5, 1e-7),
                confidence=min(90, 62 + stale_epochs * 3),
            )
        return None

    def record_decision(
        self,
        proposal: dict[str, Any],
        *,
        accepted: bool,
        applied: dict[str, Any] | None = None,
    ) -> None:
        action = str(proposal["action"])
        epoch = int(proposal["epoch"])
        self._last_proposal_epoch = epoch
        if accepted and action == "reduce_lr":
            self._reduction_count += 1
            self.best_epoch = epoch
        elif not accepted:
            self._skipped_actions.add(action)

        self.decisions.append(
            {
                "proposal_id": proposal["proposal_id"],
                "epoch": epoch,
                "action": action,
                "accepted": accepted,
                "applied": applied or {},
                "reason": proposal["reason"],
            }
        )

    def summary(self) -> dict[str, Any]:
        final_epoch = self.history[-1].epoch if self.history else 0
        final_score = self.history[-1].score if self.history else None
        return {
            "best_epoch": self.best_epoch or None,
            "best_score": self.best_score if self.best_epoch else None,
            "final_epoch": final_epoch,
            "final_score": final_score,
            "metric": "mAP50-95（不可用时回退到 fitness 或 mAP50）",
            "decisions": self.decisions,
            "history": [asdict(item) for item in self.history],
            "recommendations": _next_experiment_recommendations(
                best_epoch=self.best_epoch,
                final_epoch=final_epoch,
                best_score=self.best_score if self.best_epoch else None,
                final_score=final_score,
            ),
        }

    def _proposal(
        self,
        *,
        epoch: int,
        action: str,
        title: str,
        reason: str,
        evidence: list[str],
        current_value: float | int,
        proposed_value: float | int,
        confidence: int,
    ) -> dict[str, Any]:
        self._last_proposal_epoch = epoch
        return {
            "proposal_id": uuid4().hex,
            "epoch": epoch,
            "action": action,
            "title": title,
            "reason": reason,
            "evidence": evidence,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "confidence": confidence,
            "requires_restart": False,
        }


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _score_from_metrics(metrics: dict[str, float]) -> float | None:
    for key in SCORE_KEYS:
        if key in metrics:
            return metrics[key]
    return None


def _next_experiment_recommendations(
    *,
    best_epoch: int,
    final_epoch: int,
    best_score: float | None,
    final_score: float | None,
) -> list[str]:
    recommendations = [
        "部署和复验优先使用 best.pt；last.pt 只用于断点续训。",
        "下一组实验只改变一个因素，并保持数据划分、随机种子和评估指标一致。",
    ]
    if best_epoch and final_epoch and best_epoch < final_epoch * 0.75:
        recommendations.append(
            f"最佳点出现在 Epoch {best_epoch}，明显早于结束轮次；下一次可缩短 epochs 或 patience。"
        )
    if (
        best_score is not None
        and final_score is not None
        and best_score - final_score > 0.01
    ):
        recommendations.append(
            "最终指标低于最佳点，建议检查过拟合，并比较更弱增强或更小学习率的独立实验。"
        )
    return recommendations
