from __future__ import annotations

import unittest

from engine.training_advisor import TrainingAdvisor
from engine.worker import apply_adjustment


class TrainingAdvisorTests(unittest.TestCase):
    def test_plateau_proposes_learning_rate_reduction(self) -> None:
        advisor = TrainingAdvisor(
            min_epochs=4,
            plateau_patience=3,
            early_stop_patience=8,
            min_delta=0.01,
            cooldown_epochs=2,
        )
        proposal = None
        for epoch, score in enumerate(
            [0.20, 0.35, 0.50, 0.60, 0.602, 0.603, 0.604],
            start=1,
        ):
            proposal = advisor.observe(
                epoch=epoch,
                epochs=30,
                metrics={"metrics/mAP50-95(B)": score},
                current_lr=0.001,
            )

        self.assertIsNotNone(proposal)
        self.assertEqual("reduce_lr", proposal["action"])
        self.assertAlmostEqual(0.0005, proposal["proposed_value"])

    def test_sustained_improvement_does_not_propose_adjustment(self) -> None:
        advisor = TrainingAdvisor(
            min_epochs=4,
            plateau_patience=2,
            min_delta=0.001,
        )
        proposals = [
            advisor.observe(
                epoch=epoch,
                epochs=20,
                metrics={"metrics/mAP50-95(B)": epoch * 0.02},
                current_lr=0.001,
            )
            for epoch in range(1, 11)
        ]

        self.assertTrue(all(proposal is None for proposal in proposals))

    def test_summary_keeps_best_epoch_instead_of_last_epoch(self) -> None:
        advisor = TrainingAdvisor(min_epochs=20)
        for epoch, score in enumerate([0.2, 0.4, 0.55, 0.51], start=1):
            advisor.observe(
                epoch=epoch,
                epochs=4,
                metrics={"metrics/mAP50-95(B)": score},
                current_lr=0.001,
            )

        summary = advisor.summary()
        self.assertEqual(3, summary["best_epoch"])
        self.assertEqual(4, summary["final_epoch"])
        self.assertGreater(summary["best_score"], summary["final_score"])

    def test_learning_rate_adjustment_updates_optimizer_and_scheduler(self) -> None:
        class Optimizer:
            param_groups = [
                {"lr": 0.001, "initial_lr": 0.001},
                {"lr": 0.001, "initial_lr": 0.001},
            ]

        class Scheduler:
            base_lrs = [0.001, 0.001]

        class Trainer:
            optimizer = Optimizer()
            scheduler = Scheduler()

        result = apply_adjustment(
            Trainer(),
            {
                "action": "reduce_lr",
                "proposed_value": 0.0005,
            },
        )

        self.assertEqual([0.0005, 0.0005], result["learning_rates"])
        self.assertEqual([0.0005, 0.0005], Trainer.scheduler.base_lrs)


if __name__ == "__main__":
    unittest.main()
