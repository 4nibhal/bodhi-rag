from datetime import UTC, datetime

import pytest

from bodhi_rag.evaluation.thresholds import (
    EvaluationBudget,
    GroundingThresholds,
    RetrievalThresholds,
)


class TestRetrievalThresholds:
    def test_creation_with_required_fields(self) -> None:
        thresholds = RetrievalThresholds(
            hit_rate_min=0.6,
            pollution_rate_max=0.1,
            memory_leak_rate_max=0.05,
        )
        assert thresholds.hit_rate_min == 0.6
        assert thresholds.pollution_rate_max == 0.1
        assert thresholds.memory_leak_rate_max == 0.05

    def test_immutable(self) -> None:
        thresholds = RetrievalThresholds(0.5, 0.0, 0.0)
        with pytest.raises(AttributeError):
            thresholds.hit_rate_min = 0.7


class TestGroundingThresholds:
    def test_creation_with_required_fields(self) -> None:
        thresholds = GroundingThresholds(
            supported_fact_recall_min=0.8,
            unsupported_claim_rate_max=0.05,
        )
        assert thresholds.supported_fact_recall_min == 0.8
        assert thresholds.unsupported_claim_rate_max == 0.05

    def test_immutable(self) -> None:
        thresholds = GroundingThresholds(0.7, 0.0)
        with pytest.raises(AttributeError):
            thresholds.supported_fact_recall_min = 0.9


class TestEvaluationBudget:
    def test_create_with_current_timestamp(self) -> None:
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval,
            grounding=grounding,
            run_id="test-run-001",
            environment="ci",
        )
        assert budget.run_id == "test-run-001"
        assert budget.environment == "ci"
        assert budget.timestamp is not None
        assert budget.retrieval == retrieval
        assert budget.grounding == grounding

    def test_create_manual_timestamp(self) -> None:
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget(
            retrieval=retrieval,
            grounding=grounding,
            run_id="manual-run",
            timestamp=ts,
            environment="local",
        )
        assert budget.timestamp == ts

    def test_to_dict(self) -> None:
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval,
            grounding=grounding,
            run_id="dict-test",
            environment="test",
        )
        budget_dict = budget.to_dict()
        assert budget_dict["run_id"] == "dict-test"
        assert budget_dict["environment"] == "test"
        assert budget_dict["retrieval"]["hit_rate_min"] == 0.5
        assert budget_dict["grounding"]["supported_fact_recall_min"] == 0.7
