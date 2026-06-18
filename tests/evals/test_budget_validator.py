from unittest import TestCase

from bodhi_rag.evaluation import (
    BudgetValidator,
    GroundingCaseScore,
    GroundingSuiteScore,
    GroundingThresholds,
    RetrievalCaseScore,
    RetrievalSuiteScore,
    RetrievalThresholds,
)
from bodhi_rag.evaluation.thresholds import EvaluationBudget


class BudgetValidatorRetrievalTest(TestCase):
    def setUp(self) -> None:
        self.validator = BudgetValidator()

    def test_validate_retrieval_passes_within_thresholds(self) -> None:
        thresholds = RetrievalThresholds(
            hit_rate_min=0.5,
            pollution_rate_max=0.0,
            memory_leak_rate_max=0.0,
        )
        score = RetrievalSuiteScore(
            total_cases=10,
            hit_cases=6,
            pollution_free_cases=10,
            case_scores=tuple(
                RetrievalCaseScore(
                    query_id=f"q{i}",
                    hit=True,
                    polluted=False,
                    matched_document_ids=("doc1",),
                    forbidden_origins=(),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_retrieval(score, thresholds)
        assert passes
        assert failures == []

    def test_validate_retrieval_fails_low_hit_rate(self) -> None:
        thresholds = RetrievalThresholds(
            hit_rate_min=0.5,
            pollution_rate_max=0.0,
            memory_leak_rate_max=0.0,
        )
        score = RetrievalSuiteScore(
            total_cases=10,
            hit_cases=3,
            pollution_free_cases=10,
            case_scores=tuple(
                RetrievalCaseScore(
                    query_id=f"q{i}",
                    hit=False,
                    polluted=False,
                    matched_document_ids=(),
                    forbidden_origins=(),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_retrieval(score, thresholds)
        assert not passes
        assert len(failures) == 1
        assert "hit_rate" in failures[0]

    def test_validate_retrieval_fails_pollution(self) -> None:
        thresholds = RetrievalThresholds(
            hit_rate_min=0.5,
            pollution_rate_max=0.0,
            memory_leak_rate_max=0.0,
        )
        score = RetrievalSuiteScore(
            total_cases=10,
            hit_cases=6,
            pollution_free_cases=8,
            case_scores=tuple(
                RetrievalCaseScore(
                    query_id=f"q{i}",
                    hit=True,
                    polluted=i < 2,
                    matched_document_ids=("doc1",),
                    forbidden_origins=("forbidden",) if i < 2 else (),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_retrieval(score, thresholds)
        assert not passes
        assert len(failures) == 2  # pollution and memory_leak both fail

    def test_validate_retrieval_empty_suite(self) -> None:
        thresholds = RetrievalThresholds(0.5, 0.0, 0.0)
        score = RetrievalSuiteScore(
            total_cases=0,
            hit_cases=0,
            pollution_free_cases=0,
            case_scores=(),
        )
        passes, _failures = self.validator.validate_retrieval(score, thresholds)
        assert not passes  # 0/0 = 0.0, which fails hit_rate_min threshold


class BudgetValidatorGroundingTest(TestCase):
    def setUp(self) -> None:
        self.validator = BudgetValidator()

    def test_validate_grounding_passes_within_thresholds(self) -> None:
        thresholds = GroundingThresholds(
            supported_fact_recall_min=0.7,
            unsupported_claim_rate_max=0.0,
        )
        score = GroundingSuiteScore(
            total_cases=10,
            grounded_cases=8,
            case_scores=tuple(
                GroundingCaseScore(
                    query_id=f"q{i}",
                    grounded=True,
                    supported_sources=("source1",),
                    missing_sources=(),
                    unsupported_citations=(),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_grounding(score, thresholds)
        assert passes
        assert failures == []

    def test_validate_grounding_fails_low_recall(self) -> None:
        thresholds = GroundingThresholds(
            supported_fact_recall_min=0.7,
            unsupported_claim_rate_max=0.0,
        )
        score = GroundingSuiteScore(
            total_cases=10,
            grounded_cases=5,
            case_scores=tuple(
                GroundingCaseScore(
                    query_id=f"q{i}",
                    grounded=i < 5,
                    supported_sources=("source1",) if i < 5 else (),
                    missing_sources=("source1",) if i >= 5 else (),
                    unsupported_citations=(),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_grounding(score, thresholds)
        assert not passes
        assert len(failures) == 1
        assert "supported_fact_recall" in failures[0]

    def test_validate_grounding_fails_unsupported_claims(self) -> None:
        thresholds = GroundingThresholds(
            supported_fact_recall_min=0.7,
            unsupported_claim_rate_max=0.0,
        )
        score = GroundingSuiteScore(
            total_cases=10,
            grounded_cases=10,
            case_scores=tuple(
                GroundingCaseScore(
                    query_id=f"q{i}",
                    grounded=True,
                    supported_sources=("source1",),
                    missing_sources=(),
                    unsupported_citations=("bad_source",),
                )
                for i in range(10)
            ),
        )
        passes, failures = self.validator.validate_grounding(score, thresholds)
        assert not passes
        assert len(failures) == 1
        assert "unsupported_claim_rate" in failures[0]


class BudgetValidatorFullBudgetTest(TestCase):
    def setUp(self) -> None:
        self.validator = BudgetValidator()

    def test_validate_budget_combined(self) -> None:
        retrieval_thresholds = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding_thresholds = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval_thresholds,
            grounding=grounding_thresholds,
            run_id="budget-test",
            environment="ci",
        )
        retrieval_score = RetrievalSuiteScore(
            total_cases=10,
            hit_cases=6,
            pollution_free_cases=10,
            case_scores=tuple(
                RetrievalCaseScore(
                    query_id=f"q{i}",
                    hit=True,
                    polluted=False,
                    matched_document_ids=("doc1",),
                    forbidden_origins=(),
                )
                for i in range(10)
            ),
        )
        grounding_score = GroundingSuiteScore(
            total_cases=10,
            grounded_cases=8,
            case_scores=tuple(
                GroundingCaseScore(
                    query_id=f"q{i}",
                    grounded=True,
                    supported_sources=("source1",),
                    missing_sources=(),
                    unsupported_citations=(),
                )
                for i in range(10)
            ),
        )
        passes, details = self.validator.validate_budget(
            budget, (retrieval_score, grounding_score),
        )
        assert passes
        assert details["retrieval_passes"]
        assert details["grounding_passes"]

    def test_validate_budget_partial_failure(self) -> None:
        retrieval_thresholds = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding_thresholds = GroundingThresholds(0.9, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval_thresholds,
            grounding=grounding_thresholds,
            run_id="budget-test",
            environment="ci",
        )
        retrieval_score = RetrievalSuiteScore(
            total_cases=10,
            hit_cases=6,
            pollution_free_cases=10,
            case_scores=tuple(
                RetrievalCaseScore(
                    query_id=f"q{i}",
                    hit=True,
                    polluted=False,
                    matched_document_ids=("doc1",),
                    forbidden_origins=(),
                )
                for i in range(10)
            ),
        )
        grounding_score = GroundingSuiteScore(
            total_cases=10,
            grounded_cases=8,
            case_scores=tuple(
                GroundingCaseScore(
                    query_id=f"q{i}",
                    grounded=True,
                    supported_sources=("source1",),
                    missing_sources=(),
                    unsupported_citations=(),
                )
                for i in range(10)
            ),
        )
        passes, details = self.validator.validate_budget(
            budget, (retrieval_score, grounding_score),
        )
        assert not passes
        assert details["retrieval_passes"]
        assert not details["grounding_passes"]
