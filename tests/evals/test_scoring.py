from unittest import TestCase

from bodhi_rag.evaluation import (
    RetrievedArtifact,
    load_fixture,
    score_grounding_suite,
    score_retrieval_suite,
)


class EvaluationScoringTest(TestCase):
    def test_retrieval_suite_tracks_hits_and_pollution(self) -> None:
        fixture = load_fixture()
        score = score_retrieval_suite(
            fixture.retrieval_cases,
            {
                "corpus-policy": [
                    RetrievedArtifact(
                        document_id="policy-release-validation",
                        origin="corpus",
                        source="docs/policy.md",
                    ),
                ],
                "conversation-recall": [
                    RetrievedArtifact(
                        document_id="conversation-turn-1",
                        origin="conversation",
                        source="memory://conversation-turn-1",
                    ),
                ],
            },
        )

        assert score.total_cases == 2
        assert score.hit_cases == 2
        assert score.pollution_free_cases == 2

    def test_grounding_suite_reports_missing_support_and_bad_citations(self) -> None:
        fixture = load_fixture()
        score = score_grounding_suite(
            fixture.grounding_cases,
            {
                "corpus-policy": [
                    RetrievedArtifact(
                        document_id="policy-release-validation",
                        origin="corpus",
                        source="docs/policy.md",
                    ),
                ],
                "conversation-recall": [],
            },
            {
                "corpus-policy": ("docs/policy.md",),
                "conversation-recall": ("memory://wrong",),
            },
        )

        assert score.total_cases == 2
        assert score.grounded_cases == 1
        failed_case = score.case_scores[1]
        assert not failed_case.grounded
        assert failed_case.missing_sources == ("memory://conversation-turn-1",)
        assert failed_case.unsupported_citations == ("memory://wrong",)
