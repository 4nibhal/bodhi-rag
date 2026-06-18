from unittest import TestCase

from bodhi_rag.evaluation import load_fixture


class EvaluationFixtureTest(TestCase):
    def test_packaged_fixture_loads_with_expected_cases(self) -> None:
        fixture = load_fixture()

        assert fixture.name == "retrieval-grounding-baseline"
        assert tuple(case.query_id for case in fixture.retrieval_cases) == (
            "corpus-policy",
            "conversation-recall",
        )
        assert tuple(case.query_id for case in fixture.grounding_cases) == (
            "corpus-policy",
            "conversation-recall",
        )
