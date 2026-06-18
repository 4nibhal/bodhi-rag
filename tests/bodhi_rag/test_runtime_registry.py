from unittest import TestCase

from bodhi_rag.infrastructure.lifecycle import ManagedResource
from bodhi_rag.infrastructure.runtime_registry import RuntimeRegistry


class RuntimeRegistryTest(TestCase):
    def test_registry_caches_until_reset(self) -> None:
        created = []
        registry = RuntimeRegistry(lambda: created.append(object()) or created[-1])

        first = registry.get()
        second = registry.get()
        registry.reset()
        third = registry.get()

        assert first is second
        assert first is not third
        assert len(created) == 2

    def test_registry_start_and_stop_are_idempotent(self) -> None:
        created = []
        registry = RuntimeRegistry(lambda: created.append(object()) or created[-1])

        first = registry.start()
        second = registry.start()
        registry.stop()
        registry.stop()
        third = registry.start()

        assert first is second
        assert first is not third
        assert len(created) == 2


class ManagedResourceTest(TestCase):
    def test_stop_runs_shutdown_once_per_started_resource(self) -> None:
        created = []
        stopped = []
        resource = ManagedResource(
            lambda: created.append(object()) or created[-1],
            shutdown=stopped.append,
        )

        first = resource.start()
        resource.stop()
        resource.stop()
        second = resource.start()
        resource.stop()

        assert first is not second
        assert len(created) == 2
        assert stopped == [first, second]
