"""
Hardening contract test for CVE-2026-45829 (client-side variant).

CVE-2026-45829 has two components:
  - server-side: in `chromadb/server/fastapi/__init__.py`, reachable only
    when the client uses `chromadb.HttpClient` against a remote server.
    bodhi-rag uses `chromadb.PersistentClient` (embedded mode) so this path
    is unreachable. The `chromadb/chroma` server image is no longer
    pulled by `podman-compose.yml` either.
  - client-side: in `chromadb/api/models/CollectionCommon.py:_embed`,
    reachable only when a collection method is called WITHOUT
    pre-computed embeddings (`embeddings=` or `query_embeddings=`
    set to None or absent). The `_embed` method then falls through to
    `self.configuration.get("embedding_function")`, which calls
    `load_collection_configuration_from_json` and instantiates the
    embedding function from the stored config. A poisoned
    `configuration_json` in the local SQLite db can hijack this.

bodhi-rag's chroma adapter must always supply pre-computed embeddings to
`collection.add`, `query`, `update`, and `upsert`. This test asserts
that structural property of the source so that any future regression
is caught at PR time, not at security-audit time.

Reference: VERSIONS.md and the comment in pyproject.toml at
`chromadb==1.5.9`.
"""

from __future__ import annotations

import ast
from pathlib import Path

CHROMA_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bodhi_rag"
    / "infrastructure"
    / "vector_store"
    / "chroma.py"
)

METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS: dict[str, str] = {
    "add": "embeddings",
    "query": "query_embeddings",
    "update": "embeddings",
    "upsert": "embeddings",
}


def _is_collection_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "_collection"
        )
    return isinstance(node, ast.Name) and node.id == "collection"


def _iter_collection_method_calls(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    """
    Yield (method_name, Call node) for every invocation of a collection method.

    Handles both direct calls and `asyncio.to_thread(...)` call-by-reference
    patterns, whether the adapter uses `self._collection` inline or first binds
    it to a local `collection` variable.
    """
    results: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        if _is_collection_target(func.value):
            results.append((func.attr, node))
            continue

        if (
            func.attr == "to_thread"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
        ):
            if not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Attribute):
                continue
            if not _is_collection_target(first_arg.value):
                continue
            results.append((first_arg.attr, node))
    return results


def _kwargs_of(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def test_chroma_adapter_always_passes_precomputed_embeddings() -> None:
    """Every relevant collection call must pass pre-computed embeddings."""
    assert CHROMA_ADAPTER_PATH.exists(), (
        f"Expected chroma adapter at {CHROMA_ADAPTER_PATH}, but the file is missing."
    )
    source = CHROMA_ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    checked: list[tuple[str, str, int]] = []
    violations: list[str] = []

    for method, call in _iter_collection_method_calls(tree):
        if method not in METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS:
            continue
        kwarg_name = METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS[method]
        kwargs = _kwargs_of(call)
        line = call.lineno

        if kwarg_name not in kwargs:
            violations.append(
                f"Line {line}: collection.{method}(...) is missing `{kwarg_name}=`. "
                "This would re-activate the client-side vulnerability path of "
                "CVE-2026-45829.",
            )
            continue
        if isinstance(kwargs[kwarg_name], ast.Constant) and kwargs[kwarg_name].value is None:
            violations.append(
                f"Line {line}: collection.{method}(..., {kwarg_name}=None, ...) passes "
                "None. This would re-activate the client-side vulnerability path of "
                "CVE-2026-45829.",
            )
            continue
        checked.append((method, kwarg_name, line))

    assert not violations, "\n".join(violations)
    assert checked, (
        "No collection add/query/update/upsert calls were found in chroma.py. "
        "If the adapter was rewritten, update this test to track the new call "
        "sites or remove it if the vulnerability is no longer relevant."
    )
