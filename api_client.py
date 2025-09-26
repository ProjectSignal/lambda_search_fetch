"""HTTP client helpers for the Fetch Lambda to access search data services."""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from config import DATA_API_BASE_URL, DATA_API_KEY, DATA_API_TIMEOUT
from logging_config import setup_logger

logger = setup_logger(__name__)


class SearchServiceError(RuntimeError):
    """Raised when an upstream API call fails."""


def _build_headers() -> Dict[str, str]:
    """Create the default headers for data service requests."""
    headers = {"Content-Type": "application/json"}
    if DATA_API_KEY:
        headers["x-api-key"] = DATA_API_KEY
    return headers


def _parse_data(response: requests.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:  # pragma: no cover - defensive guard
        payload = {}

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def get_search_document(search_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the persisted search document for ``search_id``.

    Input: identifier string produced by searchInitializer.
    Output: Dict describing the search state or ``None`` if the service returns 404.
    """
    url = f"{DATA_API_BASE_URL}/search/{search_id}"
    try:
        response = requests.get(url, headers=_build_headers(), timeout=DATA_API_TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to fetch search {search_id}: {exc}") from exc

    if response.status_code == 404:
        return None
    if not response.ok:
        raise SearchServiceError(
            f"Search API returned {response.status_code} for {search_id}: {response.text}"
        )
    return _parse_data(response)


def create_search_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new search document record (primarily for integration tests).

    Input: ``payload`` dict following the search document schema.
    Output: Dict describing the persisted record as returned by the API.
    """
    url = f"{DATA_API_BASE_URL}/search"
    try:
        response = requests.post(
            url,
            json=payload,
            headers=_build_headers(),
            timeout=DATA_API_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to create search document: {exc}") from exc

    if not response.ok:
        raise SearchServiceError(
            f"Search create failed with {response.status_code}: {response.text}"
        )
    return _parse_data(response)


def update_search_document(
    search_id: str,
    *,
    set_fields: Optional[Dict[str, Any]] = None,
    append_events: Optional[Sequence[Dict[str, Any]]] = None,
    expected_statuses: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Apply partial updates to a search document.

    Inputs:
        - ``search_id``: unique identifier to update.
        - ``set_fields``: dict of fields and values that should overwrite the current document.
        - ``append_events``: iterable with event dicts appended to the document timeline.
        - ``expected_statuses``: optional list of allowed current statuses for optimistic concurrency.
    Output: Updated search document as returned by the API.
    """
    payload: Dict[str, Any] = {}
    if set_fields:
        payload["set"] = set_fields
    if append_events:
        payload["appendEvents"] = list(append_events)
    if expected_statuses:
        payload["expectedStatus"] = list(expected_statuses)

    url = f"{DATA_API_BASE_URL}/search/{search_id}"
    try:
        response = requests.patch(
            url,
            json=payload,
            headers=_build_headers(),
            timeout=DATA_API_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to update search {search_id}: {exc}") from exc

    if not response.ok:
        raise SearchServiceError(
            f"Search API returned {response.status_code} during update of {search_id}: {response.text}"
        )
    return _parse_data(response)


def delete_search_document(search_id: str) -> None:
    """
    Delete a search document (used for cleaning up test data).

    Input: ``search_id`` string. Output: ``None``. Missing documents are treated as success.
    """
    url = f"{DATA_API_BASE_URL}/search/{search_id}"
    try:
        response = requests.delete(url, headers=_build_headers(), timeout=DATA_API_TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to delete search document {search_id}: {exc}") from exc

    if response.status_code in (200, 202, 204, 404):
        return
    raise SearchServiceError(
        f"Search delete failed with {response.status_code}: {response.text}"
    )


def fetch_nodes_by_ids(
    node_ids: Iterable[str],
    *,
    projection: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve node documents for the supplied identifiers.

    Input:
        - ``node_ids``: collection of node identifiers as strings.
        - ``projection``: optional Mongo-style projection dict to reduce payload size.
    Output: Dict mapping nodeId -> node document.
    """
    ids = [str(node_id) for node_id in node_ids if node_id]
    if not ids:
        return {}

    payload: Dict[str, Any] = {"ids": ids}
    if projection:
        payload["projection"] = projection

    url = f"{DATA_API_BASE_URL}/nodes/bulk"
    try:
        response = requests.post(
            url,
            json=payload,
            headers=_build_headers(),
            timeout=DATA_API_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to bulk fetch nodes: {exc}") from exc

    if not response.ok:
        raise SearchServiceError(
            f"Node bulk fetch failed with {response.status_code}: {response.text}"
        )

    data = _parse_data(response)
    if isinstance(data, list):
        return {doc.get("_id") or doc.get("nodeId"): doc for doc in data}
    if isinstance(data, dict):
        return data
    logger.warning("Unexpected payload for bulk node fetch: %s", data)
    return {}


def aggregate_nodes(pipeline: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute an aggregation pipeline remotely against the nodes data service.

    Input: ``pipeline`` sequence mirroring PyMongo aggregation stages.
    Output: List of resulting documents.
    """
    url = f"{DATA_API_BASE_URL}/nodes/aggregate"
    try:
        response = requests.post(
            url,
            json={"pipeline": list(pipeline)},
            headers=_build_headers(),
            timeout=DATA_API_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to aggregate nodes: {exc}") from exc

    if not response.ok:
        raise SearchServiceError(
            f"Node aggregation failed with {response.status_code}: {response.text}"
        )

    data = _parse_data(response)
    if isinstance(data, list):
        return data
    logger.warning("Unexpected payload for node aggregation: %s", data)
    return []


def find_nodes(
    query: Dict[str, Any],
    *,
    projection: Optional[Dict[str, int]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run a filtered node search without constructing a full aggregation pipeline.

    Input:
        - ``query``: dict of filter clauses (Mongo-style).
        - ``projection``: optional projection specification.
        - ``limit``: optional maximum number of documents to return.
    Output: List of node documents matching the filter.
    """
    payload: Dict[str, Any] = {"filter": query}
    if projection:
        payload["projection"] = projection
    if limit is not None:
        payload["limit"] = limit

    url = f"{DATA_API_BASE_URL}/nodes/find"
    try:
        response = requests.post(
            url,
            json=payload,
            headers=_build_headers(),
            timeout=DATA_API_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        raise SearchServiceError(f"Failed to execute node find: {exc}") from exc

    if not response.ok:
        raise SearchServiceError(
            f"Node find failed with {response.status_code}: {response.text}"
        )

    data = _parse_data(response)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    logger.warning("Unexpected payload for node find: %s", data)
    return []
