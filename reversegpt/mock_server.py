"""Stand-in for the DevRev API so the whole pipeline can run offline.

Responses are deterministic: the same inputs always give the same IDs, which
keeps runs reproducible and tests stable. Output keys match the ``returns``
names in data/api_documentation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ORG = "don:core:dvrv-us-1:devo/0"
IDENTITY = "don:identity:dvrv-us-1:devo/0"


def _numbers(seed: Any, count: int) -> list[int]:
    digest = hashlib.sha256(json.dumps(seed, sort_keys=True, default=str).encode()).hexdigest()
    return [int(digest[i * 4 : i * 4 + 4], 16) % 900 + 100 for i in range(count)]


def _count(seed: Any, low: int = 2, high: int = 4) -> int:
    return low + _numbers(("count", seed), 1)[0] % (high - low + 1)


def who_am_i() -> dict[str, Any]:
    user_id = f"{IDENTITY}:devu/1"
    return {"user_id": user_id, "message": f"The current user is {user_id}."}


def get_sprint_id() -> dict[str, Any]:
    sprint_id = f"{ORG}:sprint/7"
    return {"sprint_id": sprint_id, "message": f"The current sprint is {sprint_id}."}


def work_list(limit: int = 50, **filters: Any) -> dict[str, Any]:
    kinds = filters.get("type") or ["issue"]
    kind = kinds[0] if isinstance(kinds, list) else str(kinds)
    if any(k.startswith("ticket_") for k in filters):
        kind = "ticket"
    n = min(int(limit or 50), _count(filters))
    work_ids = [f"{ORG}:{kind}/{num}" for num in _numbers(filters, n)]
    return {"work_ids": work_ids, "message": f"Found {n} work items matching the filters."}


def search_object_by_name(query: str) -> dict[str, Any]:
    object_id = f"{IDENTITY}:revo/{_numbers(query, 1)[0]}"
    return {"id": object_id, "message": f"Best match for {query!r} is {object_id}."}


def get_similar_work_items(work_id: str) -> dict[str, Any]:
    ids = [f"{ORG}:issue/{num}" for num in _numbers(work_id, _count(work_id))]
    return {
        "similar_work_items": ids,
        "message": f"Found {len(ids)} work items similar to {work_id}.",
    }


def prioritize_objects(objects: list[str]) -> dict[str, Any]:
    objects = objects if isinstance(objects, list) else [objects]
    ranked = sorted(objects, key=lambda o: _numbers(o, 1)[0])
    return {"prioritized_objects": ranked, "message": f"Prioritized {len(ranked)} objects."}


def summarize_objects(objects: list[str]) -> dict[str, Any]:
    objects = objects if isinstance(objects, list) else [objects]
    summary = f"{len(objects)} objects, mostly open bugs touching the same component."
    return {"summary": summary, "message": f"Summarized {len(objects)} objects."}


def add_work_items_to_sprint(work_ids: list[str], sprint_id: str) -> dict[str, Any]:
    work_ids = work_ids if isinstance(work_ids, list) else [work_ids]
    return {"message": f"Added {len(work_ids)} work items to {sprint_id}."}


def create_actionable_tasks_from_text(text: str) -> dict[str, Any]:
    ids = [f"{ORG}:task/{num}" for num in _numbers(text, _count(text))]
    return {"work_ids": ids, "message": f"Created {len(ids)} tasks from the text."}


HANDLERS = {
    "who_am_i": who_am_i,
    "get_sprint_id": get_sprint_id,
    "work_list": work_list,
    "search_object_by_name": search_object_by_name,
    "get_similar_work_items": get_similar_work_items,
    "prioritize_objects": prioritize_objects,
    "summarize_objects": summarize_objects,
    "add_work_items_to_sprint": add_work_items_to_sprint,
    "create_actionable_tasks_from_text": create_actionable_tasks_from_text,
}
