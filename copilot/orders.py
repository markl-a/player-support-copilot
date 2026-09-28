"""Stand-in for the payments lookup (in production: the reconciliation tables from game-ops-automation)."""
from __future__ import annotations

import hashlib


def lookup(order_id: str) -> dict:
    """Deterministic fake: a stable status per order id so tests and demos are repeatable."""
    h = int(hashlib.sha256(order_id.encode()).hexdigest(), 16) % 10
    if h < 5:
        return {"order_id": order_id, "paid": True, "granted": True}
    if h < 8:
        return {"order_id": order_id, "paid": True, "granted": False}
    return {"order_id": order_id, "paid": False, "granted": False}
