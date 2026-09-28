"""Field extraction (runs on the raw ticket, inside the trusted service) and PII masking (before anything is logged
or sent to an LLM). Order: extract first, then mask - otherwise masking would eat the order number we need."""
from __future__ import annotations

import re

GPA = re.compile(r"GPA\.\d{4}-\d{4}-\d{4}-\d{5}")
APPLE = re.compile(r"(?<!\d)\d{15}(?!\d)")                     # App Store order number printed on the receipt
MEMBER = re.compile(r"會員編號\s*[:：]?\s*(\d{8})")
# no \b: in Python regex CJK characters are word characters, so "T1234那桌" has no boundary after 4
TABLE = re.compile(r"(?<![A-Za-z0-9])T\d{3,5}(?!\d)|第\d+桌|好友房\s*\d+")
DEVICE = re.compile(r"iPhone\s*\d+|iPad|三星\s*\w+|小米\s*\d+|Pixel\s*\d+", re.I)

PHONE = re.compile(r"(?<!\d)09\d{8}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TW_ID = re.compile(r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        s += d
        alt = not alt
    return s % 10 == 0


def extract(text: str) -> dict:
    orders = GPA.findall(text) + APPLE.findall(text)
    m = MEMBER.search(text)
    return {"order_ids": orders, "member_id": m.group(1) if m else None,
            "table": (TABLE.search(text) or [None])[0], "device": (DEVICE.search(text) or [None])[0]}


def mask(text: str, keep: set[str] = frozenset()) -> str:
    """Mask phone, e-mail, national ID and Luhn-valid card numbers. `keep` = already-extracted order ids,
    which can look like card numbers (15 digits) and must survive for the reply."""
    text = PHONE.sub("09********", text)
    text = EMAIL.sub("<email>", text)
    text = TW_ID.sub("<id>", text)

    def card(m):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        return raw if digits in keep or not luhn_ok(digits) else "<card>"

    return CARD.sub(card, text)
