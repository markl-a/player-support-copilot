"""Checks every outgoing draft - template-made or LLM-polished - before a human even sees it.
A draft that fails is not sent for approval; it is sent back with the reasons."""
from __future__ import annotations

import re

from .fields import CARD, EMAIL, PHONE, luhn_ok

RULES = [
    (re.compile(r"(提供|告訴|給).{0,6}(密碼|驗證碼|OTP|信用卡號|卡號|銀行帳號)", re.I), "asks for a secret or payment credential"),
    (re.compile(r"保證|一定(會|可以)?(退|補|返還)|馬上(退|補)|立即退款|全額退"), "promises an outcome policy does not allow"),
    (re.compile(r"補償.{0,4}\d+|\d+\s*(點|金幣|幣).{0,4}(補償|賠償)"), "promises a specific compensation amount"),
    (re.compile(r"(處分|停權|封鎖).{0,6}(對方|他|該玩家)|對方.{0,4}(已|會)被(停權|封鎖)"), "discloses action taken against another player"),
]


def check(draft: str, citations: list[str]) -> list[str]:
    problems = [why for rx, why in RULES if rx.search(draft)]
    if PHONE.search(draft) or EMAIL.search(draft):
        problems.append("contains unmasked personal contact data")
    if any(luhn_ok(re.sub(r"\D", "", m.group(0))) and len(re.sub(r"\D", "", m.group(0))) in (15, 16) and "訂單" not in draft
           for m in CARD.finditer(draft)):
        problems.append("contains a card-like number")
    if not citations:
        problems.append("no policy citation")
    return problems
