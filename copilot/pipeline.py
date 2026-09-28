"""ticket -> category, route, fields, policy passages, draft reply, guard result.

Nothing is sent to a player automatically: the best case is AWAITING_APPROVAL (an agent clicks send).
Low-confidence tickets go to NEEDS_TRIAGE without a draft - a wrong confident draft costs more than no draft.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import guard, orders
from .classify import default_model, predict
from .fields import extract, mask
from .kb import Index

ROUTES = {  # category -> (team, first-response SLA in hours, policy file)
    "topup_missing": ("金流組", 4, "topup.md"),
    "account_access": ("帳號安全組", 2, "account.md"),
    "disconnect_bug": ("技術組", 8, "disconnect.md"),
    "event_reward": ("營運組", 24, "event.md"),
    "cheating_report": ("風控組", 24, "cheating.md"),
    "refund_request": ("金流組", 24, "refund.md"),
    "other": ("客服一線", 48, None),
}
MIN_CONFIDENCE = 0.35          # from the coverage/accuracy curve in results/eval.json
_ids = itertools.count(1)
_index: Index | None = None


def index() -> Index:
    global _index
    if _index is None:
        _index = Index()
    return _index


@dataclass
class Handled:
    ticket_id: str
    category: str
    confidence: float
    team: str
    sla_hours: int
    fields: dict
    masked_text: str
    citations: list[str] = field(default_factory=list)
    draft: str | None = None
    problems: list[str] = field(default_factory=list)
    status: str = "NEEDS_TRIAGE"
    classified_by: str = "tfidf-lr"


def _draft(cat: str, f: dict, tid: str) -> tuple[str, list[str]]:
    team, sla, pfile = ROUTES[cat]
    passages = index().search({"topup_missing": "付款 入帳 補發 訂單", "account_access": "密碼 驗證 被盜 凍結",
                               "disconnect_bug": "斷線 閃退 牌局", "event_reward": "活動 獎勵 信箱",
                               "cheating_report": "檢舉 牌桌 查核", "refund_request": "退款 平台 申請"}.get(cat, ""), k=2, file=pfile)
    cites = [p.cite for p in passages]
    head = f"您好，這裡是客服中心（案件編號 {tid}）。"
    if cat == "topup_missing":
        if not f["order_ids"]:
            body = "為了幫您查詢，請提供付款收據上的訂單編號（Google Play 為 GPA 開頭）或收據截圖。"
        else:
            st = [orders.lookup(o) for o in f["order_ids"]]
            if all(s["granted"] for s in st):
                body = f"查詢訂單 {', '.join(f['order_ids'])} 顯示點數已入帳，請重新登入後確認；若仍未看到，請回覆我們並附上帳戶畫面截圖。"
            elif any(s["paid"] and not s["granted"] for s in st):
                body = f"已確認訂單 {', '.join(s['order_id'] for s in st if s['paid'] and not s['granted'])} 付款成功但尚未入帳，已轉交{team}補發，完成後會以站內信通知您。"
            else:
                body = "目前查無這筆付款紀錄，請提供付款收據截圖，我們會再協助查詢。"
    elif cat == "account_access":
        body = "請注意：客服不會詢問您的密碼或簡訊驗證碼。若懷疑帳號被盜，我們會先協助凍結帳號，並以您註冊的手機或信箱驗證身分後重設密碼。"
    elif cat == "disconnect_bug":
        body = "牌局中斷線時系統會保留座位 90 秒。為了協助查核，請提供發生時間、裝置型號與系統版本" + (f"（您提到的裝置：{f['device']}）" if f["device"] else "") + "，我們會轉交技術組。"
    elif cat == "event_reward":
        body = "活動獎勵會在活動結束後 3 個工作天內發送至信箱。若確認已達成條件仍未收到，請提供活動名稱與達成時間，我們轉交營運組查核。"
    elif cat == "cheating_report":
        body = "感謝您的檢舉。請提供牌桌編號、時間與對方暱稱" + (f"（您提到 {f['table']}）" if f["table"] else "") + "，風控組會調閱牌局紀錄查核。為保護調查，查核結果不會另行告知。"
    elif cat == "refund_request":
        body = "透過 Google Play 或 App Store 購買的項目，退款需向該平台提出申請，我們無法直接退還平台款項；已使用的點數無法退款。如需申請步驟，我們可以提供說明。"
    else:
        return "", []
    return f"{head}{body}", cites


def handle(text: str, polish: Callable[[str], str] | None = None, model=None,
           second_opinion: Callable[[str], str | None] | None = None) -> Handled:
    tid = f"CS-{next(_ids):06d}"
    f = extract(text)
    masked = mask(text, keep=set(f["order_ids"]))
    cat, conf = predict(model or default_model(), [masked])[0]
    by = "tfidf-lr"
    if conf < MIN_CONFIDENCE and second_opinion:             # e.g. llm.classify - only for the unsure minority
        alt = second_opinion(masked)
        if alt:
            cat, by = alt, "llm"
    team, sla, _ = ROUTES[cat]
    h = Handled(tid, cat, round(conf, 3), team, sla, f, masked, classified_by=by)
    if (conf < MIN_CONFIDENCE and by == "tfidf-lr") or cat == "other":
        return h                                             # human triage, no draft
    draft, cites = _draft(cat, f, tid)
    if polish:                                               # optional LLM tone pass; the guard still applies
        draft = polish(draft)
    h.draft, h.citations = draft, cites
    h.problems = guard.check(draft, cites)
    h.status = "AWAITING_APPROVAL" if not h.problems else "BLOCKED_BY_GUARD"
    return h


def as_dict(h: Handled) -> dict:
    return asdict(h)
