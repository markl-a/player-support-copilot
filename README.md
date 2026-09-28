# player-support-copilot

**A first-line assistant for an online game's customer-service queue. It classifies each ticket, routes it to the right team, pulls out order numbers, looks up the payment, and drafts a reply that cites policy. A person approves every reply before it goes out.**

This is a portfolio demo from September 2026. The tickets are **synthetic**, written in Traditional Chinese with player slang, typos, emoji, and phone numbers or e-mail addresses pasted in. The policies in `kb/` are **demo policies written for this repo, not any company's actual terms**.

```
ticket ─► extract (order id, member id, table, device) ─► mask PII ─► classify (char n-gram TF-IDF + LR)
                                                                           │ confident        │ unsure
                                                                           ▼                  ▼
                                                   policy search (kb/*.md, BM25) + payment lookup   human triage
                                                                           │                  (optional LLM second opinion,
                                                                           ▼                   masked text, forced enum)
                                                                 draft reply with citations
                                                                           ▼
                                                        guard: no secrets asked, no promises, no PII, has citation
                                                                           ▼
                                                        AWAITING_APPROVAL  (never auto-sent)
```

## Results (`results/eval.json`)

The test set is **worded differently from training**: test tickets use sentence templates the classifier never saw. Randomly splitting paraphrases of the same templates would overstate accuracy.

| | Value |
|---|---|
| Accuracy, all tickets | 0.771 (macro-F1 0.749) |
| **Share drafted automatically** at confidence ≥ 0.35 | **46 %** |
| **Accuracy on that share** | **93 %** |
| Order-number extraction recall | 100 % |
| Tickets → AWAITING_APPROVAL / NEEDS_TRIAGE | 130 / 150 |

Coverage vs accuracy, which is how the threshold was chosen:

| min confidence | 0.25 | 0.30 | **0.35** | 0.45 | 0.55 |
|---|---|---|---|---|---|
| auto-drafted | 79 % | 60 % | **46 %** | 35 % | 19 % |
| accuracy on those | 82 % | 85 % | **93 %** | 97 % | 96 % |

**Reading it honestly**
- A cheap classifier on new wording is only 77 % right overall. It is still useful because it **knows when it's unsure**: the confident half is 93 % right, and the rest goes to a person.
- The weakest class is `other` (recall 0.25): off-topic questions get pulled into refund or event categories. The biggest confusions are other→refund (14) and event→account (10). The fix is more labelled "other" tickets, not a bigger model.
- The optional LLM second opinion (`copilot/llm.py`) runs only on the unsure half. It uses Claude tool use with an enum of categories, so it cannot answer in free text. **I have not measured it here, because no API key was used.** Its code path is tested with a fake client, and before switching it on I'd run it on the unsure tickets and add its accuracy to this table.

## Safety properties (each has a test)

- **Nothing is auto-sent.** The best status is `AWAITING_APPROVAL`.
- **The guard checks every draft**, including LLM-polished ones. It blocks drafts that:
  - ask for a password, OTP, card or bank number;
  - promise a refund, a time or a compensation amount;
  - disclose action taken against another player;
  - contain unmasked contact data;
  - lack a policy citation.
  A test feeds a malicious "polish" that appends 「保證三天內全額退款並補償 500 點」; the result is `BLOCKED_BY_GUARD`.
- **Prompt injection in a ticket goes nowhere.** Drafts are built from templates and facts, and the guard runs last.
- **The LLM only sees masked text.** Order numbers are extracted *before* masking. Card masking uses a Luhn check, so 15-digit App Store order numbers survive while real card numbers don't.
- **A Chinese-text pitfall, fixed with a test:** `\b` does not match between a digit and a CJK character in Python regex, so `T1234那桌` extracted nothing until the patterns switched to explicit look-arounds.

## Run

```bash
pip install -r requirements.txt
python -m copilot.make_tickets     # regenerate data (committed)
python -m copilot.evaluate         # → results/eval.json
pytest -q                          # 12 tests
```

```python
from copilot.pipeline import handle
h = handle("扣款了但點數沒加 請幫我查 GPA.1111-2222-3333-44444 手機0911222333")
h.category, h.team, h.status, h.citations, h.draft
```

## Related

- Payment status in production would come from the reconciliation tables in [game-ops-automation](https://github.com/markl-a/game-ops-automation). Here `orders.lookup` is a deterministic stub.
- [CLAUDE.md](CLAUDE.md) holds the rules a coding agent follows in this repo.

MIT License.
