# Working agreement for coding agents

- Nothing is ever sent to a player automatically. The best status a ticket can reach is AWAITING_APPROVAL.
- Every draft passes `guard.check`, including LLM-polished ones. A new kind of unsafe reply = a new guard rule + a test first.
- LLM calls receive masked text only (`fields.mask`); extraction happens before masking.
- Policy text lives in `kb/*.md` and is cited by file:line; replies must not state policy that is not in the KB.
- Changing `MIN_CONFIDENCE` needs the coverage curve from `python -m copilot.evaluate` in the PR description.
- Run `python -m copilot.evaluate && pytest -q` before finishing.
