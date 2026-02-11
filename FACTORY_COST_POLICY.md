# Factory Cost & Safety Policy

Defaults are tuned for **quality first**, then speed.

## Hard caps
- PAGES_PER_RUN: 10
- TITLES_PER_RUN: 50
- MAX_OUTPUT_TOKENS: 1600–1800 (default 1600)
- TEMPERATURE: 1 (Moonshot constraint on your account/model)
- N: 1
- SLEEP_SECONDS: 0.3

## No web research by default
The generator forbids web browsing and external links.

## Self-healing rules
- If a page JSON is invalid: retry up to 2 times.
- If still invalid: skip it (do not fail the whole run).
- Quality gates can auto-delete invalid generated pages when AUTO_DELETE_INVALID=1.

## Scaling
Only increase PAGES_PER_RUN after you have 3 consecutive clean runs with:
- minimal retries
- minimal deletes
