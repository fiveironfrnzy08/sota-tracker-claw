# Restore Artificial Analysis enrichment

One delivery: fix extraction for the current leaderboard, add regression coverage,
validate a live refresh, and deploy the committed fix to mini and sync local data.

The page removed agenticIndex and split selector metadata from rich model rows.
Decode Flight strings as JSON, find rich arrays by intelligence fields, and join
selector metadata by slug. Preserve validation before replacing the runtime cache.

Ticket: ENG-10806.

Parser implemented. Three offline regression tests pass (split chunks, separate
metadata, escaping, legacy 650-row arrays, missing values, malformed input).
Live Playwright validation on September 14 extracted 650 rows, yielding 305
active models and 302 rich rows. Deployment acceptance: mini runs this commit,
refresh succeeds, and the dashboard no longer renders its rich-data warning.
