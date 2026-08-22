"""Optional values `rag-local-eval-loop` reads if present.

Every name here is optional to the suite — a missing one falls back to a suite default. They are
provided so the report describes this system accurately rather than generically.
"""

from __future__ import annotations

# Retrieval latency budget shown in the suite's report.
#
# 50 ms is the suite's own default. Ours is stated as 60 ms because that is the SLO this project
# actually published and measured against: "retrieval latency = request received → context
# assembled, target P100 < 60 ms" (see README). Reporting our real budget rather than inheriting a
# default keeps the number in the suite's report comparable to the one in our own.
LATENCY_BUDGET_MS = 60

# Cosmetic label in the report. Tier 1 is extractive — the answer is copied verbatim from a
# retrieved passage, so there is no generation model in the usual sense, and saying so is more
# accurate than naming an LLM that did not produce this text.
GENERATION_MODEL = "shruti-tier1-extractive (no LLM; verbatim from retrieved passage)"

# Not "local": nothing here holds a shared GPU, so the suite need not clamp workers to 1.
# Tier 1 is pure CPU arithmetic over resident memory and is safe to call concurrently.
GENERATION_BACKEND = "extractive"
