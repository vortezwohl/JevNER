#!/usr/bin/env bash
# JevNER's underlying API, in plain curl.
#
# One POST /v1/systemone request, two typed questions:
#   q_detect  (Noul)  - "is the word 'Paris' part of a named entity?"
#   q_type    (Choice)- "which entity type is the span 'Paris'?"
# A decision model answers with calibrated probabilities, never prose.
#
# Usage:
#   export TYPESAFE_API_KEY=...        # required (EasyJev key)
#   export TYPESAFE_BASE_URL=...       # optional (default: https://api.easyjev.online)
#   ./ner.sh
#
# This is the learning example for the wire protocol. For real NER
# (windowing, caching, taxonomy, nested trees) use the Python SDK:
#   pip install jevner

set -euo pipefail

BASE_URL="${TYPESAFE_BASE_URL:-https://api.easyjev.online}"
API_KEY="${TYPESAFE_API_KEY:?export TYPESAFE_API_KEY first}"
MODEL="${JEVNER_MODEL:-baseten/mammoth-1}"

RESPONSE="$(curl -sS --fail-with-body "$BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: jevner-example/0.1 (+https://github.com/vortezwohl/JevNER)" \
  -d @- <<JSON
{
  "state": { "TEXT": "Maria Curie visited Paris in 1903." },
  "model": "$MODEL",
  "questions": {
    "q_detect": {
      "type": "noul",
      "instructions": {
        "INPUT": { "NAME": "Paris" },
        "TASK": "Is this word part of a named entity in the text?"
      },
      "criteria": { "true": "part of a named entity", "false": "not an entity" }
    },
    "q_type": {
      "type": "choice",
      "instructions": {
        "INPUT": { "NAME": "Paris" },
        "TASK": "Which entity type fits this span best?"
      },
      "criteria": {
        "GPE": "a city, state, or country",
        "PERSON": "a named person",
        "NONE": "none of the above"
      }
    }
  }
}
JSON
)"

echo "$RESPONSE" | python -c '
import json, sys
r = json.load(sys.stdin)
d = r["answers"]["q_detect"]          # {"type": "noul", "noul": 0.99}
t = r["answers"]["q_type"]            # {"type": "choice", "choice": "GPE", ...}
print(f"model      : {r[\"model\"]}")
print(f"detection  : noul={d[\"noul\"]:.2f}   (probability the span is an entity)")
print(f"typing     : {t[\"choice\"]} @ {t[\"confidence\"]:.2f}   {t[\"probabilities\"]}")
'
