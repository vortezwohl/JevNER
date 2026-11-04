# Supported Decision Models

JevNER runs on **System One decision models** — models that answer typed questions
([Noul](https://docs.typesafe.ai/go/noul) yes/no, [Choice](https://docs.typesafe.ai/go/choice)
selection, [Score](https://docs.typesafe.ai/go/score) rubric) with calibrated probabilities
in a single forward pass, instead of generating text. Any model registered on the
[EasyJev](https://www.easyjev.online/) gateway works; pick one per recognizer:

```python
from jevner import JevNER

ner = JevNER(model_name="baseten/mammoth-1")   # default
ner = JevNER(model_name="jev-latest")          # TypeSafe flagship alias
ner = JevNER(model_name="cloudflare/clef-omni")
```

Or per raw REST call — see [`examples/curl/ner.sh`](../examples/curl/ner.sh).

## Model catalog

Prices are the EasyJev gateway's input price in USD per million tokens
(snapshot 2026-11-03; **output tokens are billed at $0 on every model** — decision
models emit probabilities, not prose). Live pricing: [EasyJev pricing](https://www.easyjev.online/#/pricing).

| Vendor | Model ID | NER status | Input $/M |
| --- | --- | --- | --- |
| Baseten | `baseten/mammoth-1` | **Default.** Verified across the 24-language sweep | 0.10 |
| TypeSafe | `jev-latest` | Verified. Flagship Jev alias, tracks the newest release | 0.042 |
| TypeSafe | `jev-1.13.0` | Verified. Pinned Jev release | 0.042 |
| TypeSafe | `jev-1.14.0` | Verified. Pinned Jev release | 0.042 |
| Baseten | `baseten/joey-1` | Speaks the same protocol; not individually benchmarked for NER | 0.01 |
| Baseten | `baseten/elephant-1` | Speaks the same protocol; not individually benchmarked for NER | 0.035 |
| Cloudflare | `cloudflare/clef` | Same protocol; upstream truncates input around 2K tokens — prefer small `window_size` | 0.192 |
| Cloudflare | `cloudflare/clef-flash` | Same protocol; input truncates around 2K tokens | 0.038 |
| Cloudflare | `cloudflare/clef-omni` | Same protocol; no truncation caveat observed | 0.15 |
| Nace.AI | `nace-ai/drex-max` | Same protocol; not individually benchmarked for NER | 0.06 |
| Nace.AI | `nace-ai/drex-v2.0` | Same protocol; not individually benchmarked for NER | 0.06 |
| Nace.AI | `nace-ai/drex-v1.5` | Same protocol; not individually benchmarked for NER | 0.04 |
| LiquidAI | `liquid/d1` | Same protocol; the one family whose official language list includes Arabic | 0.02 |
| LiquidAI | `liquid/d1.1` | Same protocol | 0.02 |
| LiquidAI | `liquid/d1-ultra` | Same protocol | 0.042 |
| Microsoft | `microsoft/microsoft-decision-1` | Same protocol; not individually benchmarked for NER | 0.042 |
| Upstage | `upstage/solar-decide` | Same protocol; not individually benchmarked for NER | 0.05 |
| Upstage | `upstage/solar-decide-flash` | Same protocol | 0.05 |
| Inception | `inception/mercury-decide` | Same protocol; not individually benchmarked for NER | 0.02 |
| OpenAI | `openai/gpt-6-luna-decisions` | Same protocol; not individually benchmarked for NER | 0.10 |
| Perplexity | `perplexity/pplx-decider-v1.2` | Same protocol; not individually benchmarked for NER | 0.02 |
| Perplexity | `perplexity/pplx-decider-flash-v1.2` | Same protocol | 0.01 |
| Respan | `respan/span-02` | Same protocol; not individually benchmarked for NER | 0.01 |
| Respan | `respan/span-01` | Same protocol | 0.02 |
| Respan | `respan/span-01-lite` | Same protocol | 0.01 |
| ScaleDown | `scaledown/classify-1` | Same protocol; classification-oriented | 0.04 |
| Together | `togethercomputer/tev1-4b-experimental` | Same protocol; experimental checkpoint | 0.042 |
| Jared Palmer | `jaredpalmer/kev-4b` | Same protocol; the open-source [kev](https://github.com/jaredpalmer/kev) decision models on Qwen | 0.042 |

## Choosing a model

- **Start with `baseten/mammoth-1`** — the default, tuned for long-context judgment and
  verified end to end on the [24-language sweep](https://github.com/vortezwohl/JevNER/blob/main/tests/test_languages.py).
- **Cheapest per token:** `baseten/joey-1`, `respan/span-02`, `respan/span-01-lite`,
  `perplexity/pplx-decider-flash-v1.2` (all ≤ $0.01/M input).
- **Arabic-heavy workloads:** the `liquid/d1` family is the only one whose official
  language list explicitly includes Arabic.
- **Short windows only:** `cloudflare/clef` and `cloudflare/clef-flash` truncate input
  around 2K tokens upstream — pass `JevNER(..., window_size=1500)` or smaller if you use them.
- **Pin for reproducibility:** `jev-1.13.0` / `jev-1.14.0` never change underneath you;
  `jev-latest` follows the newest Jev release.

"Verified" above means the model ran the full detection → typing → gender pipeline in
JevNER's test suite. Everything else speaks the identical System One protocol and is
expected to work; run [`tests/test_e2e.py`](../tests/test_e2e.py) with your chosen
`model_name` before trusting a new model in production.

## Calling a model without the SDK

The gateway endpoint is plain JSON REST — `POST /v1/systemone` with a Bearer key.
Three-line version (full annotated version in [`examples/`](../examples/)):

```bash
curl -s "$TYPESAFE_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: jevner-example/0.1" \
  -d '{"state": {"TEXT": "I love Paris."},
       "model": "baseten/mammoth-1",
       "questions": {"q1": {"type": "noul",
                            "instructions": "Is Paris a named entity?",
                            "criteria": {"true": "a named entity", "false": "not an entity"}}}}'
# -> {"model": "...", "answers": {"q1": {"type": "noul", "noul": 0.99}}, "usage": {...}}
```

The same wire format works from Python (stdlib), TypeScript, Go, or any HTTP client — see
[`examples/`](../examples/). For windowing, caching, taxonomy and
nested-tree assembly, use the Python SDK — that logic is the package.

## Related pages

- [Alternatives: JevNER vs GLiNER vs spaCy vs hosted APIs](alternatives.md)
- [The decision-model ecosystem](ecosystem.md)
- [Performance report](performance-report.md)
