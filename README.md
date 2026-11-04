<div align="center">

<h1>
  <img src="assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**Zero-shot multilingual NER. Powered by System One judgment models.**

*A typesafe Python SDK that turns a judgment model into a named entity recognizer — nested entity trees, custom entity packs, and a bundled Qwen3 tokenizer, with no local models to train or deploy.*

<p>
  <a href="https://www.easyjev.online/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vortezwohl/jev-router-web/main/public/favicon.svg" />
      <img src="https://raw.githubusercontent.com/vortezwohl/jev-router-web/main/public/favicon.svg" alt="EasyJev" width="72" />
    </picture>
  </a>
  <br/>
  <strong>Sponsored by <a href="https://www.easyjev.online/">EasyJev</a> — every decision model, one endpoint.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner?style=flat-square&color=8CE838&label=PyPI)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**No training. Any language. The tree comes back in the return value.**

[Why](#why-jevner) &middot; [Quick start](#quick-start) &middot; [Capabilities](#core-capabilities) &middot; [Patterns](#common-patterns) &middot; [API](#public-api) &middot; [Boundaries](#design-boundaries) &middot; [Sponsor](#sponsor)

</div>

---

## Why JevNER

Most NER stacks make you choose: a local model frozen to one label set, or a hosted API that hides how confident it really is.

JevNER takes a third path. A **System One judgment model** — not a text generator — reads every question and returns **calibrated probabilities** instead of prose, and the SDK assembles those judgments into a **nested entity forest** you can print as a tree.

| Where classic NER falls short | What JevNER does instead |
| --- | --- |
| Label sets are frozen at fine-tuning time; every new domain means a new training run. | **Zero-shot by design.** Entity types are open data: any user (including non-programmers) can add types and packs at runtime through a plain JSON document. |
| Multilingual coverage means one fine-tuned checkpoint per language. | **One pipeline, 24 languages verified** — Chinese, Thai, Arabic, Hebrew, Hindi, Turkish, Indonesian, Filipino, Vietnamese, Japanese, Korean, English, French, German, Spanish, Portuguese, Italian, Russian, Dutch, Polish, and more. Low-resource and unspaced scripts included. |
| Hosted NER APIs return flat spans; nesting is flattened or dropped. | **The return value is the tree.** `node.children` holds every entity contained in `node`, derived from source-text offsets (`text == source[start:end]`), so nesting is structural, not guessed. |
| A softmax score is not a probability you can route on. | **Calibrated probabilities** from the judgment model: `0.8` behaves like "right 80% of the time", which is what confidence-gated review flows actually need. |
| One span gets one type, even when a name is both a place and an organization. | **Multi-type entities, ranked.** `Entity.types` carries every competitive type with its probability, ordered by relevance. |
| Every deployment downloads model weights. | **No local models at all.** A bundled Qwen3 tokenizer (100+ languages) ships inside the package; all inference goes through the EasyJev gateway. |

## Quick Start

```bash
uv add -U jevner        # or: pip install -U jevner
```

```python
from jevner import JevNER

ner = JevNER()  # credentials read from TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# The return value IS the nested forest: printing it renders a tree.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

That is the complete first loop: no model download, no fine-tuning, no post-processing. PERSON entities are upgraded to `Person` with a gender judgment, offsets always satisfy `text == source[start:end]`, and nesting rides in the return value itself.

> [!TIP]
> The gateway routes several System One models. The default is `baseten/mammoth-1`; switch any time with `JevNER(model_name="jev-latest")`.

## Core Capabilities

| Capability | What you get |
| --- | --- |
| Zero-shot recognition | No training, no annotation, no fine-tuning. Define what to extract and start calling. |
| Judgment-model foundation | Noul / Choice questions with calibrated probabilities — yes/no confidence and full option distributions, not free-form generation. |
| Nested entity forest | `ner(text)` returns `list[EntityNode]`. Crossing spans attach to the nearest common ancestor as siblings; nesting is derived from offsets, not substring guesswork. |
| Calibrated multi-type entities | `Entity.types` is a `(EntityType, probability)` tuple sequence ranked by relevance — one entity can be GPE *and* LOC. |
| 24-language coverage | Verified end to end across Asia, Europe, and the Americas, including low-resource and unspaced scripts. |
| Open taxonomy from JSON | `EntityTaxonomy.from_json` accepts a three-section document (`entity_types`, `entity_packs`, `default_packs`) that extends or overrides the 31 built-in types and 4 packs — built for non-programmer users. |
| Vertical entity packs | `CORE` (18 OntoNotes 5.0 types + MISC) ships by default; `BIOMED`, `FINANCE`, and `TECH` stack on demand. |
| Gender recognition | PERSON entities are upgraded to `Person` with `MALE` / `FEMALE` / `UNKNOWN` and a probability, cached in-process. |
| Bundled Qwen3 tokenizer | Byte-level BPE covering 100+ languages ships inside the package — no HuggingFace access, no `transformers`, no `torch`. |
| Request-budget safety | Windows sized from the gateway's 65,535-byte body cap; typing batches narrowed by the 512-option cap — every request stays within limits by construction. |
| Three-layer caching | Detection, typing, and gender results sit in LRU caches keyed by window hash; repeated or overlapping text returns in milliseconds. |
| Typesafe throughout | Full annotations with `py.typed`, mypy-clean under strict settings, prompts rendered through the structured prompt4py templates. |

## Common Patterns

### Print the forest, not a flat list

The return value is already the tree. `str(node)` renders a node and its descendants with box-drawing lines, types, probabilities, gender, and offsets.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### Enable a vertical pack

Packs stack by name; the type table sent to the model contains exactly the union you selected.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### Define a taxonomy from JSON

Any user — including one who never opens a code editor — can extend the recognition vocabulary with a JSON document. Custom types are first-class citizens alongside the built-ins.

```python
from jevner import EntityTaxonomy, JevNER

taxonomy = EntityTaxonomy.from_json("""
{
  "entity_types": {
    "BRAND": "A brand or trademark name (e.g. Apple, Nike)."
  },
  "entity_packs": {
    "ECOM": ["PERSON", "ORG", "BRAND"]
  },
  "default_packs": ["CORE", "ECOM"]
}
""")

ner = JevNER(taxonomy=taxonomy)
forest = ner("Tim Cook announced that Apple launched the new iPhone while Nike sponsored it.")
```

`from_json` accepts a JSON string or an already-parsed dict. Unknown type references, empty descriptions, and unregistered default packs raise `ValueError` with the offending field path.

### Route on calibrated confidence

Every probability comes from a judgment model trained for calibration, so thresholds mean what they say:

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 really means "right 80% of the time"
```

Repeated calls on the same text hit the cache in milliseconds — useful for RAG re-queries and document re-runs.

## Public API

The root package deliberately exposes eight names:

```python
from jevner import (
    Entity,          # a recognized entity: text, types, start, end
    EntityNode,      # tree node: entity + children
    EntityPack,      # named bundle of entity types (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # open type system: registry + from_json + resolve
    EntityType,      # label + description value object
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # the recognizer
    Person,          # Entity + gender + gender_probability
)
```

## Design Boundaries

JevNER owns the recognition pipeline and its contracts. It intentionally does not provide:

- local model inference of any kind;
- model hosting, fine-tuning, or weight distribution;
- training data, annotation tooling, or evaluation harnesses;
- UI components; or
- re-exports of gateway SDK internals.

That boundary keeps the SDK a thin, verifiable pipeline while leaving hosting and product experience to the platform.

## Sponsor

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="https://raw.githubusercontent.com/vortezwohl/jev-router-web/main/public/favicon.svg" alt="EasyJev" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**Every decision model. One API. Judgments in milliseconds.**

EasyJev is the aggregated home of System One decision models — **Jev, Decider, Drex, d1, Clef and more,
from 14 vendors, 28 models on sale today** — behind one `typesafe-sdk`-compatible endpoint. A single
forward pass completes each judgment in milliseconds; output tokens are billed at $0 on every model,
and every call settles token-by-token against upstream usage on an integer-quota ledger
(1,000,000,000 quota = $1, no floating-point drift). Sign in with an email code — your account is
created on first sign-in, new users get $1 in trial credit, and top-ups are one-time USDT/USDC
payments with no subscription. Existing model names (`jev-latest`, `jev-1.14.0`, …) keep working
unchanged: point `base_url` and `api_key` at the gateway and pick any model per request.

[Sign up &rarr;](https://www.easyjev.online/) &middot; [Model Square](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [Pricing](https://www.easyjev.online/#/pricing) &middot; [Terms](https://www.easyjev.online/#/tos) &middot; [Privacy](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## Development

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # style and static checks
uv run mypy jevner                          # type check
uv run python tests/test_local.py           # offline algorithm tests (12 checks)
uv run python tests/test_e2e.py             # end to end (requires TYPESAFE_API_KEY, bills the API)
uv run python tests/test_languages.py       # 24-language sweep (bills the API)
uv run python tests/bench_quick.py          # compact performance experiment (bills the API)
```

## Contributing

Contributions should improve verifiable pipeline behavior: detection recall, typing precision, nesting fidelity, taxonomy validation, cache behavior, or request-budget safety. Keep the public API narrow and add independently runnable tests for new behavior.

## License

[MIT](LICENSE)