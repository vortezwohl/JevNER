<div align="center">

<h1>
  <img src="assets/jevner-wordmark.svg" alt="JevNER" width="520" />
</h1>

**No local models. Calibrated probabilities. Nested entities.**

*A typesafe Python SDK for multilingual named entity recognition over System One gateway models, with a bundled Qwen3 tokenizer.*

[![PyPI](https://img.shields.io/pypi/v/jevner?style=flat-square&color=00CFFF&label=PyPI)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-B6FF00?style=flat-square&labelColor=07121F)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

<a href="https://docs.typesafe.ai">
  <img src="https://img.shields.io/badge/TypeSafe-System%20One-00E5FF?style=flat-square&logo=github&logoColor=white" alt="TypeSafe System One" />
</a>
<a href="https://github.com/vortezwohl/DeepLoTX">
  <img src="https://img.shields.io/badge/DeepLoTX-BertNER-7C3AED?style=flat-square&logo=github&logoColor=white" alt="DeepLoTX BertNER" />
</a>
<a href="https://huggingface.co/Qwen/Qwen3-8B">
  <img src="https://img.shields.io/badge/Qwen3-Tokenizer-B6FF00?style=flat-square&logo=huggingface&logoColor=white" alt="Qwen3 tokenizer" />
</a>

**One call. A nested entity forest. Every probability calibrated.**

[Why](#why-jevner) &middot; [Quick start](#quick-start) &middot; [Capabilities](#core-capabilities) &middot; [Patterns](#common-patterns) &middot; [API](#public-api) &middot; [Boundaries](#design-boundaries)

</div>

---

## Why JevNER

Most NER stacks make you choose: a fast local model frozen to one label set, or a hosted API with fixed entity types and no confidence you can act on.

JevNER takes a third path. A System One gateway model acts as the tagger: every question returns typed answers with calibrated probabilities instead of generated text, and the SDK assembles them into a nested entity forest you can print as a tree.

| Where classic NER falls short | What JevNER makes explicit |
| --- | --- |
| Label sets are frozen at fine-tuning time; new domains need new training runs. | Entity types and packs are open data: a JSON document defines new types, new packs, and defaults at runtime. |
| Hosted NER APIs return flat spans; nested entities are flattened or dropped. | The return value is a forest: `node.children` holds every entity contained in `node`, derived from offset containment. |
| A softmax score is not a probability you can route on. | Choice probabilities are calibrated: `0.8` behaves like "right 80% of the time", enabling confidence-gated review flows. |
| One span, one type — even when a name is both a place and an organization. | `Entity.types` carries every competitive type with its probability, ranked by relevance. |
| Multilingual coverage means one fine-tuned model per language. | The same pipeline scored 24 languages across Asia, Europe, and the Americas with zero training. |
| Every deployment pulls model weights from a hub. | No local model exists. A bundled Qwen3 tokenizer handles windowing; all inference goes through the System One gateway. |

## Quick Start

```bash
uv add -U jevner        # or: pip install -U jevner
```

```python
from jevner import JevNER

ner = JevNER()  # credentials read from TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# The return value IS the nested forest: printing it renders a tree.
forest = ner("你好, 我的名字是吴子豪, 来自福建福州.")
for node in forest:
    print(node)
# 吴子豪 (PERSON 1.00 | MALE 1.00) [9:12]
# 福建福州 (GPE 1.00) [16:20]
# ├── 福建 (GPE 1.00) [16:18]
# └── 福州 (GPE 1.00) [18:20]
```

That is the complete first loop: no model download, no fine-tuning, no post-processing. PERSON entities are upgraded to `Person` with a gender judgment, offsets always satisfy `text == source[start:end]`, and nesting rides in the return value itself.

> [!TIP]
> The gateway routes several System One models. The default is `baseten/mammoth-1`; switch any time with `JevNER(model_name="jev-latest")`.

## Core Capabilities

| Capability | What you get |
| --- | --- |
| Nested entity forest | `ner(text)` returns `list[EntityNode]`. Offset containment builds the tree; crossing spans attach to the nearest common ancestor as siblings. |
| Calibrated multi-type entities | `Entity.types` is a `(EntityType, probability)` tuple sequence ranked by relevance — one entity can be GPE *and* LOC. |
| Open taxonomy from JSON | `EntityTaxonomy.from_json` accepts a three-section document (`entity_types`, `entity_packs`, `default_packs`) that extends or overrides the 31 built-in types and 4 packs. Built for non-programmer users. |
| Vertical entity packs | `CORE` (18 OntoNotes 5.0 types + MISC) ships by default; `BIOMED`, `FINANCE`, and `TECH` stack on demand. |
| 24-language coverage | Verified end to end across Asia, Europe, and the Americas — Chinese, Thai, Arabic, Hebrew, Hindi, Turkish, English, French, German, Russian, Polish, Brazilian Portuguese, Mexican Spanish, and more. |
| Gender recognition | PERSON entities are upgraded to `Person` with `MALE` / `FEMALE` / `UNKNOWN` and a probability, cached in-process. |
| Bundled Qwen3 tokenizer | Byte-level BPE covering 100+ languages ships inside the package — no HuggingFace access, no `transformers`, no `torch`. |
| Long documents | Sliding windows sized from the gateway's 65,535-byte request budget, 50% overlap, window-level and batch-level thread pools. |
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
    "患者确诊为二型糖尿病, 服用二甲双胍控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 二甲双胍 (DRUG 1.00) [14:18]
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

### Tune the pipeline

```python
forest = ner(
    long_document,
    prob_threshold=0.7,      # stricter primary-type floor (default 0.5)
    with_gender=False,       # skip gender judgments entirely
    window_size=4096,        # smaller windows, more overlap
    max_workers=4,           # window-level concurrency (default 8)
)
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

## Performance

Measured end to end on the live gateway (single sample, cost-bounded; see [docs/performance-report.md](docs/performance-report.md)):

| Workload | Latency | Cached re-run |
| --- | --- | --- |
| Short sentence | ~12 s | ~6 ms |
| 10-sentence paragraph | ~146 s | ~6 ms |
| 30-sentence document, 4 windows in parallel | ~157 s (1.85–5.5× speedup) | ~6 ms |

The pipeline trades latency for correctness: every token is probed, every candidate span is typed with a calibrated Choice, and nested entities are all preserved. For high-frequency streaming workloads, pair JevNER with a local zero-shot tagger and reserve it for low-frequency, high-value extraction.

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