# JevNER vs GLiNER vs spaCy vs Hosted NER APIs

Looking for a zero-shot NER stack? This page compares the four realistic architectures
honestly — including when **not** to pick JevNER.

## The four architectures

| | **JevNER** | [GLiNER](https://github.com/urchade/GLiNER) | [spaCy](https://github.com/explosion/spaCy) | Hosted LLM / cloud NER APIs |
| --- | --- | --- | --- | --- |
| Approach | System One **decision model** behind an API: typed questions, calibrated probabilities, single forward pass | Bidirectional **transformer encoder** (local weights) that scores spans against label text at inference | **Fine-tuned statistical pipeline** per language (local weights) | Generative LLM prompted to extract, or fixed-schema cloud NER |
| Custom entity types | **Open JSON taxonomy at runtime** — no code, no training, no restart | Yes — type labels passed as text at inference | Requires a **training run** with annotated data per new type | LLM APIs: prompt-dependent; cloud NER (Comprehend, etc.): fixed schemas or paid custom training |
| Zero-shot | Yes, by design | Yes | No (out of the box) | LLM APIs: roughly; cloud NER: no |
| **Nested entities** | **Structural** — the return value is the entity forest; children ride in `node.children` | Flat spans; nesting needs custom multi-pass logic on top | Flat spans (`ent` objects); no native nesting | Flat; nesting must be prompted and verified |
| Probabilities | **Calibrated** — from a judgment model; `0.8` means "right 80% of the time", safe for confidence-gated review | Sigmoid span scores, usable but not calibration-trained | Model-specific scores, not calibrated | LLMs: none (or verbalized "confidence", unreliable); cloud NER: scores vary |
| Multi-type entities | Ranked `(EntityType, probability)` sequence per entity | One label per span by default | One label per span | One label per span |
| Local weights / GPU | **None** — bundled tokenizer only; all inference behind one endpoint | PyTorch encoder (hundreds of MB) on CPU/GPU | Small local models, CPU-friendly | None |
| Multilingual | One pipeline, **24 languages verified** (incl. unspaced scripts), bundled Qwen3 tokenizer covering 100+ | Multilingual checkpoints exist; one checkpoint per capability trade-off | One trained pipeline **per language** | Depends on provider |
| Offsets | Guaranteed: `text == source[start:end]` on every node | Reliable | Reliable | LLM APIs: frequently hallucinated or off-by-N; must verify |
| Cost model | Per input token (models from **$0.01/M**, output $0) + caching | Free weights, your compute | Free weights, your compute | Per request or per token; LLM extraction of long docs adds up |
| Latency | Network round trips per window (parallelized; cached repeats in ms) | Milliseconds on GPU after load | Milliseconds on CPU | Network round trips, often larger |

## Side by side: custom types

**JevNER** — a JSON document, editable by non-programmers, applied at runtime:

```python
taxonomy = EntityTaxonomy.from_json("""
{
  "entity_types": {"BRAND": "A brand or trademark name (e.g. Apple, Nike)."},
  "entity_packs": {"ECOM": ["PERSON", "ORG", "BRAND"]},
  "default_packs": ["CORE", "ECOM"]
}
""")
ner = JevNER(taxonomy=taxonomy)
```

**GLiNER** — labels passed at inference, local PyTorch weights:

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_multi-v1")
entities = model.predict_text("Tim Cook announced the new iPhone.", labels=["BRAND", "PERSON"])
```

**spaCy** — a new type means annotated data and a training run:

```python
# annotate a corpus, then:
# python -m spacy train config.cfg --output ./output --paths.train ./train.spacy
```

## When to pick what

**Pick JevNER if you want**

- zero training and zero local weights behind one API;
- calibrated probabilities to route low-confidence entities to human review;
- nested entity trees as the return value, not post-processing;
- a taxonomy that product/ops staff edit as JSON without a deploy;
- one pipeline across 24+ languages, including Chinese, Arabic, Thai, Hindi, Hebrew.

**Pick GLiNER if you want**

- fully offline or air-gapped inference;
- maximum batch throughput per dollar on your own GPU;
- millisecond on-box latency;
- and you're comfortable owning model serving.

**Pick spaCy if you want**

- a mature, CPU-cheap production pipeline with a huge tooling ecosystem;
- entity linking, dependency parse, and NER in one pass;
- and you have annotated data to train on.

**Pick a hosted generative LLM if you want**

- complex extraction instructions that mix NER with relations, normalization, and reasoning in one prompt;
- and you accept verifying offsets and output structure yourself.

## Not for JevNER

- Air-gapped or offline-first deployments — the recognizer needs the gateway.
- Sub-10-ms latency budgets — every uncached window is a network round trip.
- Ultra-high-volume batch where owned GPU hardware is already amortized — a local
  GLiNER checkpoint will be cheaper per document at scale.

## Why JevNER bets on decision models

A generative LLM asked "is this a PERSON?" will answer in prose; a decision model
returns a probability it was trained to calibrate. JevNER's pipeline asks thousands of
tiny typed questions (Noul detection per token, Choice typing per span) and assembles
the answers into an entity forest — no parsing of free-form text, no prompt drift
between calls. The [ecosystem page](ecosystem.md) maps the decision-model landscape
(TypeSafe Jev, laya, kev, AnyJev) behind that bet.

## Related pages

- [Supported decision models](models.md)
- [The decision-model ecosystem](ecosystem.md)
- [Performance report](performance-report.md)
