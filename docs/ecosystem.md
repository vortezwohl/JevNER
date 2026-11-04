# The Decision-Model Ecosystem

JevNER is one application of a broader shift: **decision models** (also called judgment
models or System One models) — networks trained to answer typed questions with
calibrated probabilities instead of generating text. This page maps the landscape and
where JevNER sits in it.

## The protocol: System One

A System One call sends a `state` (shared context) plus a batch of typed `questions`,
and gets back a probability per question — never prose:

| Primitive | Question shape | Answer shape |
| --- | --- | --- |
| [Noul](https://docs.typesafe.ai/go/noul) | Yes/no with optional outcome descriptions | `noul: 0.98` — probability of yes |
| [Choice](https://docs.typesafe.ai/go/choice) | Select among named options | `choice`, `confidence`, per-option `probabilities` |
| [Score](https://docs.typesafe.ai/go/score) | Grade against an ordered rubric | Expected score + per-level probabilities |

Because answers are typed and calibrated, they can be thresholded, routed, and audited —
the properties JevNER's pipeline is built on (one Noul per token for detection, one
Choice per span for typing, one Noul per name for gender).

## The players

| Project | What it is | Why it matters to JevNER |
| --- | --- | --- |
| [TypeSafe](https://docs.typesafe.ai/) | The protocol steward; ships the **Jev** model family (`jev-1.13`, `jev-1.14`, `jev-latest`) and the open-source [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) client that JevNER is built on | Defines the wire contract (`POST /v1/systemone`) and the Noul/Choice/Score primitives |
| [EasyJev](https://www.easyjev.online/) | Aggregated gateway: Jev, Decider, Drex, d1, Clef, kev and more from every vendor, behind one endpoint fully compatible with the [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) | The default `base_url` of JevNER; one key, any decision model |
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Non-autoregressive decision engine: typed choice, score, and yes/no decisions over any text in a single forward pass, 100+ languages, per-request checkpoint routing (~35k stars) | The most-watched open decision engine; proves the demand for typed, calibrated decisions. Ports: [laya-mlx](https://github.com/mizorewww/laya-mlx), [laya-coreml](https://github.com/mizorewww/laya-coreml) |
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | Open decision models built on Qwen (~9.8k stars) | `jaredpalmer/kev-4b` is served on the EasyJev gateway — an open-weights decision model you can point JevNER at today |
| [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) | Turns any LLM into a decision model | The bridge layer that could widen the model supply behind the protocol |
| [urchade/GLiNER](https://github.com/urchade/GLiNER) | Lightweight local zero-shot NER encoder | The local-weights counterpart to JevNER — see [alternatives](alternatives.md) |
| [explosion/spaCy](https://github.com/explosion/spaCy) | Industrial NLP pipelines | The trained-pipeline incumbent — see [alternatives](alternatives.md) |

## Where JevNER fits

JevNER is a **reference application of the protocol for information extraction**:

1. It treats NER not as generation but as thousands of tiny typed questions —
   the shape decision models are best at.
2. It shows the calibration payoff: probabilities you can gate a human-review
   queue on, not vibes.
3. It demonstrates open-vocabulary recognition — the type table is data
   (JSON), not weights.

The same recipe generalizes: any classification-heavy text task (PII redaction,
claim verification, content moderation, extraction into custom schemas) can be built
as Noul/Choice/Score batches over windows. JevNER's windowing, byte-budget safety,
three-layer caching, and parallel fan-out are the reusable engineering pattern.

## Try the protocol in one call

```bash
curl -s "$TYPESAFE_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: jevner-example/0.1" \
  -d '{"state": {"TEXT": "I love Paris."},
       "model": "jev-latest",
       "questions": {"q1": {"type": "choice",
                            "instructions": "What kind of entity is Paris?",
                            "criteria": {"GPE": "a city, state, or country",
                                         "PERSON": "a person",
                                         "NONE": "neither"}}}}'
# -> {"answers": {"q1": {"type": "choice", "choice": "GPE", "confidence": 0.97,
#                        "probabilities": {"GPE": 0.97, "PERSON": 0.02, "NONE": 0.01}}}, ...}
```

More runnable versions (TypeScript, Go, annotated shell) live in
[`examples/`](../examples/).

## Related pages

- [Supported decision models](models.md)
- [JevNER vs GLiNER vs spaCy vs hosted APIs](alternatives.md)
- [Performance report](performance-report.md)
