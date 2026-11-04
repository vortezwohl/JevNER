"""jevner compact performance experiment: core data points in one process.

One process, ~8 API calls total, one sample per measurement.
Data points: E1 short/medium/long latency (serial), E2 cache cold/hot,
E3 parallel 1 vs 4 (30 sentences, small windows -> 4 windows), E4 JSON
taxonomy build + recognition.

Run: PYTHONPATH=. uv run python tests/bench_quick.py
"""

from __future__ import annotations

import time

from jevner import EntityTaxonomy, JevNER

SHORT = "你好, 我的名字是吴子豪, 来自福建福州."
MEDIUM = ". ".join(
    f"In {2019 + i % 3}, Maria Curie visited Paris where she met Doctor Strange at Microsoft."
    for i in range(10)
)
LONG = ". ".join(
    f"Report {i}: engineers at Microsoft and Google held a summit in Tokyo "
    f"where Doctor Strange presented the treaty of Versailles to Maria Curie."
    for i in range(30)
)

CUSTOM_JSON = """
{
  "entity_types": {"BRAND": "A brand or trademark name (e.g. Apple, Nike)."},
  "entity_packs": {"ECOM": ["PERSON", "ORG", "BRAND"]},
  "default_packs": ["CORE", "ECOM"]
}
"""
BRAND_TEXT = (
    "Tim Cook announced that Apple launched the new iPhone while Nike "
    "sponsored the event on Amazon Web Services."
)


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def count_nodes(forest: list) -> int:
    """Count all nodes of a forest recursively."""
    total = 0
    stack = list(forest)
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def main() -> None:
    print("== E1 single-text latency (max_workers=1) ==")
    ner = JevNER(max_workers=1)
    for name, text in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        tokens = ner._tokenizer.count_tokens(text)
        latency, forest = timed(lambda n=ner, t=text: n(t, with_gender=True))
        print(
            f"  {name:6s}: {latency:7.2f}s | {tokens} tokens "
            f"| {count_nodes(forest)} entities (nesting incl.)"
        )

    print("== E2 cache benefit (MEDIUM, same instance) ==")
    cold, _ = timed(lambda: ner(MEDIUM, with_gender=False))
    warm, _ = timed(lambda: ner(MEDIUM, with_gender=False))
    speedup = cold / max(warm, 1e-6)
    print(f"  cold: {cold:6.2f}s | warm (cached): {warm:6.3f}s | speedup: {speedup:.0f}x")
    ner.close()

    print("== E3 window parallelism (30 sentences, window=1024 -> 4 windows) ==")
    for workers in (1, 4):
        para = JevNER(max_workers=workers)
        latency, forest = timed(
            lambda n=para: n(LONG, window_size=1024, stride=512, with_gender=False)
        )
        print(f"  workers={workers}: {latency:7.2f}s | {count_nodes(forest)} entities")
        para.close()

    print("== E4 JSON custom taxonomy ==")
    build, taxonomy = timed(lambda: EntityTaxonomy.from_json(CUSTOM_JSON))
    custom = JevNER(taxonomy=taxonomy, max_workers=1)
    latency, forest = timed(lambda: custom(BRAND_TEXT, with_gender=False))
    texts = set()
    stack = list(forest)
    while stack:
        node = stack.pop()
        texts.add(node.entity.text)
        stack.extend(node.children)
    brands = [t for t in texts if t in ("Apple", "Nike", "Amazon Web Services")]
    print(f"  from_json build: {build:.4f}s | recognize: {latency:.2f}s | BRAND hits: {brands}")
    custom.close()

    print("Experiment complete.")


if __name__ == "__main__":
    main()
