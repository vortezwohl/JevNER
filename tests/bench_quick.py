"""jevner 精简性能实验：单进程最小成本拿核心数据点。

设计原则：一次进程跑完、总调用约 8 次、每次测量单样本。
数据点：E1 短/中/长延迟（串行）、E2 缓存冷/热、E3 并行 1 vs 4
（30 句文本 + 小窗口强制 4 窗）、E4 JSON taxonomy 构建与识别。

运行：PYTHONPATH=. uv run python tests/bench_quick.py
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
    """递归统计森林全部节点数。"""
    total = 0
    stack = list(forest)
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def main() -> None:
    print("== E1 单文本延迟 (max_workers=1) ==")
    ner = JevNER(max_workers=1)
    for name, text in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        tokens = ner._tokenizer.count_tokens(text)
        latency, forest = timed(lambda n=ner, t=text: n(t, with_gender=True))
        print(
            f"  {name:6s}: {latency:7.2f}s | {tokens} tokens "
            f"| {count_nodes(forest)} entities (含嵌套)"
        )

    print("== E2 缓存收益 (MEDIUM 同实例二次调用) ==")
    cold, _ = timed(lambda: ner(MEDIUM, with_gender=False))
    warm, _ = timed(lambda: ner(MEDIUM, with_gender=False))
    print(f"  冷: {cold:6.2f}s | 热(全缓存): {warm:6.3f}s | 加速: {cold / max(warm, 1e-6):.0f}x")
    ner.close()

    print("== E3 窗口级并行 (30 句文本, window=1024 → 4 窗口) ==")
    for workers in (1, 4):
        para = JevNER(max_workers=workers)
        latency, forest = timed(
            lambda n=para: n(LONG, window_size=1024, stride=512, with_gender=False)
        )
        print(f"  workers={workers}: {latency:7.2f}s | {count_nodes(forest)} entities")
        para.close()

    print("== E4 JSON 自定义 taxonomy ==")
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
    print(f"  from_json 构建: {build:.4f}s | 识别: {latency:.2f}s | BRAND 命中: {brands}")
    custom.close()

    print("实验完成。")


if __name__ == "__main__":
    main()
