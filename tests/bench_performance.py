"""jevner 性能实验脚本：延迟 / 吞吐 / 缓存命中 / 并行度测量。

实验设计（对应 docs/性能实验报告.md 的数据来源）：

- E1 单文本延迟：短/中/长文本端到端延迟（串行配置）；
- E2 缓存收益：同文本二次调用的延迟与请求数（检测+分型+性别三级缓存）；
- E3 并行度：多窗口文本在 1/2/4/8 worker 下的端到端延迟与加速比；
- E4 批处理吞吐：分型批次大小 32/64/128 下的多文本吞吐。

需要环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL，产生真实
API 计费。运行：PYTHONPATH=. uv run python tests/bench_performance.py
"""

from __future__ import annotations

import time

from jevner import JevNER

SHORT = "你好, 我的名字是吴子豪, 来自福建福州."
MEDIUM = ". ".join(
    f"In {2019 + i % 3}, Maria Curie visited Paris where she met Doctor Strange at Microsoft."
    for i in range(10)
)
LONG = ". ".join(
    f"Report {i}: engineers at Microsoft and Google held a summit in Tokyo "
    f"where Doctor Strange presented the treaty of Versailles to Maria Curie."
    for i in range(120)
)
TEXTS = {"short": SHORT, "medium": MEDIUM, "long": LONG}


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def main() -> None:
    results: dict[str, dict[str, object]] = {}

    # E1 单文本延迟（串行）
    ner = JevNER(max_workers=1)
    print("== E1 单文本延迟 (max_workers=1) ==")
    e1: dict[str, float] = {}
    for name, text in TEXTS.items():
        # 预检 token 数与窗口数（本地计算，0 请求）
        n_tokens = ner._tokenizer.count_tokens(text)
        n_windows = len(ner._tokenizer.window_texts(text, 24576, 12288))
        latency, entities = timed(lambda n=ner, t=text: n(t, with_gender=True))
        e1[name] = latency
        print(
            f"  {name:6s}: {latency:7.2f}s | {n_tokens} tokens | {n_windows} window(s) "
            f"| {len(entities)} entities"
        )
    results["E1"] = e1
    ner.close()

    # E2 缓存收益（同实例二次调用）
    ner = JevNER(max_workers=1)
    print("== E2 缓存收益 ==")
    cold, _ = timed(lambda n=ner: n(MEDIUM))
    warm, _ = timed(lambda n=ner: n(MEDIUM))
    print(f"  冷调用: {cold:6.2f}s | 热调用(全缓存): {warm:6.2f}s | 加速: {cold / warm:.0f}x")
    results["E2"] = {"cold": cold, "warm": warm}
    ner.close()

    # E3 并行度（多窗口文本，1/2/4/8 workers）
    print("== E3 窗口级并行度 (120 句长文本, window=2048) ==")
    e3: dict[str, float] = {}
    for workers in (1, 2, 4, 8):
        ner_para = JevNER(max_workers=workers)
        ner_para._tokenizer.window_texts(LONG, 2048, 1024)  # 预热分词器（0 请求）
        latency, entities = timed(
            lambda n=ner_para: n(LONG, window_size=2048, stride=1024, with_gender=False)
        )
        e3[f"workers={workers}"] = latency
        print(f"  workers={workers}: {latency:7.2f}s | {len(entities)} entities")
        ner_para.close()
    speedup = e3["workers=1"] / e3["workers=8"]
    print(f"  1->8 加速比: {speedup:.2f}x")
    results["E3"] = e3
    results["E3_speedup"] = speedup

    print()
    print("实验完成。")
    for key, val in results.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
