"""jevner 本地单测：不依赖网络的算法组件验证。

覆盖随包分词器离线加载、token 统计、长文本滑动窗口切分、词元
偏移一致性、边界标点剥离、实体包解析、多类型 Entity 语义与
跨窗口合并（含嵌套）逻辑。全部为纯本地确定性计算，不发起
任何 API 请求。
"""

from __future__ import annotations

import pytest

from jevner.entities import (
    BIOMED,
    CORE,
    ENTITY_PACKS,
    TECH,
    Entity,
    EntityType,
    resolve_entity_types,
)
from jevner.ner import (
    DEFAULT_WINDOW_SIZE,
    QUESTION_CONTEXT_LIMIT,
    QUESTION_HEADROOM,
    JevNER,
)
from jevner.tokenizer import BundledTokenizer, strip_edge_punctuation

# 中文（无空格）+ 泰文（无空格复杂 cluster）+ 阿拉伯文（RTL）
ZH_TEXT = "你好, 我的名字是吴子豪, 来自福建福州."
TH_TEXT = "สวัสดีครับผมชื่อสมชายมาจากกรุงเทพมหานคร"
AR_TEXT = "مرحبا اسمي محمد وأعمل في شركة أرامكو في الرياض"

tokenizer = BundledTokenizer()
print("[1] 分词器离线加载: OK")

counts = {text: tokenizer.count_tokens(text) for text in (ZH_TEXT, TH_TEXT, AR_TEXT)}
assert all(n > 0 for n in counts.values()), "token 计数必须为正"
print(f"[2] token 计数: 中文={counts[ZH_TEXT]}, 泰文={counts[TH_TEXT]}, 阿拉伯文={counts[AR_TEXT]}")

# 词元偏移一致性：text[start:end] 必须还原词元文本
for text in (ZH_TEXT, TH_TEXT, AR_TEXT):
    for start, end, tok in tokenizer.window_tokens(text):
        assert text[start:end] == tok, f"词元偏移不一致: {tok!r} vs {text[start:end]!r}"
print("[3] 词元偏移一致性: OK")

# 短文本：整段单窗口；默认窗口为 jev per-question 限制减余量
assert DEFAULT_WINDOW_SIZE == QUESTION_CONTEXT_LIMIT - QUESTION_HEADROOM
assert QUESTION_CONTEXT_LIMIT == 32_768 and QUESTION_HEADROOM == 8_192
windows = tokenizer.window_texts(ZH_TEXT, window_size=128, stride=64)
assert windows == [ZH_TEXT], f"短文本应为单一整段窗口, got {windows}"
print(f"[4] 短文本单窗口 + 24k 默认窗口常量 ({DEFAULT_WINDOW_SIZE} token): OK")

# 长文本：多窗口 + 重叠 + 关键实体覆盖
long_zh = "。".join(f"第{i}句话提到北京和上海以及腾讯公司" for i in range(80))
windows_zh = tokenizer.window_texts(long_zh, window_size=64, stride=32)
assert len(windows_zh) > 1, "长文本应切出多个窗口"
covered = "".join(windows_zh)
assert "北京" in covered and "上海" in covered and "腾讯" in covered, "关键实体应出现在窗口中"
print(f"[5] 长文本滑窗: {len(windows_zh)} 个窗口 (window=64, stride=32), 关键实体覆盖: OK")

# 边界标点剥离：标点/空白剥离，货币符号（非 P 类）保留
assert strip_edge_punctuation("子豪,") == "子豪"
assert strip_edge_punctuation(" 美国 ") == "美国"
assert strip_edge_punctuation("。北京。") == "北京"
assert strip_edge_punctuation("$100!") == "$100", "货币符号 $ 属于 Sc 类, 不应剥离"
assert strip_edge_punctuation(",,") == ""
print("[6] 边界标点剥离: OK")

# 实体包解析：默认包 / 名称字符串 / 对象 / 显式类型叠加
default_types = resolve_entity_types(entity_packs=None, entity_types=None)
assert default_types == set(CORE.entity_types), "默认应启用 CORE 基础包"
assert len(default_types) == 19, f"CORE 应含 19 类, got {len(default_types)}"
mixed = resolve_entity_types(entity_packs=["BIOMED", TECH], entity_types={EntityType.PERSON})
assert mixed == set(BIOMED.entity_types) | set(TECH.entity_types) | {EntityType.PERSON}
assert ENTITY_PACKS["CORE"] is CORE and ENTITY_PACKS["FINANCE"].name == "FINANCE"
with pytest.raises(KeyError):
    resolve_entity_types(entity_packs=["NOT_EXIST"], entity_types=None)
with pytest.raises(ValueError):
    resolve_entity_types(entity_packs=[], entity_types=[])
print("[7] 实体包解析: OK")

# 多类型 Entity 语义：types 降序 + type/base_probability 属性兼容单类型访问
multi = Entity(
    text="日本",
    types=((EntityType.GPE, 0.9), (EntityType.LOC, 0.7), (EntityType.ORG, 0.05)),
)
assert multi.type is EntityType.GPE, "主类型应为 types 首项"
assert multi.base_probability == 0.9, "主概率应为 types 首项概率"
probs = [p for _, p in multi.types]
assert probs == sorted(probs, reverse=True), "types 必须按概率降序"
print(f"[8] 多类型 Entity 语义: {multi.type.value}/{[t.value for t, _ in multi.types]}: OK")

# 跨窗口合并：同文本类型并集 + 嵌套判定（不触发 API）
combiner = JevNER.__new__(JevNER)  # 仅复用 _combine 纯函数，不初始化客户端
ents = [
    # 窗口 A: "中华人民共和国外交部" ORG；窗口 B: 同文本 + GPE 次要类型
    Entity(text="中华人民共和国外交部", types=((EntityType.ORG, 0.95),)),
    Entity(text="中华人民共和国外交部", types=((EntityType.ORG, 0.90), (EntityType.GPE, 0.30))),
    # 内层嵌套 GPE "中华人民共和国"
    Entity(text="中华人民共和国", types=((EntityType.GPE, 1.00),)),
    # 窗口边界碎片（主类型与外层实体相同 → 被吸收）
    Entity(text="共和国外交部", types=((EntityType.ORG, 0.60),)),
    # 独立实体
    Entity(text="福建", types=((EntityType.GPE, 1.00),)),
    Entity(text="福州", types=((EntityType.GPE, 0.99),)),
]
merged = combiner._combine(ents, deduplicate=True)
by_text = {e.text: e for e in merged}
# 同文本合并为类型并集，同类型取最高概率
org_ent = by_text["中华人民共和国外交部"]
org_types = {t for t, _ in org_ent.types}
assert EntityType.ORG in org_types and EntityType.GPE in org_types, (
    f"跨窗口同文本应合并类型并集: {org_ent}"
)
org_probs = {t: p for t, p in org_ent.types}
assert org_probs[EntityType.ORG] == 0.95, f"同类型应取最高概率: {org_ent}"
# 嵌套实体：主类型不同（ORG 外层内嵌 GPE 内层）两侧保留
assert "中华人民共和国" in by_text, f"嵌套内层 GPE 应保留: {merged}"
assert "共和国外交部" not in by_text, f"主类型相同的包含碎片应被吸收: {merged}"
assert "福建" in by_text and "福州" in by_text
summary = sorted((e.text, [t.value for t, _ in e.types]) for e in merged)
print(f"[9] 跨窗口合并 + 嵌套 + 多类型并集: {summary}")

print()
print("全部本地单测通过。")
