"""jevner 本地单测：不依赖网络的算法组件验证。

覆盖随包分词器离线加载、token 统计、原文偏移词元窗口、词元偏移
一致性、偏移同步剥离、实体包解析、多类型 Entity 语义、跨窗口
合并（按出现位置）与实体嵌套树（entity_tree/format_entity_tree）。
全部为纯本地确定性计算，不发起任何 API 请求。
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
    Gender,
    Person,
    entity_tree,
    resolve_entity_types,
)
from jevner.ner import DEFAULT_WINDOW_SIZE, QUESTION_CONTEXT_LIMIT, QUESTION_HEADROOM, JevNER
from jevner.tokenizer import BundledTokenizer, strip_edge_punctuation, strip_span

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
wins = tokenizer.windows(ZH_TEXT, window_size=128, stride=64)
assert len(wins) == 1, "短文本应为单一整段窗口"
assert wins[0][0] == ZH_TEXT and len(wins[0][1]) == counts[ZH_TEXT], "窗口文本应为原文"
print(f"[4] 短文本单窗口 + 24k 默认窗口常量 ({DEFAULT_WINDOW_SIZE} token): OK")

# 长文本：多窗口 + 重叠 + 关键实体覆盖；偏移全部锚定原文，
# 窗口文本取原文字符区间（词元 join 重建会因空白交叠产生重复字符）
long_zh = "。".join(f"第{i}句话提到北京和上海以及腾讯公司" for i in range(80))
wins = tokenizer.windows(long_zh, window_size=64, stride=32)
assert len(wins) > 1, "长文本应切出多个窗口"
covered = "".join(tok for _wt, w in wins for _s, _e, tok in w)
assert "北京" in covered and "上海" in covered and "腾讯" in covered, "关键实体应出现在窗口中"
for window_text, w in wins:
    c_start, c_end = w[0][0], w[-1][1]
    assert window_text == long_zh[c_start:c_end], "窗口文本必须取原文字符区间"
    for start, end, tok in w:
        assert long_zh[start:end] == tok, "窗口词元偏移必须锚定原文"
print(f"[5] 长文本滑窗: {len(wins)} 个窗口 (window=64, stride=32), 原文偏移锚定: OK")

# 偏移同步剥离：文本与偏移恒一致（嵌套关联的锚点不变式）
text, s, e = strip_span(ZH_TEXT, 6, 13)  # ", 我的名字是" 区间 → 剥离首尾标点空白
assert ZH_TEXT[s:e] == text, "strip_span 后偏移必须与文本一致"
assert strip_span("子豪,", 0, 3) == ("子豪", 0, 2)
assert strip_span(" 美国 ", 0, 4) == ("美国", 1, 3)
money = strip_span("$100!", 0, 5)
assert money == ("$100", 0, 4), f"货币符号 $ 属 Sc 不剥离、尾部 ! 属 Po 应剥离: {money}"
# 不变式抽查：任意区间剥离后 text == s[start:end] 恒成立
for probe in (",,", "。北京。", "  ", "a，b"):
    t2, s2, e2 = strip_span(probe, 0, len(probe))
    assert probe[s2:e2] == t2, f"strip_span 不变式破坏: {probe!r} -> {(t2, s2, e2)}"
assert strip_edge_punctuation("子豪,") == "子豪"
print("[6] 偏移同步剥离不变式: OK")

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

# 多类型 Entity 语义：types 降序 + type/base_probability 属性兼容
multi = Entity(
    text="日本",
    types=((EntityType.GPE, 0.9), (EntityType.LOC, 0.7), (EntityType.ORG, 0.05)),
    start=0,
    end=2,
)
assert multi.type is EntityType.GPE and multi.base_probability == 0.9
probs = [p for _, p in multi.types]
assert probs == sorted(probs, reverse=True), "types 必须按概率降序"
print(f"[8] 多类型 Entity 语义: {multi.type.value}: OK")

# 跨窗口合并：同位置类型并集 / 不同位置独立保留（不触发 API）
combiner = JevNER.__new__(JevNER)  # 仅复用 _combine 纯函数，不初始化客户端
ents = [
    # 位置 [0,10) "中华人民共和国外交部"：窗口 A ORG；窗口 B ORG + GPE 次要类型
    Entity(text="中华人民共和国外交部", types=((EntityType.ORG, 0.95),), start=0, end=10),
    Entity(
        text="中华人民共和国外交部",
        types=((EntityType.ORG, 0.90), (EntityType.GPE, 0.30)),
        start=0,
        end=10,
    ),
    # 嵌套内层 GPE [0,7)
    Entity(text="中华人民共和国", types=((EntityType.GPE, 1.00),), start=0, end=7),
    # 不同位置的独立实体
    Entity(text="福建", types=((EntityType.GPE, 1.00),), start=16, end=18),
    Entity(text="福州", types=((EntityType.GPE, 0.99),), start=18, end=20),
]
merged = combiner._combine(ents, deduplicate=True)
by_span = {(e.start, e.end): e for e in merged}
# 同位置合并为类型并集，同类型取最高概率
org = by_span[(0, 10)]
org_types = {t for t, _ in org.types}
assert EntityType.ORG in org_types and EntityType.GPE in org_types, f"同位置应合并类型并集: {org}"
org_probs = dict(org.types)
assert org_probs[EntityType.ORG] == 0.95, f"同类型应取最高概率: {org}"
# 交叠/包含不做任何吸收：嵌套内层与独立实体一律保留（关联交由 entity_tree 派生）
assert (0, 7) in by_span, f"嵌套内层 GPE 应保留: {merged}"
assert (16, 18) in by_span and (18, 20) in by_span, f"独立实体应保留: {merged}"
print(f"[9] 跨窗口合并（按位置，不吸收交叠）: {[(e.text, e.start, e.end) for e in merged]}")

# 实体嵌套树：偏移包含关系派生 + 交叉嵌套挂公共外层
nested = [
    Entity(text="中华人民共和国外交部发言人", types=((EntityType.ORG, 0.99),), start=0, end=12),
    Entity(text="中华人民共和国", types=((EntityType.GPE, 1.00),), start=0, end=7),  # 外层内嵌 GPE
    Entity(
        text="耿爽", types=((EntityType.PERSON, 0.90),), start=9, end=11
    ),  # 交叉：在 ORG 内、GPE 外
    Entity(text="北京", types=((EntityType.GPE, 0.95),), start=14, end=16),  # 独立顶层
]
roots = entity_tree(nested)
top_texts = [n.entity.text for n in roots]
assert set(top_texts) == {"中华人民共和国外交部发言人", "北京"}, (
    f"顶层应只有不被包含者: {top_texts}"
)
outer = next(n for n in roots if n.entity.text == "中华人民共和国外交部发言人")
child_texts = [c.entity.text for c in outer.children]
# 嵌套内层 GPE 与交叉 PERSON 都直接挂在外层下（互不包含 → 兄弟）
assert set(child_texts) == {"中华人民共和国", "耿爽"}, f"嵌套/交叉应挂公共外层: {child_texts}"
gpe_child = next(c for c in outer.children if c.entity.text == "中华人民共和国")
assert gpe_child.children == (), "内层 GPE 下应无更深层"
# __str__ 渲染：print(node) 即得可读树形
rendered = str(outer)
assert "├──" in rendered and "└──" in rendered, f"渲染应含树形框线: {rendered}"
print("[10] 实体嵌套树 + 交叉嵌套 + 渲染: OK")
print(rendered)
person = Person(
    text="耿爽",
    types=((EntityType.PERSON, 0.9),),
    start=9,
    end=11,
    gender=Gender.MALE,
    gender_probability=0.8,
)
assert "MALE" in str(entity_tree([person])), "渲染应含性别"
print("[11] Person 树渲染（含性别）: OK")

print()
print("全部本地单测通过。")
