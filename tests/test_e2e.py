"""jevner end-to-end tests: full pipeline against the live API.

Covers the full call chain: multi-type entities, nesting, vertical
packs, long-text windows, and gender (plus cache-hit checks).
Window parallelism set to 4 (the latency sweet spot from the
performance report; higher concurrency triggers transient gateway
500s). Requires TYPESAFE_API_KEY / TYPESAFE_BASE_URL and bills the
live API. Language coverage lives in tests/test_languages.py.
"""

from __future__ import annotations

from jevner import Gender, JevNER, Person
from jevner.entities import DEFAULT_TAXONOMY

T = DEFAULT_TAXONOMY.entity_types  # label -> EntityType lookup table

ZH_TEXT = "你好, 我的名字是吴子豪, 来自福建福州."

ner = JevNER(max_workers=4)


def show(forest: list) -> None:
    """Print entities line by line (with the nesting tree)."""
    print("\n".join("  " + str(n) for n in forest))


print("===== [1] Chinese short text (CORE + gender) =====")
forest = ner(ZH_TEXT)
show(forest)
assert any(isinstance(n.entity, Person) and n.entity.text == "吴子豪" for n in forest), (
    f"should recognize 吴子豪: {forest}"
)
# nested entities must coexist: 福建/福州 inner, 福建福州 outer
# (collect recursively — inner entities are not top-level)


def flatten(nodes: list) -> list:
    """Recursively collect every node of a forest."""
    out: list = []
    for n in nodes:
        out.append(n)
        out.extend(flatten(list(n.children)))
    return out


all_nodes = flatten(forest)
flat_texts = {n.entity.text for n in all_nodes}
assert {"福建", "福州"} <= flat_texts, (
    f"flat return must keep inner nested entities: {sorted(flat_texts)}"
)
assert "福建福州" in flat_texts, f"flat return must keep the compound entity: {sorted(flat_texts)}"
assert all(not t.startswith("来自") for t in flat_texts if t != "福建福州"), (
    f"entities must not glue verbs: {flat_texts}"
)
# 低置信碎片不混入：所有实体（含内层）的主类型概率必须过默认阈值 0.5
for n in all_nodes:
    assert n.entity.base_probability >= 0.5, f"low-confidence fragment leaked in: {n.entity}"
wuzihao = next(
    n.entity for n in forest if isinstance(n.entity, Person) and n.entity.text == "吴子豪"
)
assert wuzihao.gender in (Gender.MALE, Gender.FEMALE, Gender.UNKNOWN)
print(f"  -> 吴子豪 性别: {wuzihao.gender.value}, 概率: {wuzihao.gender_probability}")
# 福建福州 must contain 福建/福州 as children (nesting rides in the return)
fz = next(n for n in forest if n.entity.text == "福建福州")
inner = {c.entity.text for c in fz.children}
assert inner == {"福建", "福州"}, f"inner of 福建福州 should be 福建/福州: {sorted(inner)}"

print()
print("===== [2] multi-type entities (country as region and place) =====")
forest = ner("I visited Japan last year and loved Tokyo.")
show(forest)
japan = [n.entity for n in forest if n.entity.text in ("Japan", "Tokyo")]
assert japan, f"应识别出 Japan/Tokyo: {forest}"
for e in japan:
    # types 按相关性（概率）降序排列
    probs = [p for _, p in e.types]
    assert probs == sorted(probs, reverse=True), f"types 应按概率降序: {e.types}"
    # 主类型属性与 types 首项一致
    assert e.type is e.types[0][0] and e.base_probability == e.types[0][1]
print("  -> multi-type ranking + primary-type properties: OK")

print()
print("===== [3] nested entities (outer ORG, inner GPE) =====")
nested_text = "Bank of America opened a new branch in New York."
forest = ner(nested_text, with_gender=False)
show(forest)
texts_types = {(n.entity.text, n.entity.type) for n in flatten(forest)}
# 外层机构名必须完整保留
assert ("Bank of America", T["ORG"]) in texts_types, f"nested outer ORG must survive: {forest}"
# inner GPE: at least one of America/New York exists (children included)
nested_inner = [
    ("America", T["GPE"]),
    ("New York", T["GPE"]),
]
assert any(t in texts_types for t in nested_inner), f"nested inner GPE must survive: {forest}"
# offset invariant: entity text must be an exact source substring
for n in forest:
    assert nested_text[n.entity.start : n.entity.end] == n.entity.text, (
        f"offset invariant broken: {n.entity}"
    )
# 返回值自身即含嵌套关系：Bank of America 的内层应包含 America
# (under API variance America may appear top-level or as a child;
#   nesting logic is locked by offline tests, so existence is all we assert)
bank = next((n for n in forest if n.entity.text == "Bank of America"), None)
if bank is not None and bank.children:
    inner = [c.entity.text for c in bank.children]
    print(f"  Bank of America 内层: {inner}")
# Chinese nesting (deterministic logic covered by offline tests)
zh_nested_text = "中华人民共和国外交部发言人耿爽在北京举行记者会"
zh_forest = ner(zh_nested_text, with_gender=False)
for n in flatten(zh_forest):
    assert zh_nested_text[n.entity.start : n.entity.end] == n.entity.text, (
        f"Chinese offset invariant broken: {n.entity}"
    )
assert ("中华人民共和国外交部" in {n.entity.text for n in flatten(zh_forest)}) or (
    T["ORG"] in {n.entity.type for n in flatten(zh_forest)}
), f"Chinese nesting keeps at least the outer org: {zh_forest}"

print()
print("===== [4] vertical pack (BIOMED) =====")
med_text = "患者确诊为二型糖尿病, 服用二甲双胍控制血糖, 并检测了 BRCA1 基因与胰岛素水平."
forest = ner(med_text, entity_packs=["BIOMED"], with_gender=False)
show(forest)
all_types = {t for n in forest for t, _ in n.entity.types}
assert T["DISEASE"] in all_types, f"should recognize the disease: {forest}"
# drug allows token-level variance: 二甲双胍 or truncated forms both pass
med_texts = {n.entity.text for n in forest}
assert any("双胍" in t for t in med_texts), (
    f"should recognize the drug (token-variance forms ok): {forest}"
)
# offset invariant: entity text must be an exact source substring
for n in forest:
    assert med_text[n.entity.start : n.entity.end] == n.entity.text, (
        f"offset invariant broken: {n.entity}"
    )

print()
print("===== [5] long text (default 12k window) =====")
# long text fits one window at the default size
long_text = ". ".join(
    f"In {2019 + i % 3}, Maria Curie visited Paris where she met Doctor Strange at Microsoft, "
    f"and they discussed the treaty of Versailles signed in France."
    for i in range(10)
)
print(f"  text length: {len(long_text)} chars")
forest = ner(long_text, with_gender=False)
show(forest)
texts = {n.entity.text for n in forest}
assert any("Maria" in t or "Curie" in t for t in texts), (
    f"long text should surface a person: {sorted(texts)}"
)
assert any("Microsoft" in t for t in texts), f"long text should surface an org: {sorted(texts)}"

print()
print("===== [6] boundary behavior =====")
assert ner("", with_gender=False) == [], "empty text returns an empty list"
assert ner("   ", with_gender=False) == [], "blank text returns an empty list"
forest = ner(ZH_TEXT, with_gender=False)
assert all(not isinstance(n.entity, Person) for n in forest), (
    "with_gender=False must not produce Person"
)
pack: str = "BIOMED"
ner("苹果公司发布了新产品", entity_packs=[pack], with_gender=False)
# gender cache hit (吴子豪 judged in [1])
assert ner._gender_cache.read("吴子豪") is not None, "cache should hit"

ner.close()
print()
print("All end-to-end tests passed.")
print()
print("All end-to-end tests passed.")
