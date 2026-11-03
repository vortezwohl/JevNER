"""jevner 端到端测试：真实 TypeSafe API 全流程验证。

覆盖 JevNER 完整调用链：多类型实体、嵌套实体、垂直领域实体包、
超长文本上下文窗口与性别识别（含缓存命中验证）。
窗口级并行 4 路（性能实验结论的延迟甜点，高并发会触发服务端网关瞬态 500）。
需要环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL，产生真实
API 计费。语种覆盖另见 tests/test_languages.py。
"""

from __future__ import annotations

from jevner import EntityPack, Gender, JevNER, Person
from jevner.entities import DEFAULT_TAXONOMY

T = DEFAULT_TAXONOMY.entity_types  # label → EntityType 查找表

ZH_TEXT = "你好, 我的名字是吴子豪, 来自福建福州."

ner = JevNER(max_workers=4)


def show(forest: list) -> None:
    """逐行打印实体（含嵌套树形）。"""
    print("\n".join("  " + str(n) for n in forest))


print("===== [1] 中文短文本 (默认 CORE 包 + 性别) =====")
forest = ner(ZH_TEXT)
show(forest)
assert any(isinstance(n.entity, Person) and n.entity.text == "吴子豪" for n in forest), (
    f"应识别出人名 吴子豪: {forest}"
)
# 嵌套实体必须共存于标准返回：内层 福建/福州 与外层 福建福州 都在
# （收集需递归 children——内层实体不在顶层）


def flatten(nodes: list) -> list:
    """递归收集森林的全部实体节点。"""
    out: list = []
    for n in nodes:
        out.append(n)
        out.extend(flatten(list(n.children)))
    return out


all_nodes = flatten(forest)
flat_texts = {n.entity.text for n in all_nodes}
assert {"福建", "福州"} <= flat_texts, f"标准返回应保留内层嵌套实体: {sorted(flat_texts)}"
assert "福建福州" in flat_texts, f"标准返回应保留外层复合实体: {sorted(flat_texts)}"
assert all(not t.startswith("来自") for t in flat_texts if t != "福建福州"), (
    f"实体不得粘带动词: {flat_texts}"
)
# 低置信碎片不混入：所有实体（含内层）的主类型概率必须过默认阈值 0.5
for n in all_nodes:
    assert n.entity.base_probability >= 0.5, f"低置信碎片混入: {n.entity}"
wuzihao = next(
    n.entity for n in forest if isinstance(n.entity, Person) and n.entity.text == "吴子豪"
)
assert wuzihao.gender in (Gender.MALE, Gender.FEMALE, Gender.UNKNOWN)
print(f"  -> 吴子豪 性别: {wuzihao.gender.value}, 概率: {wuzihao.gender_probability}")
# 福建福州 的内层必须是 福建/福州（返回值自身即含嵌套关系）
fz = next(n for n in forest if n.entity.text == "福建福州")
inner = {c.entity.text for c in fz.children}
assert inner == {"福建", "福州"}, f"福建福州 的内层应为 福建/福州: {sorted(inner)}"

print()
print("===== [2] 多类型实体 (国家同时是政区与地点) =====")
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
print("  -> 多类型排序与主类型属性: OK")

print()
print("===== [3] 嵌套实体 (外层 ORG 内层 GPE) =====")
nested_text = "Bank of America opened a new branch in New York."
forest = ner(nested_text, with_gender=False)
show(forest)
texts_types = {(n.entity.text, n.entity.type) for n in flatten(forest)}
# 外层机构名必须完整保留
assert ("Bank of America", T["ORG"]) in texts_types, f"应保留嵌套外层 ORG: {forest}"
# 内层嵌套 GPE：America 或 New York 至少一个存在（含 children 层）
nested_inner = [
    ("America", T["GPE"]),
    ("New York", T["GPE"]),
]
assert any(t in texts_types for t in nested_inner), f"应保留嵌套内层 GPE: {forest}"
# 偏移不变式：实体文本必须是原文精确子串
for n in forest:
    assert nested_text[n.entity.start : n.entity.end] == n.entity.text, (
        f"偏移不变式破坏: {n.entity}"
    )
# 返回值自身即含嵌套关系：Bank of America 的内层应包含 America
# （API 方差下 America 可能只作为顶层/children 之一出现，嵌套关系
#   本身由本地单测 entity_tree 锁定，此处校验存在性即可）
bank = next((n for n in forest if n.entity.text == "Bank of America"), None)
if bank is not None and bank.children:
    inner = [c.entity.text for c in bank.children]
    print(f"  Bank of America 内层: {inner}")
# 中文嵌套展示（跨类型嵌套保留的确定性逻辑由本地单测覆盖）
zh_nested_text = "中华人民共和国外交部发言人耿爽在北京举行记者会"
zh_forest = ner(zh_nested_text, with_gender=False)
for n in flatten(zh_forest):
    assert zh_nested_text[n.entity.start : n.entity.end] == n.entity.text, (
        f"中文偏移不变式破坏: {n.entity}"
    )
assert ("中华人民共和国外交部" in {n.entity.text for n in flatten(zh_forest)}) or (
    T["ORG"] in {n.entity.type for n in flatten(zh_forest)}
), f"中文嵌套至少应保留外层机构: {zh_forest}"

print()
print("===== [4] 垂直领域实体包 (BIOMED) =====")
med_text = "患者确诊为二型糖尿病, 服用二甲双胍控制血糖, 并检测了 BRCA1 基因与胰岛素水平."
forest = ner(med_text, entity_packs=[EntityPack.BIOMED], with_gender=False)
show(forest)
all_types = {t for n in forest for t, _ in n.entity.types}
assert T["DISEASE"] in all_types, f"应识别出疾病: {forest}"
# 药物允许词元级检测方差：二甲双胍 或其前缀/截断形态（甲双胍）任一命中即可
med_texts = {n.entity.text for n in forest}
assert any("双胍" in t for t in med_texts), f"应识别出药物（含词元方差形态）: {forest}"
# 偏移不变式：实体文本必须是原文精确子串
for n in forest:
    assert med_text[n.entity.start : n.entity.end] == n.entity.text, f"偏移不变式破坏: {n.entity}"

print()
print("===== [5] 超长文本最大上下文窗口 (默认 24k token 窗口) =====")
# 长文本：默认窗口为 jev per-question 限制减余量，此长度单窗口完成
long_text = ". ".join(
    f"In {2019 + i % 3}, Maria Curie visited Paris where she met Doctor Strange at Microsoft, "
    f"and they discussed the treaty of Versailles signed in France."
    for i in range(10)
)
print(f"  文本长度: {len(long_text)} 字符")
forest = ner(long_text, with_gender=False)
show(forest)
texts = {n.entity.text for n in forest}
assert any("Maria" in t or "Curie" in t for t in texts), f"长文本应识别出人名: {sorted(texts)}"
assert any("Microsoft" in t for t in texts), f"长文本应识别出机构: {sorted(texts)}"

print()
print("===== [6] 边界行为 =====")
assert ner("", with_gender=False) == [], "空文本应返回空列表"
assert ner("   ", with_gender=False) == [], "纯空白应返回空列表"
forest = ner(ZH_TEXT, with_gender=False)
assert all(not isinstance(n.entity, Person) for n in forest), "with_gender=False 时不应有 Person"
pack: EntityPack = EntityPack.BIOMED
ner("苹果公司发布了新产品", entity_packs=[pack], with_gender=False)
# 性别缓存命中（[1] 已判定过吴子豪）
assert ner._gender_cache.read("吴子豪") is not None, "缓存应命中"

ner.close()
print()
print("全部端到端测试通过。")
print()
print("全部端到端测试通过。")
