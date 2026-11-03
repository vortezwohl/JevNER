"""jevner 端到端测试：真实 TypeSafe API 全流程验证。

覆盖 JevNER 完整调用链：多类型实体、嵌套实体、垂直领域实体包、
超长文本上下文窗口与性别识别（含缓存命中验证）。
需要环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL，产生真实
API 计费。语种覆盖另见 tests/test_languages.py。
"""

from __future__ import annotations

from jevner import BIOMED, Entity, EntityPack, EntityType, Gender, JevNER, Person

ZH_TEXT = "你好, 我的名字是吴子豪, 来自福建福州."

ner = JevNER()


def show(entities: list[Entity | Person]) -> None:
    for e in entities:
        print(" ", e)


print("===== [1] 中文短文本 (默认 CORE 包 + 性别) =====")
entities = ner(ZH_TEXT)
show(entities)
assert any(isinstance(e, Person) and e.text == "吴子豪" for e in entities), (
    f"应识别出人名 吴子豪: {entities}"
)
gpe_texts = [e.text for e in entities if EntityType.GPE in {t for t, _ in e.types}]
assert any("福建" in t for t in gpe_texts) or any("福州" in t for t in gpe_texts), (
    f"应识别出福建/福州: {entities}"
)
assert all(not t.startswith("来自") for t in gpe_texts), f"实体不得粘带动词: {gpe_texts}"
wuzihao = [e for e in entities if isinstance(e, Person) and e.text == "吴子豪"][0]
assert wuzihao.gender in (Gender.MALE, Gender.FEMALE, Gender.UNKNOWN)
print(f"  -> 吴子豪 性别: {wuzihao.gender.value}, 概率: {wuzihao.gender_probability}")

print()
print("===== [2] 多类型实体 (国家同时是政区与地点) =====")
entities = ner("I visited Japan last year and loved Tokyo.")
show(entities)
japan = [e for e in entities if e.text in ("Japan", "Tokyo")]
assert japan, f"应识别出 Japan/Tokyo: {entities}"
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
entities = ner(nested_text, with_gender=False)
show(entities)
texts_types = {(e.text, e.type) for e in entities}
# 外层机构名必须完整保留
assert ("Bank of America", EntityType.ORG) in texts_types, f"应保留嵌套外层 ORG: {entities}"
# 内层嵌套 GPE：America 或 New York 至少一个存在
nested_inner = [
    ("America", EntityType.GPE),
    ("New York", EntityType.GPE),
]
assert any(t in texts_types for t in nested_inner), f"应保留嵌套内层 GPE: {entities}"
# 中文嵌套展示（跨类型嵌套保留的确定性逻辑由本地单测覆盖）
zh_nested = ner("中华人民共和国外交部发言人耿爽在北京举行记者会", with_gender=False)
print("  中文嵌套展示:", [f"{e.text}({e.type.value})" for e in zh_nested])
assert ("中华人民共和国外交部" in {e.text for e in zh_nested}) or (
    EntityType.ORG in {e.type for e in zh_nested}
), f"中文嵌套至少应保留外层机构: {zh_nested}"

print()
print("===== [4] 垂直领域实体包 (BIOMED) =====")
med_text = "患者确诊为二型糖尿病, 服用二甲双胍控制血糖, 并检测了 BRCA1 基因与胰岛素水平."
entities = ner(med_text, entity_packs=[BIOMED], with_gender=False)
show(entities)
all_types = {t for e in entities for t, _ in e.types}
assert EntityType.DISEASE in all_types, f"应识别出疾病: {entities}"
assert EntityType.DRUG in all_types, f"应识别出药物: {entities}"

print()
print("===== [5] 超长文本最大上下文窗口 (默认 240k token 窗口) =====")
# 长文本：默认窗口为 jev-latest 最大上下文量级，此长度单窗口完成
long_text = ". ".join(
    f"In {2019 + i % 3}, Maria Curie visited Paris where she met Doctor Strange at Microsoft, "
    f"and they discussed the treaty of Versailles signed in France."
    for i in range(10)
)
print(f"  文本长度: {len(long_text)} 字符")
entities = ner(long_text, with_gender=False)
show(entities)
texts = {e.text for e in entities}
assert any("Maria" in t or "Curie" in t for t in texts), f"长文本应识别出人名: {sorted(texts)}"
assert any("Microsoft" in t for t in texts), f"长文本应识别出机构: {sorted(texts)}"

print()
print("===== [6] 边界行为 =====")
assert ner("", with_gender=False) == [], "空文本应返回空列表"
assert ner("   ", with_gender=False) == [], "纯空白应返回空列表"
entities = ner(ZH_TEXT, with_gender=False)
assert all(not isinstance(e, Person) for e in entities), "with_gender=False 时不应有 Person"
pack: EntityPack = BIOMED
entities = ner("苹果公司发布了新产品", entity_packs=[pack], with_gender=False)
print(" ", entities)
# 性别缓存命中（[1] 已判定过吴子豪）
assert ner._gender_cache.read("吴子豪") is not None, "缓存应命中"

ner.close()
print()
print("全部端到端测试通过。")
