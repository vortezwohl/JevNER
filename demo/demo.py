from pprint import pp

from dotenv import load_dotenv

from jevner import EntityTaxonomy, JevNER

load_dotenv()

ner = JevNER()  # credentials read from TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# The return value is the nested entity forest: pp prints the readable structure.
forest = ner("你好, 我的名字是吴子豪, 来自福建福州.")
pp(forest)
# 吴子豪 (PERSON 1.00 | MALE 1.00) [9:12]

forest = ner("สวัสดีครับ ผมชื่อสมชาย มาจากกรุงเทพมหานคร")  # Thai
pp(forest)
# สมชาย (PERSON 1.00 | MALE 1.00) [17:22]

# Multi-type entities: Japan is both a place and a region, ranked by relevance.
forest = ner("I visited Japan last year and loved Tokyo.")
pp(forest)
# Japan (GPE 1.00) [10:15]

# Vertical domain: enable the biomedical pack by name.
forest = ner("患者确诊为二型糖尿病, 服用二甲双胍控制血糖.", entity_packs=["BIOMED"])
pp(forest)
# 二型糖尿病 (DISEASE 1.00) [5:10]

# Consumer-user JSON customization: add a BRAND type + an ECOM pack + new defaults.
taxonomy = EntityTaxonomy.from_json(
    """
    {
      "entity_types": {"BRAND": "A brand or trademark name (e.g. Apple, Nike)."},
      "entity_packs": {"ECOM": ["PERSON", "ORG", "BRAND"]},
      "default_packs": ["CORE", "ECOM"]
    }
    """
)
ner = JevNER(taxonomy=taxonomy)
forest = ner("Apple released the iPhone 17 Pro, and Nike sponsored it.")
pp(forest)
# EntityNode(entity=Entity(text='Apple', types=((BRAND, 0.9), ...)), ...)
