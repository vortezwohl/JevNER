from pprint import pp

from dotenv import load_dotenv

from jevner import EntityTaxonomy, JevNER

load_dotenv()

ner = JevNER()  # 凭据读取环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# 返回值即实体嵌套森林：pp 直接得到可读结构
forest = ner("你好, 我的名字是吴子豪, 来自福建福州.")
pp(forest)
# 吴子豪 (PERSON 1.00 | MALE 1.00) [9:12]

forest = ner("สวัสดีครับ ผมชื่อสมชาย มาจากกรุงเทพมหานคร")  # 泰文
pp(forest)
# สมชาย (PERSON 1.00 | MALE 1.00) [17:22]

# 多类型实体：日本既是政区又是地点，types 按相关性降序
forest = ner("I visited Japan last year and loved Tokyo.")
pp(forest)
# Japan (GPE 1.00) [10:15]

# 垂直领域：按名称启用生物医学实体包
forest = ner("患者确诊为二型糖尿病, 服用二甲双胍控制血糖.", entity_packs=["BIOMED"])
pp(forest)
# 二型糖尿病 (DISEASE 1.00) [5:10]

# C 端用户 JSON 自定义：新增 BRAND 类型 + ECOM 包 + 默认包切换
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
