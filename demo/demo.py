from jevner import JevNER
from dotenv import load_dotenv

load_dotenv()

ner = JevNER()  # 凭据读取环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL

res = ner("你好, 我的名字是吴子豪, 来自福建福州.")
print(res)
# [Person(text='吴子豪', types=((PERSON, 1.0),), gender=MALE, gender_probability=1.0),
#  Entity(text='福建', types=((GPE, 1.0),))]

res = ner("สวัสดีครับ ผมชื่อสมชาย มาจากกรุงเทพมหานคร")   # 泰文
print(res)
# [Person(text='สมชาย', types=((PERSON, 0.99),), gender=MALE, ...),
#  Entity(text='กรุงเทพ', types=((GPE, 0.83),))]

# 多类型实体：日本既是政区又是地点，types 按相关性降序
res = ner("I visited Japan last year.")
print(res)
# [Entity(text='Japan', types=((GPE, 1.0), (LOC, 0.35), ...)), ...]

# 垂直领域：生物医学实体包
res = ner("患者确诊为二型糖尿病, 服用二甲双胍...", entity_packs=["BIOMED"])
print(res)
