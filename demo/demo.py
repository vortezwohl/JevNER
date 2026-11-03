from pprint import pp
from dotenv import load_dotenv

from jevner import EntityPack, JevNER

load_dotenv()

ner = JevNER()  # 凭据读取环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# 返回值即实体嵌套森林：print 直接得到可读树形
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

# 嵌套关系内含于返回值：福建福州 的内层是 福建/福州
forest = ner("患者确诊为二型糖尿病, 服用二甲双胍控制血糖.", entity_packs=[EntityPack.BIOMED])
pp(forest)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 二甲双胍 (DRUG 1.00) [14:18]
