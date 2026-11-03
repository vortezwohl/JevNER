"""jevner：无本地模型的多语言命名实体识别 SDK。

以 TypeSafe System One 模型 jev（"超级 BERT"，默认别名 jev-latest
自动路由最新稳定版）替代全部本地 BERT 推理，参考 DeepLoTX BertNER
的长文本滑动窗口算法，随包分发 Qwen3 分词器。

实体类型体系：基础实体包 CORE（OntoNotes 5.0 十八类 + MISC 兜底）
默认启用，垂直领域包 BIOMED（生物医学）、FINANCE（金融）、TECH
（软件科技）按需叠加。

调用方最常用的入口：
- JevNER：识别器实例，调用即返回实体嵌套森林（list[EntityNode]，
  print 直接得到可读树形）；
- EntityType / Gender：类型与性别枚举；
- Entity / Person / EntityNode：结果数据类；
- BIOMED / CORE / FINANCE / TECH：可选实体包。

凭据从环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 读取；
除 TypeSafe API 外没有任何模型下载或本地推理。
"""

from jevner.entities import BIOMED, CORE, Entity, EntityNode, EntityPack, EntityType, Gender, Person
from jevner.ner import JevNER

__version__ = "0.1.0"

__all__ = [
    "BIOMED",
    "CORE",
    "Entity",
    "EntityNode",
    "EntityPack",
    "EntityType",
    "FINANCE",
    "Gender",
    "JevNER",
    "Person",
    "TECH",
]
