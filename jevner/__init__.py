"""jevner：无本地模型的多语言命名实体识别 SDK。

以 System One 网关模型（默认 baseten/mammoth-1）替代全部本地 BERT
推理，参考 DeepLoTX BertNER 的长文本滑动窗口算法，随包分发 Qwen3
分词器。模型经 model_name 参数随时切换（jev-latest 等网关注册模型
均可）。

实体类型体系：基础实体包 EntityPack.CORE（OntoNotes 5.0 十八类 +
MISC 兜底）默认启用，垂直领域包 EntityPack.BIOMED（生物医学）、
EntityPack.FINANCE（金融）、EntityPack.TECH（软件科技）按需叠加。

调用方最常用的入口：
- JevNER：识别器实例，调用即返回实体嵌套森林（list[EntityNode]，
  print 直接得到可读树形）；
- EntityType / Gender / EntityPack：类型、性别与实体包枚举；
- Entity / Person / EntityNode：结果数据类。

凭据从环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 读取；
除 TypeSafe API 外没有任何模型下载或本地推理。
"""

from jevner.entities import (
    Entity,
    EntityNode,
    EntityPack,
    EntityTaxonomy,
    EntityType,
    Gender,
    Person,
)
from jevner.ner import JevNER

__version__ = "0.1.0"

__all__ = [
    "Entity",
    "EntityNode",
    "EntityPack",
    "EntityType",
    "EntityTaxonomy",
    "Gender",
    "JevNER",
    "Person",
]
