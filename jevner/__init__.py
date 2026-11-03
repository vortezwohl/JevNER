"""jevner：无本地模型的多语言命名实体识别 SDK。

以 TypeSafe System One 模型 jev（"超级 BERT"，默认别名 jev-latest
自动路由最新稳定版）替代全部本地 BERT 推理，参考 DeepLoTX BertNER
的长文本滑动窗口算法，随包分发 Qwen3 分词器。

实体类型体系：基础实体包 CORE（OntoNotes 5.0 十八类 + MISC 兜底）
默认启用，另提供生物医学（BIOMED）、金融（FINANCE）、软件科技
（TECH）垂直领域实体包按需叠加，调用方通过 entity_packs 选择。

调用方与维护者进入本包时最需要先知道：
- 核心入口是 JevNER 实例，调用方式与 BertNER 对齐：
  ner = JevNER(); ner('你好, 我的名字是吴子豪, 来自福建福州.');
- 凭据从环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 读取；
- 除 TypeSafe API 外没有任何模型下载或本地推理。
"""

from jevner.entities import (
    BIOMED,
    CORE,
    DEFAULT_PACKS,
    ENTITY_PACKS,
    FINANCE,
    TECH,
    Entity,
    EntityNode,
    EntityPack,
    EntityType,
    Gender,
    Person,
    entity_tree,
    format_entity_tree,
)
from jevner.ner import (
    DEFAULT_MODEL,
    DEFAULT_PROB_THRESHOLD,
    DEFAULT_STRIDE_DIVISOR,
    DEFAULT_WINDOW_SIZE,
    JevNER,
)
from jevner.tokenizer import BundledTokenizer

__version__ = "0.1.0"

__all__ = [
    "BIOMED",
    "BundledTokenizer",
    "CORE",
    "DEFAULT_MODEL",
    "DEFAULT_PACKS",
    "DEFAULT_PROB_THRESHOLD",
    "DEFAULT_STRIDE_DIVISOR",
    "DEFAULT_WINDOW_SIZE",
    "ENTITY_PACKS",
    "Entity",
    "EntityNode",
    "EntityPack",
    "EntityType",
    "FINANCE",
    "Gender",
    "JevNER",
    "Person",
    "TECH",
    "entity_tree",
    "format_entity_tree",
]
