"""实体类型体系、垂直领域实体包与识别结果的数据结构定义。

本模块定义三部分内容：

1. EntityType 枚举：通用命名实体类型（沿用 NER 发展至今最全面的
   OntoNotes 5.0 十八类体系，另加 MISC 兜底）+ 各垂直领域的细分类型
   （生物医学、金融、软件科技）；
2. EntityPack 实体包：基础包 CORE + 垂直领域包，调用方按需选择，
   类型表只下发所选包的并集，控制请求体积与判定精度；
3. Entity/Person 结果 dataclass 与 Gender 枚举。

调用方与维护者进入本文件时最需要先知道：
- 默认只启用 CORE 基础包；垂直领域通过 entity_packs 参数显式叠加，
  或用 entity_types 直接指定任意类型子集；
- 枚举成员的字符串值即发送给 jev 的 Choice criteria 标签名
  （全大写英文，语言无关）；
- Entity 支持多类型：一个实体可同时具有多种类型（如国家既是政区
  GPE 又是地点 LOC），types 按相关性（概率）降序排列。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    """命名实体类型枚举。

    通用部分采用 OntoNotes 5.0 的十八类标准实体体系（PERSON..CARDINAL），
    追加 MISC 作为其余具名专有名词的兜底类型；垂直领域部分（DISEASE..
    PLATFORM）供对应领域实体包选用。枚举成员的字符串值就是发送给
    jev 的 Choice criteria 标签。
    """

    # --- 通用（OntoNotes 5.0 + MISC） ---
    PERSON = "PERSON"
    NORP = "NORP"
    FAC = "FAC"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATE = "DATE"
    TIME = "TIME"
    PERCENT = "PERCENT"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    MISC = "MISC"
    # --- 生物医学 ---
    DISEASE = "DISEASE"
    DRUG = "DRUG"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    CHEMICAL = "CHEMICAL"
    # --- 金融 ---
    TICKER = "TICKER"
    STOCK_INDEX = "STOCK_INDEX"
    CURRENCY = "CURRENCY"
    # --- 软件科技 ---
    SOFTWARE = "SOFTWARE"
    PROTOCOL = "PROTOCOL"
    PLATFORM = "PLATFORM"


class Gender(StrEnum):
    """人名性别判定结果枚举。UNKNOWN 表示无明显性别倾向。"""

    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"


ENTITY_TYPE_DESCRIPTIONS: dict[EntityType, str] = {
    EntityType.PERSON: "A real or fictional person's name.",
    EntityType.NORP: (
        "A nationality, religious group, or political group "
        "(e.g. Han Chinese, Muslims, Republicans)."
    ),
    EntityType.FAC: ("A built facility: airport, highway, bridge, tunnel, stadium, and the like."),
    EntityType.ORG: (
        "An organization: company, government body, school, hospital, sports team, and the like."
    ),
    EntityType.GPE: (
        "A geo-political entity: a country, city, state, or other administrative region."
    ),
    EntityType.LOC: "A non-administrative location: mountain, river, sea, continent, and the like.",
    EntityType.PRODUCT: ("A physical product: vehicle, weapon, food, device, and the like."),
    EntityType.EVENT: (
        "A named event: war, earthquake, conference, sports tournament, and the like."
    ),
    EntityType.WORK_OF_ART: "A title of a work: book, film, song, game, and the like.",
    EntityType.LAW: "The name of a law or legal document.",
    EntityType.LANGUAGE: "The name of a natural language (e.g. Mandarin, English).",
    EntityType.DATE: (
        "A date or date range, absolute (e.g. October 2023) or relative (e.g. yesterday)."
    ),
    EntityType.TIME: "A time of day smaller than a day (e.g. 9 a.m., 14:30).",
    EntityType.PERCENT: "A percentage (e.g. 50%, thirty percent).",
    EntityType.MONEY: (
        "A monetary amount including the currency (e.g. 100 US dollars, five yuan)."
    ),
    EntityType.QUANTITY: (
        "A non-monetary measurement: weight, distance, volume, temperature, and the like."
    ),
    EntityType.ORDINAL: "An ordinal number (e.g. first, 3rd, primary).",
    EntityType.CARDINAL: "A bare quantity number (e.g. forty-seven, 2.5).",
    EntityType.MISC: "A miscellaneous proper name that fits none of the above types.",
    EntityType.DISEASE: "The name of a disease or syndrome (e.g. diabetes, COVID-19).",
    EntityType.DRUG: "The name of a drug or medication (e.g. aspirin, metformin).",
    EntityType.GENE: "The name of a gene (e.g. BRCA1, TP53).",
    EntityType.PROTEIN: "The name of a protein (e.g. insulin, hemoglobin).",
    EntityType.CHEMICAL: "The name of a chemical substance (e.g. ethanol, glucose).",
    EntityType.TICKER: "A security ticker symbol (e.g. AAPL, 600519.SH).",
    EntityType.STOCK_INDEX: "The name of a market index (e.g. the SSE Composite, the S&P 500).",
    EntityType.CURRENCY: (
        "The name of a currency (e.g. US dollar, euro), as distinct from a monetary amount."
    ),
    EntityType.SOFTWARE: (
        "The name of a software program, library, or framework (e.g. Python, PyTorch)."
    ),
    EntityType.PROTOCOL: (
        "The name of a network or communication protocol (e.g. HTTP, Bluetooth)."
    ),
    EntityType.PLATFORM: ("The name of a platform or online service (e.g. WeChat, AWS, GitHub)."),
}
"""全部实体类型的英文描述，随请求 state 下发给 jev 作为类型表。"""


@dataclass(frozen=True)
class EntityPack:
    """一组可选的实体类型集合（实体包）。

    Attributes:
        name: 包名（大写英文，调用方以此引用）。
        description: 包的适用场景说明。
        entity_types: 包含的实体类型集合。
    """

    name: str
    description: str
    entity_types: frozenset[EntityType]


CORE = EntityPack(
    name="CORE",
    description="基础实体包：OntoNotes 5.0 十八类通用实体 + MISC 兜底，适合一般文本",
    entity_types=frozenset(
        {
            EntityType.PERSON,
            EntityType.NORP,
            EntityType.FAC,
            EntityType.ORG,
            EntityType.GPE,
            EntityType.LOC,
            EntityType.PRODUCT,
            EntityType.EVENT,
            EntityType.WORK_OF_ART,
            EntityType.LAW,
            EntityType.LANGUAGE,
            EntityType.DATE,
            EntityType.TIME,
            EntityType.PERCENT,
            EntityType.MONEY,
            EntityType.QUANTITY,
            EntityType.ORDINAL,
            EntityType.CARDINAL,
            EntityType.MISC,
        }
    ),
)
"""基础实体包：默认启用，覆盖通用 NER 的全部标准类型。"""

BIOMED = EntityPack(
    name="BIOMED",
    description="生物医学实体包：疾病、药物、基因、蛋白质、化学物质",
    entity_types=frozenset(
        {
            EntityType.DISEASE,
            EntityType.DRUG,
            EntityType.GENE,
            EntityType.PROTEIN,
            EntityType.CHEMICAL,
        }
    ),
)
"""生物医学垂直领域实体包。"""

FINANCE = EntityPack(
    name="FINANCE",
    description="金融实体包：证券代码、市场指数、货币名称",
    entity_types=frozenset(
        {
            EntityType.TICKER,
            EntityType.STOCK_INDEX,
            EntityType.CURRENCY,
        }
    ),
)
"""金融垂直领域实体包。"""

TECH = EntityPack(
    name="TECH",
    description="软件科技实体包：软件与库、网络协议、平台服务",
    entity_types=frozenset(
        {
            EntityType.SOFTWARE,
            EntityType.PROTOCOL,
            EntityType.PLATFORM,
        }
    ),
)
"""软件科技垂直领域实体包。"""

ENTITY_PACKS: dict[str, EntityPack] = {pack.name: pack for pack in (CORE, BIOMED, FINANCE, TECH)}
"""全部实体包的名字注册表。"""

DEFAULT_PACKS: tuple[EntityPack, ...] = (CORE,)
"""默认启用的实体包：仅基础包。"""


def resolve_entity_types(
    entity_packs: Iterable[EntityPack | str] | None,
    entity_types: Iterable[EntityType] | None,
) -> set[EntityType]:
    """把实体包选择与显式类型叠加解析为最终类型集合。

    Args:
        entity_packs: 实体包（对象或名称字符串）列表；None 时用默认包。
        entity_types: 额外显式指定的类型，与包并集叠加。

    Returns:
        最终启用的实体类型集合；始终非空。

    Raises:
        KeyError: entity_packs 中的名称字符串未注册。
        ValueError: 解析结果为空集合。
    """
    if entity_packs is None:
        packs: list[EntityPack] = list(DEFAULT_PACKS)
    else:
        packs = [ENTITY_PACKS[pack] if isinstance(pack, str) else pack for pack in entity_packs]
    resolved: set[EntityType] = set()
    for pack in packs:
        resolved.update(pack.entity_types)
    if entity_types is not None:
        resolved.update(entity_types)
    if not resolved:
        raise ValueError("Resolved entity types cannot be empty.")
    return resolved


NONE_LABEL = "NONE"
"""Choice 类型判定问题中的非实体选项标签。"""


@dataclass(frozen=True)
class Entity:
    """一个识别出的命名实体（可同时具有多种类型，按相关性排序）。

    实体的判断始终依赖上下文（完整窗口作为请求 state）；一个
    片段可同时符合多种类型（如一个国家既是政区 GPE 又是地点 LOC），
    types 按相关性（概率）降序排列，不强制单一类型。

    Attributes:
        text: 实体文本（窗口原文的精确子串）。
        types: (实体类型, 概率) 元组序列，按概率降序排列，至少一项。
    """

    text: str
    types: tuple[tuple[EntityType, float], ...]

    @property
    def type(self) -> EntityType:
        """主类型：相关性最高的类型（types 首项）。"""
        return self.types[0][0]

    @property
    def base_probability(self) -> float:
        """主类型的概率（types 首项概率）。"""
        return self.types[0][1]


@dataclass(frozen=True)
class Person(Entity):
    """附加性别识别结果的人物实体（主类型为 PERSON 时升级）。

    Attributes:
        gender: 性别判定结果。
        gender_probability: jev 判定该性别的概率（来自 Choice probabilities）。
    """

    gender: Gender
    gender_probability: float
