"""实体类型体系、垂直领域实体包、识别结果与嵌套关联结构定义。

本模块定义四部分内容：

1. EntityType 枚举：通用命名实体类型（沿用 NER 发展至今最全面的
   OntoNotes 5.0 十八类体系，另加 MISC 兜底）+ 各垂直领域的细分类型
   （生物医学、金融、软件科技）；
2. EntityPack 实体包：基础包 CORE + 垂直领域包，调用方按需选择，
   类型表只下发所选包的并集，控制请求体积与判定精度；
3. Entity/Person 结果 dataclass 与 Gender 枚举：Entity 携带原文
   全局偏移 start/end（保证 text == source[start:end]）；
4. EntityNode/entity_tree/format_entity_tree：依据偏移包含关系把
   平铺实体重建为嵌套树并渲染，嵌套关联（如 GPE"福建"内嵌于
   GPE"福建福州"）由偏移锚定，可靠且可读。

调用方与维护者进入本文件时最需要先知道：
- 默认只启用 CORE 基础包；垂直领域通过 entity_packs 参数显式叠加，
  或用 entity_types 直接指定任意类型子集；
- 枚举成员的字符串值即发送给 jev 的 Choice criteria 标签名
  （全大写英文，语言无关）；
- Entity 支持多类型：一个实体可同时具有多种类型（如国家既是政区
  GPE 又是地点 LOC），types 按相关性（概率）降序排列；
- 嵌套关联用 entity_tree() 派生而非内嵌 parent/children 引用：
  偏移包含是派生事实，跨窗口合并后按输出重建，纯函数可测试。
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

    start/end 是原文全局字符偏移（end 不含），保证不变式
    text == source[start:end]——实体嵌套关联（entity_tree）由偏移
    包含关系判定，比文本包含判断可靠：同一实体多次出现、空白
    规范化差异都不会误判归属。

    Attributes:
        text: 实体文本（原文的精确子串）。
        types: (实体类型, 概率) 元组序列，按概率降序排列，至少一项。
        start: 原文起点偏移（含）。
        end: 原文终点偏移（不含）。
    """

    text: str
    types: tuple[tuple[EntityType, float], ...]
    start: int
    end: int

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


@dataclass(frozen=True)
class EntityNode:
    """实体嵌套树节点：一个实体与其全部内层子实体。

    由 entity_tree() 依据偏移包含关系派生；children 为偏移完全
    落在本节点区间内的实体（交叉嵌套的实体互不包含，挂到最近的
    公共外层下成为兄弟）。

    Attributes:
        entity: 本节点实体。
        children: 内层子实体节点，按 (start, -end, text) 排序。
    """

    entity: Entity
    children: tuple[EntityNode, ...] = ()


def entity_tree(entities: list[Entity]) -> list[EntityNode]:
    """依据原文偏移包含关系把平铺实体重建为嵌套树。

    包含关系按偏移判定：outer 包含 inner 当且仅当
    outer.start <= inner.start 且 inner.end <= outer.end。排序后
    用栈一次扫描建树：按 (start, -end, text) 升序遍历，栈顶即
    最近一个未被闭合的外层实体。

    嵌套实体不是严格的树：交叉嵌套（互不包含的交叠区间，如
    GPE(0,10) 与 PERSON(2,5)、FAC(4,8)）会被挂到最近的公共外层
    下成为兄弟节点，而不是强行归入某一支。

    Args:
        entities: 平铺实体列表（允许同文本多处出现）。

    Returns:
        顶层（不被任何实体包含）节点列表，按 (start, -end) 排序。
    """
    ordered = sorted(entities, key=lambda e: (e.start, -e.end, e.text))
    parent: list[int] = [-1] * len(ordered)
    stack: list[int] = []
    for i, e in enumerate(ordered):
        while stack:
            outer = ordered[stack[-1]]
            if outer.start <= e.start and e.end <= outer.end:
                break
            stack.pop()
        if stack:
            parent[i] = stack[-1]
        stack.append(i)
    children: dict[int, list[int]] = {}
    for i in range(len(ordered)):
        if parent[i] >= 0:
            children.setdefault(parent[i], []).append(i)

    def build(i: int) -> EntityNode:
        return EntityNode(entity=ordered[i], children=tuple(build(j) for j in children.get(i, [])))

    return [build(i) for i in range(len(ordered)) if parent[i] < 0]


def format_entity_tree(roots: list[EntityNode]) -> str:
    """把实体嵌套树渲染为可读的缩进文本。

    每行一个实体：文本、全部类型（按概率降序）、可选性别、
    原文偏移区间。子实体用框线缩进挂在外层实体之下，嵌套关联
    一目了然，例如：

    ```
    福建福州 (GPE 0.99) [16:20]
    ├── 福建 (GPE 1.00) [16:18]
    └── 福州 (GPE 1.00) [18:20]
    ```

    Args:
        roots: entity_tree() 返回的顶层节点列表。

    Returns:
        渲染文本；roots 为空时为空字符串。
    """

    def label(e: Entity) -> str:
        types = " · ".join(f"{t.value} {p:.2f}" for t, p in e.types)
        if isinstance(e, Person):
            head = f"{e.text} ({types} | {e.gender.value} {e.gender_probability:.2f})"
            return f"{head} [{e.start}:{e.end}]"
        return f"{e.text} ({types}) [{e.start}:{e.end}]"

    lines: list[str] = []

    def walk(node: EntityNode, prefix: str) -> None:
        count = len(node.children)
        for j, child in enumerate(node.children):
            last = j == count - 1
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{label(child.entity)}")
            walk(child, prefix + ("    " if last else "│   "))

    for root in roots:
        lines.append(label(root.entity))
        walk(root, "")
    return "\n".join(lines)
