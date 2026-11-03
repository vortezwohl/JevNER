"""实体类型体系：开放式类型/包定义、识别结果与嵌套关联结构。

本模块定义四部分内容：

1. EntityType / EntityPack：开放式值对象——标签 + 描述构成一个实体
   类型，命名组合构成一个实体包。不再使用封闭枚举：全部类型与包
   都可以由用户经 JSON 扩展或覆盖（含 C 端非程序员用户）；
2. EntityTaxonomy：实体类型体系——类型注册表 + 包注册表 + 默认包，
   可从 JSON 构建（from_json），内置体系作为扩展基座（BUILTIN_*）；
3. Entity / Person / EntityNode：识别结果 dataclass——Entity 携带
   原文全局偏移 start/end（保证 text == source[start:end]），
   EntityNode 由 JevNER 直接返回（嵌套森林，__str__ 渲染树形）；
4. Gender：性别判定枚举（MALE/FEMALE/UNKNOWN，系统固定语义）。

调用方与维护者进入本文件时最需要先知道：
- 类型是数据不是代码：EntityType("BRAND", "A brand name.") 即可构造，
  与内置类型平权进入类型表；
- JSON 配置三段式（均可省略）：entity_types（新增/覆盖类型）、
  entity_packs（定义包）、default_packs（默认启用的包名）；
- Entity 支持多类型：types 为 (EntityType, probability) 元组序列，
  按相关性（概率）降序排列。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self


@dataclass(frozen=True)
class EntityType:
    """命名实体类型：一个标签与一段语义描述。

    开放式定义：不依赖封闭枚举，任意 label + description 组合都是
    合法的实体类型。label 是发送给 jev 的 Choice criteria 标签
    （建议全大写英文）；description 随请求 state 的类型表下发，
    直接影响判定精度，应写给模型看。

    Attributes:
        label: 类型标签（如 "PERSON"、"BRAND"）。
        description: 类型语义描述（发给 jev 的类型表条目）。
    """

    label: str
    description: str


@dataclass(frozen=True)
class EntityPack:
    """实体包：一组实体类型的命名组合，调用方按需选择启用。

    Attributes:
        name: 包名（如 "CORE"、"BIOMED"）。
        description: 包的适用场景说明。
        entity_types: 包含的实体类型集合。
    """

    name: str
    description: str
    entity_types: frozenset[EntityType]


class Gender(StrEnum):
    """人名性别判定结果枚举。UNKNOWN 表示无明显性别倾向。"""

    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"


BUILTIN_ENTITY_TYPES: dict[str, EntityType] = {
    label: EntityType(label, description)
    for label, description in {
        # --- 通用（OntoNotes 5.0 十八类 + MISC 兜底） ---
        "PERSON": "A real or fictional person's name.",
        "NORP": "A nationality, religious group, or political group.",
        "FAC": "A built facility: airport, highway, bridge, stadium, and the like.",
        "ORG": "An organization: company, government body, school, hospital, and the like.",
        "GPE": "A geo-political entity: a country, city, state, or other administrative region.",
        "LOC": "A non-administrative location: mountain, river, sea, continent, and the like.",
        "PRODUCT": "A physical product: vehicle, weapon, food, device, and the like.",
        "EVENT": "A named event: war, earthquake, conference, sports tournament, and the like.",
        "WORK_OF_ART": "A title of a work: book, film, song, game, and the like.",
        "LAW": "The name of a law or legal document.",
        "LANGUAGE": "The name of a natural language (e.g. Mandarin, English).",
        "DATE": "A date or date range, absolute (e.g. October 2023) or relative (e.g. yesterday).",
        "TIME": "A time of day smaller than a day (e.g. 9 a.m., 14:30).",
        "PERCENT": "A percentage (e.g. 50%, thirty percent).",
        "MONEY": ("A monetary amount including the currency (e.g. 100 US dollars, five yuan)."),
        "QUANTITY": (
            "A non-monetary measurement: weight, distance, volume, temperature, and the like."
        ),
        "ORDINAL": "An ordinal number (e.g. first, 3rd, primary).",
        "CARDINAL": "A bare quantity number (e.g. forty-seven, 2.5).",
        "MISC": "A miscellaneous proper name that fits none of the above types.",
        # --- 生物医学 ---
        "DISEASE": "The name of a disease or syndrome (e.g. diabetes, COVID-19).",
        "DRUG": "The name of a drug or medication (e.g. aspirin, metformin).",
        "GENE": "The name of a gene (e.g. BRCA1, TP53).",
        "PROTEIN": "The name of a protein (e.g. insulin, hemoglobin).",
        "CHEMICAL": "The name of a chemical substance (e.g. ethanol, glucose).",
        # --- 金融 ---
        "TICKER": "A security ticker symbol (e.g. AAPL, 600519.SH).",
        "STOCK_INDEX": "The name of a market index (e.g. the SSE Composite, the S&P 500).",
        "CURRENCY": "The name of a currency (e.g. US dollar), as distinct from a monetary amount.",
        # --- 软件科技 ---
        "SOFTWARE": "The name of a software program, library, or framework (e.g. Python, PyTorch).",
        "PROTOCOL": "The name of a network or communication protocol (e.g. HTTP, Bluetooth).",
        "PLATFORM": "The name of a platform or online service (e.g. WeChat, AWS, GitHub).",
    }.items()
}
"""内置实体类型注册表：通用 + 四个垂直领域共 31 类，作为自定义扩展的基座。"""


def _builtin_pack(name: str, description: str, labels: list[str]) -> EntityPack:
    """由标签列表构建内置实体包（标签必须在内置类型表中）。"""
    return EntityPack(
        name=name,
        description=description,
        entity_types=frozenset(BUILTIN_ENTITY_TYPES[label] for label in labels),
    )


BUILTIN_ENTITY_PACKS: dict[str, EntityPack] = {
    pack.name: pack
    for pack in (
        _builtin_pack(
            "CORE",
            "Base pack: the 18 OntoNotes 5.0 general types plus a MISC fallback.",
            [
                "PERSON",
                "NORP",
                "FAC",
                "ORG",
                "GPE",
                "LOC",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "LANGUAGE",
                "DATE",
                "TIME",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "ORDINAL",
                "CARDINAL",
                "MISC",
            ],
        ),
        _builtin_pack(
            "BIOMED",
            "Biomedical pack: diseases, drugs, genes, proteins, chemicals.",
            ["DISEASE", "DRUG", "GENE", "PROTEIN", "CHEMICAL"],
        ),
        _builtin_pack(
            "FINANCE",
            "Finance pack: tickers, market indices, currency names.",
            ["TICKER", "STOCK_INDEX", "CURRENCY"],
        ),
        _builtin_pack(
            "TECH",
            "Software pack: programs and libraries, protocols, platforms.",
            ["SOFTWARE", "PROTOCOL", "PLATFORM"],
        ),
    )
}
"""内置实体包注册表：CORE 默认启用，三个垂直领域包按需叠加。"""


@dataclass(frozen=True)
class EntityTaxonomy:
    """实体类型体系：类型注册表、包注册表与默认包选择。

    一份 taxonomy 完整描述"有哪些类型、有哪些包、默认启用哪些包"，
    是 JevNER 的全部类型语义来源。可由 JSON 构建（from_json），
    供非程序员用户经数据文件扩展；代码侧也可直接构造。

    Attributes:
        entity_types: 标签到实体类型的注册表。
        entity_packs: 包名到实体包的注册表。
        default_packs: 默认启用的包名序列（JevNER 调用未显式
            指定 entity_packs 时使用）。
    """

    entity_types: dict[str, EntityType]
    entity_packs: dict[str, EntityPack]
    default_packs: tuple[str, ...]

    @classmethod
    def from_json(cls, config: dict[str, Any] | str, base: Self | None = None) -> Self:
        """从 JSON 配置构建实体类型体系（默认在内置体系上扩展）。

        配置三段式（均可省略）::

            {
              "entity_types": {"BRAND": "A brand name."},        # 新增/覆盖类型
              "entity_packs": {"ECOM": ["PERSON", "BRAND"]},      # 新增/覆盖包
              "default_packs": ["CORE", "ECOM"]                  # 默认包
            }

        - entity_types：label → description 的映射，同 label 覆盖内置描述；
        - entity_packs：包名 → 标签列表（或 {"description": ..., "entity_types": [...]})
          的映射，引用的类型必须在合并后的类型表中存在；
        - default_packs：默认启用的包名列表，缺省为 base 的默认包；
        - config 也可传 JSON 字符串（自动解析）。

        Args:
            config: JSON 配置字典或 JSON 字符串。
            base: 扩展基座，None 时用内置体系（BUILTIN_*）。

        Returns:
            构建完成的实体类型体系。

        Raises:
            TypeError: config 既不是 dict 也不是 JSON 字符串。
            ValueError: 配置结构非法——未知标签引用、空类型描述、
                包内引用了不存在的类型等（错误信息含具体字段路径）。
        """
        if isinstance(config, str):
            config = json.loads(config)
        if not isinstance(config, dict):
            raise TypeError(f"config must be a JSON dict or a JSON string, got {type(config)!r}.")
        if base is None:
            base = cls(
                entity_types=dict(BUILTIN_ENTITY_TYPES),
                entity_packs=dict(BUILTIN_ENTITY_PACKS),
                default_packs=("CORE",),
            )
        entity_types = dict(base.entity_types)
        entity_packs = dict(base.entity_packs)
        # 解析 entity_types：label → description，覆盖或新增
        custom_types = config.get("entity_types") or {}
        if not isinstance(custom_types, dict):
            raise ValueError(f'"entity_types" must be a JSON object, got {type(custom_types)!r}.')
        for label, description in custom_types.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Entity type label must be a non-empty string, got {label!r}.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f'Entity type "{label}" needs a non-empty description.')
            entity_types[label] = EntityType(label=label, description=description)
        # 解析 entity_packs：name → 标签列表（或对象形式带 description）
        custom_packs = config.get("entity_packs") or {}
        if not isinstance(custom_packs, dict):
            raise ValueError(f'"entity_packs" must be a JSON object, got {type(custom_packs)!r}.')
        for name, spec in custom_packs.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Entity pack name must be a non-empty string, got {name!r}.")
            if isinstance(spec, list):
                description, labels = "", spec
            elif isinstance(spec, dict):
                description = spec.get("description", "")
                labels = spec.get("entity_types", [])
                if not isinstance(description, str):
                    raise ValueError(f'Entity pack "{name}" description must be a string.')
                if not isinstance(labels, list):
                    raise ValueError(f'Entity pack "{name}" entity_types must be a list of labels.')
            else:
                raise ValueError(
                    f'Entity pack "{name}" must be a label list or an object, got {type(spec)!r}.'
                )
            types: list[EntityType] = []
            for label in labels:
                if label not in entity_types:
                    raise ValueError(f'Entity pack "{name}" references unknown type "{label}".')
                types.append(entity_types[label])
            entity_packs[name] = EntityPack(
                name=name,
                description=description,
                entity_types=frozenset(types),
            )
        # 解析 default_packs
        default_packs = config.get("default_packs")
        if default_packs is None:
            default_packs = list(base.default_packs)
        if not isinstance(default_packs, list) or not all(
            isinstance(p, str) for p in default_packs
        ):
            raise ValueError('"default_packs" must be a list of pack names.')
        for name in default_packs:
            if name not in entity_packs:
                raise ValueError(f'"default_packs" references unknown pack "{name}".')
        return cls(
            entity_types=entity_types,
            entity_packs=entity_packs,
            default_packs=tuple(default_packs),
        )

    def resolve(
        self,
        entity_packs: Iterable[EntityPack | str] | None,
        entity_types: Iterable[EntityType | str] | None,
    ) -> tuple[EntityType, ...]:
        """把包选择与显式类型叠加解析为最终类型序列（按 label 排序去重）。

        Args:
            entity_packs: 实体包（对象或包名）列表；None 时用默认包。
            entity_types: 额外显式启用的类型（对象或标签），与包并集。

        Returns:
            去重后的类型序列，按 label 排序（保证 state 渲染稳定）。

        Raises:
            ValueError: 引用了体系外未注册的包名/类型标签，或解析结果为空。
        """
        if entity_packs is None:
            packs = [self.entity_packs[name] for name in self.default_packs]
        else:
            packs = []
            for pack in entity_packs:
                packs.append(self._pack_of(pack))
        resolved: dict[str, EntityType] = {}
        for pack in packs:
            for etype in pack.entity_types:
                resolved[etype.label] = etype
        if entity_types is not None:
            for item in entity_types:
                resolved[item.label if isinstance(item, EntityType) else item] = self._type_of(item)
        if not resolved:
            raise ValueError("Resolved entity types cannot be empty.")
        return tuple(resolved[label] for label in sorted(resolved))

    def _pack_of(self, pack: EntityPack | str) -> EntityPack:
        """包名转包对象（对象原样返回，未知名抛 ValueError）。"""
        if isinstance(pack, EntityPack):
            return pack
        if pack not in self.entity_packs:
            raise ValueError(f'Unknown entity pack "{pack}".')
        return self.entity_packs[pack]

    def _type_of(self, item: EntityType | str) -> EntityType:
        """类型标签转类型对象（对象原样返回，未知标签抛 ValueError）。"""
        if isinstance(item, EntityType):
            return item
        if item not in self.entity_types:
            raise ValueError(f'Unknown entity type "{item}".')
        return self.entity_types[item]


DEFAULT_TAXONOMY = EntityTaxonomy(
    entity_types=dict(BUILTIN_ENTITY_TYPES),
    entity_packs=dict(BUILTIN_ENTITY_PACKS),
    default_packs=("CORE",),
)
"""默认实体类型体系：内置类型 + 内置包，CORE 默认启用。"""

PERSON_LABEL = "PERSON"
"""PERSON 类型标签：with_gender 时按此标签升级实体为 Person。"""

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

    由 JevNER 调用的返回值直接给出：__str__ 渲染为缩进树形，
    print 即得可读嵌套结构，无需下游再做后处理。

    Attributes:
        entity: 本节点实体。
        children: 内层子实体节点，按 (start, -end, text) 排序。
    """

    entity: Entity
    children: tuple[EntityNode, ...] = ()

    def __str__(self) -> str:
        """渲染本节点为树根的可读缩进树。

        Returns:
            树形文本，如
            "福建福州 (GPE 1.00) [16:20]\\n├── 福建 (GPE 1.00) [16:18]"。
        """
        return "\n".join(_render_node(self, ""))

    __repr__ = __str__


def _render_node(node: EntityNode, prefix: str) -> list[str]:
    """渲染单节点与其子树的行列表（含框线前缀）。

    自身标签行不加框线；子树的连接线由父级在拼接时统一加：
    每层延续父前缀（最后一个子级用空格、其余用竖线）。

    Args:
        node: 当前渲染节点。
        prefix: 本节点之前累积的框线前缀（自身行不加，子行使用）。

    Returns:
        逐行文本列表：首行为本节点纯标签，
        子行为 prefix + 连接线 + 子节点首行，孙行递归拼接。
    """
    lines = [_label(node.entity)]
    count = len(node.children)
    for j, child in enumerate(node.children):
        last = j == count - 1
        child_prefix = prefix + ("    " if last else "│   ")
        connector = "└── " if last else "├── "
        sub = _render_node(child, child_prefix)
        lines.append(prefix + connector + sub[0])
        lines.extend(child_prefix + line for line in sub[1:])
    return lines


def _label(e: Entity) -> str:
    """单个实体的单行标签：文本、类型（概率降序）、性别与偏移区间。"""
    parts = [e.text, "(" + " · ".join(f"{t.label} {p:.2f}" for t, p in e.types) + ")"]
    if isinstance(e, Person):
        parts.append(f"[{e.gender.value} {e.gender_probability:.2f}]")
    parts.append(f"[{e.start}:{e.end}]")
    return " ".join(parts)


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
