"""Entity type system: open type/pack definitions, results, and nesting structures.

This module defines four parts:

1. EntityType / EntityPack — open value objects: a label plus a
   description forms an entity type; a named bundle of types forms an
   entity pack. No closed enum: every type and pack can be extended or
   overridden by any user through JSON (including non-programmer users).
2. EntityTaxonomy — the type system: a type registry, a pack registry,
   and the default pack selection. Buildable from JSON via from_json;
   the built-in system (BUILTIN_*) serves as the extension base.
3. Entity / Person / EntityNode — result dataclasses. Entity carries
   original-text offsets start/end (invariant: text == source[start:end]);
   JevNER returns EntityNode forests directly, and __str__ renders trees.
4. Gender — the gender judgment enum (MALE/FEMALE/UNKNOWN, fixed
   system semantics).

What callers and maintainers need to know first:
- Types are data, not code: EntityType("BRAND", "A brand name.") is a
  fully valid type, on equal footing with the built-ins.
- JSON configuration has three optional sections: entity_types
  (add/override types), entity_packs (define packs), default_packs
  (which packs are on by default).
- Entity supports multiple types: types is a sequence of
  (EntityType, probability) tuples ranked by relevance (probability
  descending).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self, TypeGuard


def _nonempty_strs(seq: object) -> TypeGuard[list[str]]:
    """True when seq is a list of non-empty (non-blank) strings."""
    return isinstance(seq, list) and all(isinstance(item, str) and item.strip() for item in seq)


@dataclass(frozen=True)
class EntityType:
    """A named entity type: a label, a semantic description, and
    optional boundary guides.

    Open definition: no closed enum is involved — any label +
    description pair is a valid entity type. The label is the Choice
    criteria tag sent to the model (uppercase English recommended);
    the description rides in the request state's type table and
    directly affects judgment precision, so write it for the model.
    Examples and counter-examples render into the same table as
    positive ("e.g.") and negative ("not") boundary guides — one
    well-chosen counter-example often fixes a systematically confused
    boundary (external zero-shot NER benchmarks have measured a single
    boundary sentence moving a dataset F1 from 0.778 to 0.945).

    Attributes:
        label: The type label (e.g. "PERSON", "BRAND").
        description: The semantic description (the type-table entry
            sent to the model).
        examples: Positive examples (at most three are rendered).
        counter_examples: Negative boundary guides — things this type
            must NOT match (at most three are rendered).
        min_chars: Minimum entity length in characters; shorter
            predictions are dropped (guards single-character noise in
            unspaced scripts; use 2 for person/place types in Chinese).
        include_brackets: Whether enclosing book-title brackets
            (《》〈〉) belong to entities of this type — WORK_OF_ART
            style labeling conventions.
    """

    label: str
    description: str
    examples: tuple[str, ...] = ()
    counter_examples: tuple[str, ...] = ()
    min_chars: int = 1
    include_brackets: bool = False


@dataclass(frozen=True)
class EntityPack:
    """An entity pack: a named bundle of entity types.

    Attributes:
        name: The pack name (e.g. "CORE", "BIOMED").
        description: The pack's intended-use description.
        entity_types: The set of entity types in the pack.
    """

    name: str
    description: str
    entity_types: frozenset[EntityType]


class Gender(StrEnum):
    """Gender judgment result. UNKNOWN means no clear gender signal."""

    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"


_BUILTIN_SPECS: tuple[dict[str, Any], ...] = (
    # --- General (18 OntoNotes 5.0 types + MISC fallback) ---
    {
        "label": "PERSON",
        "description": "A real or fictional person's name.",
        "examples": ("陈嘉禾", "Marie Curie"),
        "counter_examples": ("职业或称谓如「总经理」", "代词如「他」"),
        "min_chars": 2,
    },
    {
        "label": "NORP",
        "description": "A nationality, religious group, or political group.",
        "examples": ("中国人", "Islam"),
        "counter_examples": ("具体人名（那是 PERSON）",),
    },
    {
        "label": "FAC",
        "description": "A built facility: airport, highway, bridge, stadium, and the like.",
        "examples": ("北京首都机场", "Golden Gate Bridge"),
    },
    {
        "label": "ORG",
        "description": (
            "An organization: company, government body, school, hospital, and the like."
        ),
        "examples": ("阿里巴巴", "联合国"),
        "counter_examples": ("设施名称如「首都机场」（那是 FAC）",),
    },
    {
        "label": "GPE",
        "description": (
            "A geo-political entity: a country, city, state, or other administrative region."
        ),
        "examples": ("江苏南京", "New York"),
        "counter_examples": ("自然地貌如「长江」（那是 LOC）",),
    },
    {
        "label": "LOC",
        "description": (
            "A non-administrative location: mountain, river, sea, continent, and the like."
        ),
        "examples": ("长江", "南极洲"),
        "counter_examples": ("城市或行政区名（那是 GPE）",),
    },
    {
        "label": "PRODUCT",
        "description": "A physical product: vehicle, weapon, food, device, and the like.",
        "examples": ("iPhone 15", "波音747"),
    },
    {
        "label": "EVENT",
        "description": (
            "A named event: war, earthquake, conference, sports tournament, and the like."
        ),
        "examples": ("第二次世界大战", "世界杯"),
    },
    {
        "label": "WORK_OF_ART",
        "description": "A title of a work: book, film, song, game, and the like.",
        "examples": ("红楼梦", "流浪地球"),
        "counter_examples": ("作者名（那是 PERSON）",),
        "include_brackets": True,
    },
    {
        "label": "LAW",
        "description": "The name of a law or legal document.",
        "examples": ("民法典", "宪法第一修正案"),
    },
    {
        "label": "LANGUAGE",
        "description": "The name of a natural language (e.g. Mandarin, English).",
        "examples": ("普通话", "Swahili"),
    },
    {
        "label": "DATE",
        "description": (
            "A date or date range, absolute (e.g. October 2023) or relative (e.g. yesterday)."
        ),
        "examples": ("2023年10月", "昨天"),
    },
    {
        "label": "TIME",
        "description": "A time of day smaller than a day (e.g. 9 a.m., 14:30).",
        "examples": ("上午9点", "14:30"),
    },
    {
        "label": "PERCENT",
        "description": "A percentage (e.g. 50%, thirty percent).",
        "examples": ("50%", "百分之三十"),
    },
    {
        "label": "MONEY",
        "description": (
            "A monetary amount including the currency (e.g. 100 US dollars, five yuan)."
        ),
        "examples": ("100美元", "五元"),
    },
    {
        "label": "QUANTITY",
        "description": (
            "A non-monetary measurement: weight, distance, volume, temperature, and the like."
        ),
        "examples": ("三公里", "20千克"),
    },
    {
        "label": "ORDINAL",
        "description": "An ordinal number (e.g. first, 3rd, primary).",
        "examples": ("第一", "3rd"),
    },
    {
        "label": "CARDINAL",
        "description": "A bare quantity number (e.g. forty-seven, 2.5).",
        "examples": ("四十七", "2.5"),
    },
    {
        "label": "MISC",
        "description": ("A miscellaneous proper name that fits none of the above types."),
    },
    # --- Biomedical ---
    {
        "label": "DISEASE",
        "description": "The name of a disease or syndrome (e.g. diabetes, COVID-19).",
        "examples": ("二型糖尿病", "COVID-19"),
        "counter_examples": ("症状词如「发烧」（不是疾病名）",),
    },
    {
        "label": "DRUG",
        "description": "The name of a drug or medication (e.g. aspirin, metformin).",
        "examples": ("阿司匹林", "metformin"),
    },
    {
        "label": "GENE",
        "description": "The name of a gene (e.g. BRCA1, TP53).",
        "examples": ("BRCA1", "TP53"),
    },
    {
        "label": "PROTEIN",
        "description": "The name of a protein (e.g. insulin, hemoglobin).",
        "examples": ("胰岛素", "hemoglobin"),
    },
    {
        "label": "CHEMICAL",
        "description": "The name of a chemical substance (e.g. ethanol, glucose).",
        "examples": ("乙醇", "glucose"),
    },
    # --- Finance ---
    {
        "label": "TICKER",
        "description": "A security ticker symbol (e.g. AAPL, 600519.SH).",
        "examples": ("AAPL", "600519.SH"),
    },
    {
        "label": "STOCK_INDEX",
        "description": "The name of a market index (e.g. the SSE Composite, the S&P 500).",
        "examples": ("标普500", "上证综指"),
    },
    {
        "label": "CURRENCY",
        "description": (
            "The name of a currency (e.g. US dollar), as distinct from a monetary amount."
        ),
        "examples": ("美元", "人民币"),
    },
    # --- Software ---
    {
        "label": "SOFTWARE",
        "description": (
            "The name of a software program, library, or framework (e.g. Python, PyTorch)."
        ),
        "examples": ("Python", "PyTorch"),
    },
    {
        "label": "PROTOCOL",
        "description": ("The name of a network or communication protocol (e.g. HTTP, Bluetooth)."),
        "examples": ("HTTP", "蓝牙"),
    },
    {
        "label": "PLATFORM",
        "description": "The name of a platform or online service (e.g. WeChat, AWS, GitHub).",
        "examples": ("微信", "GitHub"),
    },
)
"""Type specs for BUILTIN_ENTITY_TYPES below (examples/counter-examples
render into typing states as boundary guides)."""

BUILTIN_ENTITY_TYPES: dict[str, EntityType] = {
    spec["label"]: EntityType(**spec) for spec in _BUILTIN_SPECS
}
"""Built-in entity type registry: 31 types across general + four vertical
domains, serving as the base for user extensions."""


def _builtin_pack(name: str, description: str, labels: list[str]) -> EntityPack:
    """Build a built-in pack from a label list (labels must be built-in)."""
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
"""Built-in entity pack registry: CORE on by default, three vertical
packs stacked on demand."""


@dataclass(frozen=True)
class EntityTaxonomy:
    """The entity type system: type registry, pack registry, default packs.

    One taxonomy fully describes "which types exist, which packs exist,
    and which packs are on by default" — it is the single source of
    type semantics for JevNER. Buildable from JSON (from_json) for
    non-programmer users via data files; constructible directly in
    code as well.

    Attributes:
        entity_types: Registry mapping label to entity type.
        entity_packs: Registry mapping pack name to entity pack.
        default_packs: Pack names enabled by default (used when a
            JevNER call does not pass entity_packs explicitly).
    """

    entity_types: dict[str, EntityType]
    entity_packs: dict[str, EntityPack]
    default_packs: tuple[str, ...]

    @classmethod
    def from_json(cls, config: dict[str, Any] | str, base: Self | None = None) -> Self:
        """Build a taxonomy from JSON configuration (extends built-ins by default).

        Three optional sections::

            {
              "entity_types": {"BRAND": "A brand name."},        # add/override types
              "entity_packs": {"ECOM": ["PERSON", "BRAND"]},     # add/override packs
              "default_packs": ["CORE", "ECOM"]                  # default packs
            }

        - entity_types: label -> description string (compact form), or
          label -> object {"description": ..., "examples": [...],
          "counter_examples": [...], "min_chars": n,
          "include_brackets": bool} (all fields except description
          optional); a same-label entry overrides the built-in;
        - entity_packs: pack name -> label list (or an object
          {"description": ..., "entity_types": [...]}); referenced
          types must exist in the merged type table;
        - default_packs: list of pack names enabled by default;
          defaults to the base taxonomy's default packs;
        - config may also be a JSON string (parsed automatically).

        Args:
            config: JSON configuration dict or JSON string.
            base: Extension base; None uses the built-in system.

        Returns:
            The assembled taxonomy.

        Raises:
            TypeError: config is neither a dict nor a JSON string.
            ValueError: Invalid configuration structure — unknown type
                references, empty descriptions, packs referencing
                missing types (error messages include field paths).
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
        # Parse entity_types: label -> description, override or add.
        custom_types = config.get("entity_types") or {}
        if not isinstance(custom_types, dict):
            raise ValueError(f'"entity_types" must be a JSON object, got {type(custom_types)!r}.')
        for label, spec in custom_types.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Entity type label must be a non-empty string, got {label!r}.")
            if isinstance(spec, str):
                spec = {"description": spec}
            if not isinstance(spec, dict):
                raise ValueError(
                    f'Entity type "{label}" must be a description string or an object, '
                    f"got {type(spec)!r}."
                )
            description = spec.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f'Entity type "{label}" needs a non-empty description.')
            examples = spec.get("examples", [])
            if not _nonempty_strs(examples):
                raise ValueError(
                    f'Entity type "{label}" examples must be a list of non-empty strings.'
                )
            counter_examples = spec.get("counter_examples", [])
            if not _nonempty_strs(counter_examples):
                raise ValueError(
                    f'Entity type "{label}" counter_examples must be a list of non-empty strings.'
                )
            min_chars = spec.get("min_chars", 1)
            if not isinstance(min_chars, int) or isinstance(min_chars, bool) or min_chars < 1:
                raise ValueError(f'Entity type "{label}" min_chars must be a positive integer.')
            include_brackets = spec.get("include_brackets", False)
            if not isinstance(include_brackets, bool):
                raise ValueError(f'Entity type "{label}" include_brackets must be a boolean.')
            entity_types[label] = EntityType(
                label=label,
                description=description,
                examples=tuple(examples),
                counter_examples=tuple(counter_examples),
                min_chars=min_chars,
                include_brackets=include_brackets,
            )
        # Parse entity_packs: name -> label list (or object with description).
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
        # Parse default_packs.
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
        """Resolve pack selection plus explicit types into a final type sequence.

        Args:
            entity_packs: Packs (objects or names); None uses the defaults.
            entity_types: Extra explicit types (objects or labels),
                unioned with the packs.

        Returns:
            Deduplicated type sequence sorted by label (stable state
            rendering).

        Raises:
            ValueError: An unregistered pack name or type label was
                referenced, or the resolution came out empty.
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
        """Pack name to pack object (objects pass through; unknown names raise)."""
        if isinstance(pack, EntityPack):
            return pack
        if pack not in self.entity_packs:
            raise ValueError(f'Unknown entity pack "{pack}".')
        return self.entity_packs[pack]

    def _type_of(self, item: EntityType | str) -> EntityType:
        """Type label to type object (objects pass through; unknown labels raise)."""
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
"""Default taxonomy: built-in types + built-in packs, CORE on by default."""

PERSON_LABEL = "PERSON"
"""PERSON type label: entities with this primary type upgrade to Person
when with_gender is enabled."""

NONE_LABEL = "NONE"
"""The non-entity option label in typing Choice questions."""

NONE_DESCRIPTION = (
    "None of the entity types: a common word, phrase, or sentence; also a date, "
    "time, plain number, monetary amount, phone number, email address, web "
    "address, or a lone job title or honorific when no enabled type covers it."
)
"""Option description for the NONE veto in typing Choice questions.

A bare NONE label leaves the model to guess what "not an entity"
means; an explicit enumeration of common false positives sharpens the
veto (A1c in docs/optimization-roadmap.md)."""

PARTIAL_LABEL = "PARTIAL"
"""The truncated-entity veto label in typing Choice questions."""

PARTIAL_DESCRIPTION = (
    "The span truncates a named entity: it is only a fragment of a longer "
    "name, missing its beginning or end."
)

MIXED_LABEL = "MIXED"
"""The name-plus-glue-words veto label in typing Choice questions."""

MIXED_DESCRIPTION = (
    "The span sticks to a named entity: it contains a name plus attached "
    "verbs, prepositions, particles, titles, or modifiers."
)

VETO_LABELS: tuple[str, ...] = (NONE_LABEL, PARTIAL_LABEL, MIXED_LABEL)
"""Typing choices that veto an entity regardless of other probabilities.

PARTIAL and MIXED turn near-miss boundaries (a truncated name, a name
with glue words) into explicit classes instead of forcing the model
to choose between NONE and a real type (A2 in the roadmap)."""


@dataclass(frozen=True)
class Entity:
    """A recognized named entity (possibly holding multiple ranked types).

    Judgment always depends on context (the full window is the request
    state); one span can match several types at once (a country is both
    a GPE and a LOC). types is ranked by relevance (probability
    descending); a single type is never forced.

    start/end are original-text character offsets (end exclusive) with
    the invariant text == source[start:end] — nesting (entity_tree) is
    derived from offset containment, which is more reliable than
    substring tests: repeated mentions of one entity and whitespace
    normalization differences never confuse ownership.

    Attributes:
        text: Entity text (an exact substring of the source).
        types: (entity type, probability) tuples ranked by probability;
            always at least one entry.
        start: Start offset in the source text (inclusive).
        end: End offset in the source text (exclusive).
    """

    text: str
    types: tuple[tuple[EntityType, float], ...]
    start: int
    end: int

    @property
    def type(self) -> EntityType:
        """Primary type: the most relevant type (first entry of types)."""
        return self.types[0][0]

    @property
    def base_probability(self) -> float:
        """Probability of the primary type (first entry of types)."""
        return self.types[0][1]


@dataclass(frozen=True)
class Person(Entity):
    """A person entity upgraded with a gender judgment (primary type PERSON).

    Attributes:
        gender: The gender judgment.
        gender_probability: The model's probability for that gender
            (from Choice probabilities).
    """

    gender: Gender
    gender_probability: float


@dataclass(frozen=True)
class EntityNode:
    """A node in the entity nesting tree: one entity plus its inner entities.

    Returned directly by JevNER calls: __str__ renders an indented
    tree, so printing gives a readable nesting structure with no
    downstream post-processing.

    Attributes:
        entity: The entity at this node.
        children: Inner entity nodes sorted by (start, -end, text).
    """

    entity: Entity
    children: tuple[EntityNode, ...] = ()

    def __str__(self) -> str:
        """Render this node as the root of a readable indented tree.

        Returns:
            Tree text, e.g.
            "江苏南京 (GPE 1.00) [16:20]\\n├── 江苏 (GPE 1.00) [16:18]".
        """
        return "\n".join(_render_node(self, ""))

    __repr__ = __str__


def _render_node(node: EntityNode, prefix: str) -> list[str]:
    """Render a node and its subtree into lines (with box-drawing prefixes).

    The node's own label line carries no box-drawing prefix; child
    connectors are added uniformly by the parent: each level continues
    the parent prefix (spaces after the last child, vertical bars
    otherwise).

    Args:
        node: The node to render.
        prefix: The accumulated box-drawing prefix before this node
            (not applied to its own line; used for children).

    Returns:
        Lines of text: first line is this node's plain label; child
        lines are prefix + connector + child's first line, and deeper
        lines are joined recursively.
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
    """One-line label for an entity: text, ranked types, gender, offsets."""
    parts = [e.text, "(" + " · ".join(f"{t.label} {p:.2f}" for t, p in e.types) + ")"]
    if isinstance(e, Person):
        parts.append(f"[{e.gender.value} {e.gender_probability:.2f}]")
    parts.append(f"[{e.start}:{e.end}]")
    return " ".join(parts)


def entity_tree(entities: list[Entity]) -> list[EntityNode]:
    """Rebuild a flat entity list into a nesting tree by offset containment.

    Containment is judged by offsets: outer contains inner if and only
    if outer.start <= inner.start and inner.end <= outer.end. One
    stack scan after sorting builds the tree: iterate in
    (start, -end, text) order; the stack top is the nearest enclosing
    entity not yet closed.

    Nested entities are not a strict tree: crossing spans (overlapping
    ranges that contain neither other, e.g. GPE(0,10) vs PERSON(2,5)
    and FAC(4,8)) attach to the nearest common ancestor as siblings
    rather than being forced under one branch.

    Args:
        entities: Flat entity list (repeated mentions allowed).

    Returns:
        Top-level (uncontained) nodes sorted by (start, -end).
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
