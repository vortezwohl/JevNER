"""jevner offline tests: algorithm components verified without network.

Covers offline tokenizer loading, token offset consistency,
source-anchored windows, offset-synced stripping, the open type
system (JSON taxonomy), multi-type Entity semantics, per-occurrence
cross-window merging, and the entity tree (entity_tree/__str__).
All pure local deterministic computation; no API requests.
"""

from __future__ import annotations

import pytest

from jevner.entities import (
    DEFAULT_TAXONOMY,
    Entity,
    EntityTaxonomy,
    EntityType,
    Gender,
    Person,
    entity_tree,
)
from jevner.ner import (
    BYTES_PER_TOKEN_WORST,
    DEFAULT_WINDOW_SIZE,
    REQUEST_MAX_BYTES,
    REQUEST_STATE_BUDGET,
    JevNER,
)
from jevner.tokenizer import BundledTokenizer, strip_edge_punctuation, strip_span

# Chinese (unspaced) + Thai (complex clusters) + Arabic (RTL)
ZH_TEXT = "你好, 我的名字是陈嘉禾, 来自江苏南京."
TH_TEXT = "สวัสดีครับผมชื่อสมชายมาจากกรุงเทพมหานคร"
AR_TEXT = "مرحبا اسمي محمد وأعمل في شركة أرامكو في الرياض"

# Open types: EntityType(label, description) is all it takes
T = DEFAULT_TAXONOMY.entity_types

tokenizer = BundledTokenizer()
print("[1] Offline tokenizer load: OK")

counts = {text: tokenizer.count_tokens(text) for text in (ZH_TEXT, TH_TEXT, AR_TEXT)}
assert all(n > 0 for n in counts.values()), "token counts must be positive"
print(f"[2] token counts: zh={counts[ZH_TEXT]}, th={counts[TH_TEXT]}, ar={counts[AR_TEXT]}")

# Token offset consistency: text[start:end] must reproduce the token
for text in (ZH_TEXT, TH_TEXT, AR_TEXT):
    for start, end, tok in tokenizer.window_tokens(text):
        assert text[start:end] == tok, f"token offset mismatch: {tok!r} vs {text[start:end]!r}"
print("[3] Token offset consistency: OK")

# short text: one window; default sized from the 65,535-byte budget
assert DEFAULT_WINDOW_SIZE == REQUEST_STATE_BUDGET // BYTES_PER_TOKEN_WORST == 12_000
assert REQUEST_MAX_BYTES == 65_535, "server request-body hard limit"
wins = tokenizer.windows(ZH_TEXT, window_size=128, stride=64)
assert len(wins) == 1, "short text should be one whole window"
assert wins[0][0] == ZH_TEXT and len(wins[0][1]) == counts[ZH_TEXT] and wins[0][2] == 0, (
    "window text should equal the source"
)
print(f"[4] Single window for short text + 12k default ({DEFAULT_WINDOW_SIZE} tokens): OK")

# long text: multi-window overlap, key entities covered, offsets source-anchored
long_zh = "。".join(f"第{i}句话提到北京和广州以及华为公司" for i in range(80))
wins = tokenizer.windows(long_zh, window_size=64, stride=32)
assert len(wins) > 1, "long text should split into multiple windows"
# base indices advance by stride and index the source token stream
assert [b for _wt, _w, b in wins] == [i * 32 for i in range(len(wins))], "bases follow stride"
covered = "".join(tok for _wt, w, _b in wins for _s, _e, tok in w)
assert "北京" in covered and "广州" in covered and "华为" in covered, (
    "key entities must appear in the windows"
)
for window_text, w, _b in wins:
    c_start, c_end = w[0][0], w[-1][1]
    assert window_text == long_zh[c_start:c_end], "window text must be the source character range"
    for start, end, tok in w:
        assert long_zh[start:end] == tok, "window token offsets must anchor to the source"
print(f"[5] Long-text windows: {len(wins)} (window=64, stride=32), source-anchored: OK")

# offset-synced stripping: text and offsets stay consistent (the anchor)
text, s, e = strip_span(ZH_TEXT, 6, 13)  # strip edge punctuation/whitespace with synced offsets
assert ZH_TEXT[s:e] == text, "strip_span offsets must match the text"
assert strip_span("子豪,", 0, 3) == ("子豪", 0, 2)
assert strip_span(" 美国 ", 0, 4) == ("美国", 1, 3)
money = strip_span("$100!", 0, 5)
assert money == ("$100", 0, 4), f"currency $ (Sc) stays, trailing ! (Po) strips: {money}"
for probe in (",,", "。北京。", "  ", "a，b"):
    t2, s2, e2 = strip_span(probe, 0, len(probe))
    assert probe[s2:e2] == t2, f"strip_span invariant broken: {probe!r} -> {(t2, s2, e2)}"
assert strip_edge_punctuation("子豪,") == "子豪"
print("[6] Offset-synced stripping invariant: OK")

# open type system: taxonomy.resolve (defaults / mixed objects+names / extra types)
default_types = set(DEFAULT_TAXONOMY.resolve(None, None))
core_types = set(DEFAULT_TAXONOMY.entity_packs["CORE"].entity_types)
assert default_types == core_types, "CORE should be on by default"
mixed = DEFAULT_TAXONOMY.resolve(["BIOMED", "TECH"], [T["PERSON"]])
biomed_labels = ("DISEASE", "DRUG", "GENE", "PROTEIN", "CHEMICAL")
tech_labels = ("SOFTWARE", "PROTOCOL", "PLATFORM")
assert set(mixed) == {T[lab] for lab in biomed_labels} | {T[lab] for lab in tech_labels} | {
    T["PERSON"]
}
# objects and names mixed
mixed2 = DEFAULT_TAXONOMY.resolve([DEFAULT_TAXONOMY.entity_packs["FINANCE"], "BIOMED"], None)
fin_labels = {t.label for t in mixed2}
assert fin_labels == {
    "TICKER",
    "STOCK_INDEX",
    "CURRENCY",
    "DISEASE",
    "DRUG",
    "GENE",
    "PROTEIN",
    "CHEMICAL",
}
with pytest.raises(ValueError):
    DEFAULT_TAXONOMY.resolve(["NOT_EXIST"], None)
with pytest.raises(ValueError):
    DEFAULT_TAXONOMY.resolve([], [])
print("[7] taxonomy.resolve (objects/names/defaults): OK")

# EntityTaxonomy.from_json: consumer JSON with three sections (types + packs + defaults)
custom = EntityTaxonomy.from_json(
    """
    {
      "entity_types": {"BRAND": "A brand or trademark name (e.g. Apple, Nike)."},
      "entity_packs": {"ECOM": ["PERSON", "ORG", "BRAND"]},
      "default_packs": ["CORE", "ECOM"]
    }
    """
)
assert "BRAND" in custom.entity_types, f"custom type should register: {sorted(custom.entity_types)}"
assert custom.entity_types["BRAND"] == EntityType(
    "BRAND", "A brand or trademark name (e.g. Apple, Nike)."
)
assert "ECOM" in custom.entity_packs, f"custom pack should register: {sorted(custom.entity_packs)}"
ecom_labels = {t.label for t in custom.entity_packs["ECOM"].entity_types}
assert ecom_labels == {"PERSON", "ORG", "BRAND"}, f"ECOM should contain three types: {ecom_labels}"
resolved = custom.resolve(None, None)
labels = {t.label for t in resolved}
assert "BRAND" in labels and "GPE" in labels, (
    f"CORE+ECOM should contain built-in and custom types: {sorted(labels)}"
)
# dict form (with description) and built-in description override
custom2 = EntityTaxonomy.from_json(
    {
        "entity_types": {
            "PERSON": "An individual human being.",
            "BRAND": "A brand or trademark name (e.g. Apple, Nike).",
        },
        "entity_packs": {
            "ECOM": {"description": "E-commerce pack.", "entity_types": ["PERSON", "BRAND"]}
        },
    }
)
assert custom2.entity_types["PERSON"].description == "An individual human being."
assert custom2.entity_packs["ECOM"].description == "E-commerce pack."
# invalid configurations raise per case
with pytest.raises(ValueError):
    EntityTaxonomy.from_json({"entity_packs": {"X": ["NOPE"]}})  # pack references unknown type
with pytest.raises(ValueError):
    EntityTaxonomy.from_json({"default_packs": ["NOPE"]})  # default pack not registered
with pytest.raises(ValueError):
    EntityTaxonomy.from_json({"entity_types": {"BRAND": ""}})  # empty description
with pytest.raises(TypeError):
    EntityTaxonomy.from_json(42)  # invalid input type
print("[8] EntityTaxonomy.from_json (custom taxonomy): OK")

# multi-type Entity semantics: ranking + compat properties
multi = Entity(
    text="日本",
    types=((T["GPE"], 0.9), (T["LOC"], 0.7), (T["ORG"], 0.05)),
    start=0,
    end=2,
)
assert multi.type is T["GPE"] and multi.type.label == "GPE"
assert multi.base_probability == 0.9
probs = [p for _, p in multi.types]
assert probs == sorted(probs, reverse=True), "types must rank by probability"
print(f"[9] multi-type Entity semantics: {multi.type.label}: OK")

# merge: same-occurrence union, distinct occurrences kept (offline)
combiner = JevNER.__new__(JevNER)  # reuse the pure _combine only; no client initialized
ents = [
    # occurrence [0,10): window A ORG; window B ORG + secondary GPE
    Entity(text="中华人民共和国外交部", types=((T["ORG"], 0.95),), start=0, end=10),
    Entity(
        text="中华人民共和国外交部", types=((T["ORG"], 0.90), (T["GPE"], 0.30)), start=0, end=10
    ),
    # nested inner GPE [0,7)
    Entity(text="中华人民共和国", types=((T["GPE"], 1.00),), start=0, end=7),
    # independent entities at other positions
    Entity(text="江苏", types=((T["GPE"], 1.00),), start=16, end=18),
    Entity(text="南京", types=((T["GPE"], 0.99),), start=18, end=20),
]
merged = combiner._combine(ents, deduplicate=True)
by_span = {(e.start, e.end): e for e in merged}
org = by_span[(0, 10)]
org_labels = {t.label for t, _ in org.types}
assert org_labels == {"ORG", "GPE"}, f"same occurrence unions types: {org_labels}"
org_probs = {t.label: p for t, p in org.types}
assert org_probs["ORG"] == 0.95, f"same type takes max probability: {org_probs}"
assert (0, 7) in by_span, f"inner GPE must survive: {merged}"
assert (16, 18) in by_span and (18, 20) in by_span, f"independent entities must survive: {merged}"
print(f"[10] cross-window merge (per occurrence): {[(e.text, e.start, e.end) for e in merged]}")

# entity tree: offset containment + crossing siblings + __str__ rendering
nested = [
    Entity(text="中华人民共和国外交部发言人", types=((T["ORG"], 0.99),), start=0, end=12),
    Entity(
        text="中华人民共和国", types=((T["GPE"], 1.00),), start=0, end=7
    ),  # inner GPE of the outer span
    Entity(
        text="周明远", types=((T["PERSON"], 0.90),), start=9, end=11
    ),  # crossing: inside ORG, outside GPE
    Entity(text="北京", types=((T["GPE"], 0.95),), start=14, end=16),  # independent top-level
]
roots = entity_tree(nested)
top_texts = [n.entity.text for n in roots]
assert set(top_texts) == {"中华人民共和国外交部发言人", "北京"}, (
    f"top level holds only uncontained entities: {top_texts}"
)
outer = next(n for n in roots if n.entity.text == "中华人民共和国外交部发言人")
child_texts = [c.entity.text for c in outer.children]
# inner GPE and crossing PERSON both sit under the outer (siblings)
assert set(child_texts) == {"中华人民共和国", "周明远"}, (
    f"nested/crossing attach to the common ancestor: {child_texts}"
)
gpe_child = next(c for c in outer.children if c.entity.text == "中华人民共和国")
assert gpe_child.children == (), "inner GPE has no deeper level"
rendered = str(outer)
assert "├──" in rendered and "└──" in rendered, (
    f"rendering should contain box-drawing lines: {rendered}"
)
print("[11] 实体嵌套树 + 交叉嵌套 + 渲染: OK")
print(rendered)
person = Person(
    text="周明远",
    types=((T["PERSON"], 0.9),),
    start=9,
    end=11,
    gender=Gender.MALE,
    gender_probability=0.8,
)
assert "MALE" in str(entity_tree([person])), "rendering should contain gender"
print("[12] Person 树渲染（含性别）: OK")

# --- phase 3/5/7 offline coverage ---

# from_json object form: examples / counter_examples / min_chars / include_brackets
obj = EntityTaxonomy.from_json(
    {
        "entity_types": {
            "BRAND": {
                "description": "A brand name.",
                "examples": ["Apple", "Nike"],
                "counter_examples": ["generic nouns like 'phone'"],
                "min_chars": 2,
                "include_brackets": True,
            }
        },
        "entity_packs": {"ECOM": ["BRAND"]},
        "default_packs": ["ECOM"],
    }
)
brand = obj.entity_types["BRAND"]
assert brand.examples == ("Apple", "Nike") and brand.counter_examples != ()
assert brand.min_chars == 2 and brand.include_brackets is True
with pytest.raises(ValueError):
    EntityTaxonomy.from_json({"entity_types": {"X": {"examples": ["ok"]}}})  # no description
with pytest.raises(ValueError):
    EntityTaxonomy.from_json({"entity_types": {"X": {"description": "d", "min_chars": 0}}})
print("[13] from_json object form (guides/min_chars/brackets): OK")

# built-in types carry guides and conventions
assert DEFAULT_TAXONOMY.entity_types["PERSON"].min_chars == 2
assert DEFAULT_TAXONOMY.entity_types["PERSON"].counter_examples != ()
assert DEFAULT_TAXONOMY.entity_types["WORK_OF_ART"].include_brackets is True
print("[14] built-in type guides loaded: OK")

# stopword-edge pruning: Chinese particles and Latin prepositions at edges
from jevner.ner import bad_span_edge  # noqa: E402

assert bad_span_edge("的张伟")
assert bad_span_edge("张伟的")
assert bad_span_edge("the United States")
assert bad_span_edge("Ministry of")
assert not bad_span_edge("张伟")
assert not bad_span_edge("Bank of America")  # internal preposition survives
assert not bad_span_edge("北京大学")
print("[15] bad_span_edge (stopword edges, internal words survive): OK")

# A7 flatten: same-type fragments dropped, adjacent same-type merged
from jevner.ner import _flatten_fragments  # noqa: E402

src_text = "Tim Cook 在 Bank of America 工作"
t_per = T["PERSON"]
frag_case = [
    Entity(text="Tim Cook", types=((t_per, 0.99),), start=0, end=8),
    Entity(text="Tim", types=((t_per, 0.90),), start=0, end=3),
]
flat1 = _flatten_fragments(src_text, frag_case)
assert [e.text for e in flat1] == ["Tim Cook"], f"same-type fragment dropped: {flat1}"
split_case = [
    Entity(text="Tim", types=((t_per, 0.99),), start=0, end=3),
    Entity(text="Cook", types=((t_per, 0.95),), start=4, end=8),
]
flat2 = _flatten_fragments(src_text, split_case)
assert [e.text for e in flat2] == ["Tim Cook"], f"adjacent same-type merged: {flat2}"
mixed_case = [
    Entity(text="江苏南京", types=((T["GPE"], 0.99),), start=0, end=4),
    Entity(text="江苏", types=((T["GPE"], 1.0),), start=0, end=2),  # nested same type
    Entity(text="南京", types=((T["GPE"], 0.9),), start=2, end=4),
]
flat3 = _flatten_fragments("江苏南京", mixed_case)
assert [e.text for e in flat3] == ["江苏南京"], f"nested same-type collapsed: {flat3}"
print("[16] _flatten_fragments (fragments/adjacent/nested): OK")

# disk-cache codecs roundtrip: ranked types survive encode -> decode
from jevner.ner import _decode_ranked, _encode_ranked  # noqa: E402

ranked = ((T["GPE"], 0.9), (T["LOC"], 0.7))
assert _decode_ranked(_encode_ranked(ranked)) == ranked
assert _decode_ranked(_encode_ranked(())) == ()
print("[17] ranked-type disk codec roundtrip: OK")

print()
print("All offline tests passed.")
