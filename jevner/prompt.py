"""jevner 的提示词模板。

基于 prompt4py 的 GeneralTemplate 渲染三类请求级 state 提示词，
全部遵循结构化提示词设计方法论（七层语义模型 + 规范序列化顺序）：
词元检测 state（逐词元 Noul）、实体分型 state（子 span Choice）、
人名性别 state。提示词主体语言为英文（System One 模型以英语为
主要训练语言，英文指令的语义分辨最稳定），示例中的文本数据
本身保留原语种（多语种示例是受管理的 CONTEXT，刻意覆盖无空格
文字与右到左文字以支撑全球语种兼容）。

调用方与维护者进入本文件时最需要先知道：
- 三个 state 均以 SOURCE_TEXT 为 INPUT 载荷，类型表与定义作为只读
  CONTEXT 下发，每条问题的 instructions 只携带最小动态载荷；
- 输出协议三件套（OUTPUT_DATATYPE/OUTPUT_FORMAT/OUTPUT_EXAMPLE）
  在每个 state 中齐备，保证 jev 的回答可被程序校验；
- 渲染结果为 JSON 字符串（render(markdown=False)），System One
  模型被训练为理解结构化文本，键名即语义标签。
"""

from __future__ import annotations

from typing import cast

from prompt4py.template import GeneralTemplate

from jevner.entities import EntityType


def render_detection_state(source_text: str) -> str:
    """渲染词元检测请求的 state 提示词。

    Each question answers, as a Noul, whether INPUT.TOKEN is part of
    some named entity in SOURCE_TEXT. Hits drive the deterministic
    region reassembly in code.

    Args:
        source_text: 窗口原文（本请求全部问题的共享上下文）。

    Returns:
        JSON 字符串形式的 state，供 system_one 请求使用。
    """
    template = GeneralTemplate()
    template.role = "You are a meticulous multilingual text analyst working at token granularity."
    template.objective = (
        "Decide, for every question, whether its INPUT token is part of "
        "some named entity in SOURCE_TEXT."
    )
    template.instruction = [
        "Locate the token named by INPUT.TOKEN at INPUT.POSITION within SOURCE_TEXT.",
        "Treat INPUT.LEFT and INPUT.RIGHT as the adjacent tokens surrounding it.",
        "Judge whether that token lies inside the contiguous text of one named entity.",
    ]
    template.capability = (
        "You read every writing system: unspaced scripts (Chinese, Japanese, Thai), "
        "right-to-left scripts (Arabic, Hebrew), and agglutinative languages "
        "(Turkish, Malay, Indonesian)."
    )
    template.context = {
        "definition": (
            "A named entity is a proper name (person, place, organization, product, "
            "work, law, language) or an explicit date, time, or quantity. Common nouns, "
            "verbs, pronouns, and adjectives are not named entities."
        ),
        "examples": [
            'In "吴子豪来自福建", the tokens "吴", "子", "豪" are parts of a person '
            'name (yes); the verb "来自" is not (no).',
            'In "กรุงเทพมหานคร" (Bangkok Metropolis), the tokens "กร", "ุง", "เทพ" '
            "are each parts of the place name (yes).",
            'In "مرحبا اسمي محمد وأعمل في شركة أرامكو في الرياض", the tokens of '
            '"محمد" and "الرياض" are entity parts (yes); the preposition "في" is not (no).',
        ],
    }
    template.input = {"SOURCE_TEXT": source_text}
    template.output_dtype = "One noul value per question: a real number between 0 and 1."
    template.output_format = (
        "noul close to 1 means the token is part of a named entity; close to 0 means it is not."
    )
    template.output_example = {"noul": 0.97}
    template.constraint = [
        (
            "Judge only from SOURCE_TEXT and the question's INPUT; "
            "never invent entities from outside knowledge."
        ),
        (
            "A short token in an unspaced script can still be an entity part; "
            "judge every token on its own merits."
        ),
        "Whitespace-only and punctuation-only tokens are never entity parts.",
        "Function words (prepositions, verbs, particles) are never entity parts.",
    ]
    return cast(str, template.render(markdown=False))


def render_typing_state(source_text: str, entity_types: tuple[EntityType, ...]) -> str:
    """渲染实体分型请求的 state 提示词。

    Each question picks, as a Choice, the best entity type for
    INPUT.SPAN from CONTEXT.ENTITY_TYPES; NONE means the span is not
    a named entity. 开放式类型下每个 EntityType 自带 label 与
    description，类型表直接由参数生成（不再查全局描述表）。

    Args:
        source_text: 窗口原文（本请求全部问题的共享上下文）。
        entity_types: 本次识别启用的实体类型序列（已按 label 排序），
            类型表写入 CONTEXT 下发。

    Returns:
        JSON 字符串形式的 state，供 system_one 请求使用。
    """
    template = GeneralTemplate()
    template.role = "You are an expert multilingual named-entity-recognition annotator."
    template.objective = (
        "For every question, choose the entity type of its INPUT.SPAN within "
        "SOURCE_TEXT, or NONE when the span is not a named entity."
    )
    template.instruction = [
        "Read INPUT.SPAN together with its surrounding context in SOURCE_TEXT.",
        "Match the span's meaning in context against the type definitions in CONTEXT.ENTITY_TYPES.",
        (
            "Answer NONE when the span is a common word, or sticks to "
            "a verb, preposition, particle, or punctuation."
        ),
        "Answer NONE when the span cuts a word in half, leaving a truncated word form.",
    ]
    template.capability = (
        "You annotate named entities in every language with equal care, from "
        "unspaced scripts to right-to-left scripts."
    )
    template.context = {
        "ENTITY_TYPES": {t.label: t.description for t in entity_types},
        "note_on_nested_spans": (
            "A span may sit inside a longer entity. Type each span on its own "
            "content: the inner and the outer span answer separately."
        ),
        "examples": [
            'In "中华人民共和国外交部发表声明", SPAN "中华人民共和国" → GPE and '
            'SPAN "中华人民共和国外交部" → ORG both hold at once.',
            'In "I live in New York City", both SPAN "New York" and SPAN "New York City" → GPE.',
            'In "ผมชื่อสมชาย" (my name is Somchai), SPAN "สมชาย" → PERSON, while '
            'SPAN "อสม" and SPAN "ื่อสมชาย" truncate a word → NONE.',
            'In "مرحبا اسمي محمد", SPAN "محمد" → PERSON, while SPAN "اسمي محمد" '
            "sticks to the verb → NONE.",
        ],
    }
    template.input = {"SOURCE_TEXT": source_text}
    template.output_dtype = "One string label per question."
    template.output_format = (
        "A label from CONTEXT.ENTITY_TYPES, or NONE when the span is not a named entity."
    )
    template.output_example = {"choice": "GPE"}
    template.constraint = [
        "Choose only a label from CONTEXT.ENTITY_TYPES or NONE; never output any other string.",
        "The span must cover exactly one whole named entity: no attached verbs, "
        "prepositions, particles, or punctuation.",
        "A span that truncates a word (an incomplete word form, common in unspaced "
        "scripts) is NONE.",
        (
            "Judge the span by its actual meaning in SOURCE_TEXT, "
            "not by surface similarity to a type name."
        ),
    ]
    return cast(str, template.render(markdown=False))


def render_gender_state() -> str:
    """渲染人名性别判定请求的 state 提示词。

    Returns:
        JSON 字符串形式的 state，供 system_one 请求使用。
    """
    template = GeneralTemplate()
    template.role = (
        "You are an expert on given-name gender conventions across the world's languages."
    )
    template.objective = "For every question, decide the most likely gender of its INPUT.NAME."
    template.instruction = [
        "Read the given name in INPUT.NAME.",
        "Weigh its word formation and the naming conventions of its language.",
        "Answer UNKNOWN when the name carries no clear gender signal or is commonly unisex.",
    ]
    template.capability = (
        "You know given-name conventions in Chinese, English, Arabic, Thai, Russian, "
        "and other languages, including names that transliterate across scripts."
    )
    template.context = {
        "options": {
            "MALE": "A name conventionally borne by men.",
            "FEMALE": "A name conventionally borne by women.",
            "UNKNOWN": "No clear gender signal, or a name commonly used for any gender.",
        },
    }
    template.input = {
        "NOTE": "This request shares no common text; all evidence is in each question's INPUT.NAME."
    }
    template.output_dtype = "One string label per question."
    template.output_format = "Exactly one of: MALE, FEMALE, UNKNOWN."
    template.output_example = {"choice": "MALE"}
    template.constraint = [
        (
            "Judge from the name itself and its language conventions; "
            "no other information is needed or allowed."
        ),
        "Never output any string other than MALE, FEMALE, or UNKNOWN.",
    ]
    return cast(str, template.render(markdown=False))
