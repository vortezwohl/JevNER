"""Prompt templates for jevner.

Renders the three request-level state prompts through prompt4py's
GeneralTemplate, all following the structured prompt design
methodology (seven-layer semantic model + canonical serialization
order): token detection state (per-token Noul), entity typing state
(sub-span Choice), and name gender state. Prompt bodies are English
(System One models are trained primarily on English, so English
instructions resolve most stably), while example text data stays in
its original language (multilingual examples are managed CONTEXT,
deliberately covering unspaced and right-to-left scripts for
worldwide language support).

What callers and maintainers need to know first:
- All three states carry SOURCE_TEXT as the INPUT payload; the type
  table and definitions ride as read-only CONTEXT, and each question's
  instructions carry only the minimal dynamic payload.
- The output protocol triple (OUTPUT_DATATYPE / OUTPUT_FORMAT /
  OUTPUT_EXAMPLE) is present in every state, keeping model answers
  programmatically checkable.
- Rendered results are JSON strings (render(markdown=False)); System
  One models are trained to read structured text, and key names are
  the semantic labels.
"""

from __future__ import annotations

from typing import cast

from prompt4py.template import GeneralTemplate

from jevner.entities import EntityType


def render_detection_state(source_text: str) -> str:
    """Render the state prompt for a token-detection request.

    Each question answers, as a Noul, whether INPUT.TOKEN is part of
    some named entity in SOURCE_TEXT. Hits drive the deterministic
    region reassembly in code.

    Args:
        source_text: The window text (shared context for all questions
            in this request).

    Returns:
        The state as a JSON string, for use in a system_one request.
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
            'In "陈嘉禾来自江苏", the tokens "陈", "嘉", "禾" are parts of a person '
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
    """Render the state prompt for an entity-typing request.

    Each question picks, as a Choice, the best entity type for
    INPUT.SPAN from CONTEXT.ENTITY_TYPES; NONE means the span is not
    a named entity. With open types, every EntityType carries its own
    label and description, and the type table is generated directly
    from the argument (no global description table involved).

    Args:
        source_text: The window text (shared context for all questions
            in this request).
        entity_types: The enabled entity types for this recognition,
            already sorted by label; written into CONTEXT.

    Returns:
        The state as a JSON string, for use in a system_one request.
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
    """Render the state prompt for a name-gender request.

    Returns:
        The state as a JSON string, for use in a system_one request.
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
