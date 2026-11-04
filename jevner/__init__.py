"""jevner: a no-local-model SDK for multilingual named entity recognition.

A System One gateway model (default baseten/mammoth-1) replaces all
local BERT inference, following DeepLoTX BertNER's long-text sliding
window algorithm, with a bundled Qwen3 tokenizer. Switch models any
time via model_name (any model registered on the gateway works, e.g.
jev-latest).

Type system: the base pack EntityPack.CORE (18 OntoNotes 5.0 types +
MISC fallback) is on by default; vertical packs EntityPack.BIOMED
(biomedical), EntityPack.FINANCE (finance), and EntityPack.TECH
(software) stack on demand. The whole system is open: extend types
and packs at runtime through EntityTaxonomy.from_json.

Most-used entry points:
- JevNER: the recognizer; calling it returns the nested entity forest
  (list[EntityNode]; printing renders a readable tree);
- EntityType / Gender / EntityPack: type, gender, and pack objects;
- Entity / Person / EntityNode: result dataclasses.

Credentials come from the TYPESAFE_API_KEY / TYPESAFE_BASE_URL
environment variables; beyond the gateway API there is no model
download and no local inference.
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
