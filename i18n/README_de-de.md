<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**Zero-Shot-mehrsprachige Entitätenerkennung. Angetrieben von System-One-Entscheidungsmodellen.**

*Ein typsicheres Python-SDK, das ein Entscheidungsmodell in einen Entitätenerkennungsdienst verwandelt — verschachtelte Entitätsbäume, benutzerdefinierte Entitätspakete und ein mitgelieforter Qwen3-Tokenizer, ganz ohne lokale Modelle, die trainiert oder deployt werden müssten.*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="EasyJev-Logo" width="72" />
  </a>
  <br/>
  <strong>Gesponsert von <a href="https://www.easyjev.online/">EasyJev</a> — alle Entscheidungsmodelle, ein Endpunkt.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**Kein Training. Jede Sprache. Der Baum kommt im Rückgabewert.**

[English](../README.md) | [简体中文](README_zh-hans.md) | [繁體中文](README_zh-hant.md) | [日本語](README_ja-jp.md) | [한국어](README_ko-kr.md) | [Français](README_fr-fr.md) | [Español](README_es-es.md) | Deutsch | [Português](README_pt-br.md)

</div>

---

## Warum JevNER

Die meisten NER-Stacks zwingen zur Wahl: ein lokales Modell, dessen Labelmenge beim Fine-Tuning eingefroren wird, oder eine gehostete API, die verschweigt, wie sicher sie wirklich ist.

JevNER geht einen dritten Weg. Ein **System-One-Entscheidungsmodell** — kein Textgenerator — liest jede Frage und liefert **kalibrierte Wahrscheinlichkeiten** statt Prosa, und das SDK setzt diese Urteile zu einem **verschachtelten Entitätswald** zusammen, der sich als Baum ausdrucken lässt.

| Wo klassisches NER versagt | Was JevNER stattdessen tut |
| --- | --- |
| Labelmengen frieren beim Fine-Tuning ein; jede neue Domain bedeutet einen neuen Trainingslauf. | **Zero-Shot by Design.** Entitätstypen sind offene Daten: Jeder Nutzer (auch ohne Programmierkenntnisse) kann Typen und Pakete zur Laufzeit über ein einfaches JSON-Dokument ergänzen. |
| Mehrsprachigkeit bedeutet ein feinabgestimmter Checkpoint pro Sprache. | **Eine Pipeline, 24 Sprachen verifiziert** — Chinesisch, Thailändisch, Arabisch, Hebräisch, Hindi, Türkisch, Indonesisch, Filipino, Vietnamesisch, Japanisch, Koreanisch, Englisch, Französisch, Deutsch, Spanisch, Portugiesisch, Italienisch, Russisch, Niederländisch, Polnisch und mehr. Inklusive ressourcenarmer und leerzeichenloser Schriften. |
| Gehostete NER-APIs liefern flache Spans; Verschachtelung wird plattgedrückt oder verworfen. | **Der Rückgabewert IST der Baum.** `node.children` hält jede in `node` enthaltene Entität, abgeleitet aus den Offsets des Quelltexts (`text == source[start:end]`) — Verschachtelung ist strukturell, nicht geraten. |
| Ein Softmax-Score ist keine Wahrscheinlichkeit, nach der man routen kann. | **Kalibrierte Wahrscheinlichkeiten** vom Entscheidungsmodell: `0.8` verhält sich wie „in 80 % der Fälle richtig“ — genau das, was Review-Workflows mit Confidence-Schwellen brauchen. |
| Ein Span bekommt einen Typ, selbst wenn ein Name zugleich Ort und Organisation ist. | **Multi-Typ-Entitäten, gerankt.** `Entity.types` trägt jeden konkurrierenden Typ mit Wahrscheinlichkeit, sortiert nach Relevanz. |
| Jedes Deployment lädt Modellgewichte herunter. | **Überhaupt keine lokalen Modelle.** Ein Qwen3-Tokenizer (100+ Sprachen) liegt dem Paket bei; sämtliche Inferenz läuft über das EasyJev-Gateway. |

## Installation

- Von [PyPI](https://pypi.org/project/jevner/)

    ```shell
    pip install -U jevner
    ```

- Empfohlen, mit [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- Von [GitHub](https://github.com/vortezwohl/JevNER) — Zugang zu unveröffentlichten Funktionen.

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## Schnellstart

```python
from jevner import JevNER

ner = JevNER()  # Zugangsdaten aus TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# Der Rückgabewert IST der verschachtelte Wald: print rendert einen Baum.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

Das ist die vollständige erste Schleife: kein Modell-Download, kein Fine-Tuning, kein Post-Processing. PERSON-Entitäten werden zu `Person` mit einem Gender-Urteil aufgewertet, Offsets erfüllen stets `text == source[start:end]`, und die Verschachtelung reist im Rückgabewert selbst mit.

> [!TIP]
> Das Gateway routet mehrere System-One-Modelle. Standard ist `baseten/mammoth-1`; wechseln Sie jederzeit mit `JevNER(model_name="jev-latest")`.

> Aufruf aus TypeScript, Go oder purem `curl`? Das Wire-Protokoll ist pures JSON-REST — siehe [`examples/`](../examples/README.md).

## Kernfunktionen

| Funktion | Was Sie bekommen |
| --- | --- |
| Zero-Shot-Erkennung | Kein Training, keine Annotation, kein Fine-Tuning. Definieren, was extrahiert wird, und loslegen. |
| Entscheidungsmodell-Fundament | Noul-/Choice-Fragen mit kalibrierten Wahrscheinlichkeiten — Ja/Nein-Konfidenz und vollständige Optionsverteilungen, keine freie Generierung. |
| Verschachtelter Entitätswald | `ner(text)` liefert `list[EntityNode]`. Überkreuzende Spans hängen als Geschwister am nächsten gemeinsamen Vorfahren; Verschachtelung entsteht aus Offsets, nicht aus Teilstring-Raterei. |
| Kalibrierte Multi-Typ-Entitäten | `Entity.types` ist eine nach Relevanz sortierte Sequenz aus `(EntityType, probability)`-Tupeln — eine Entität kann GPE *und* LOC sein. |
| 24 Sprachen Abdeckung | Ende-zu-Ende über Asien, Europa und Amerika verifiziert, inklusive ressourcenarmer und leerzeichenloser Schriften. |
| Offene Taxonomie aus JSON | `EntityTaxonomy.from_json` nimmt ein Dreiseiten-Dokument (`entity_types`, `entity_packs`, `default_packs`), das die 31 eingebauten Typen und 4 Pakete erweitert oder überschreibt — für Nicht-Programmierer gebaut. |
| Vertikale Entitätspakete | `CORE` (18 OntoNotes-5.0-Typen + MISC) ist standardmäßig dabei; `BIOMED`, `FINANCE` und `TECH` stapeln sich auf Wunsch. |
| Gender-Erkennung | PERSON-Entitäten werden zu `Person` mit `MALE` / `FEMALE` / `UNKNOWN` und einer Wahrscheinlichkeit, prozessintern gecacht. |
| Qwen3-Tokenizer inklusive | Byte-Level-BPE für 100+ Sprachen liegt im Paket — kein HuggingFace-Zugriff, kein `transformers`, kein `torch`. |
| Anforderungsbudget-Sicherheit | Fenster werden an der 65.535-Byte-Body-Grenze des Gateways bemessen; Typing-Batches an der 512-Options-Grenze verengt — jede Anfrage bleibt konstruktionsgemäß in den Grenzen. |
| Dreistufiges Caching | Detektions-, Typing- und Gender-Ergebnisse liegen in LRU-Caches, verschlüsselt nach Fenster-Hash; wiederholter oder überlappender Text kommt in Millisekunden zurück. |
| Durchgehend typsicher | Vollständige Annotationen mit `py.typed`, mypy-sauber im Strict-Modus, Prompts über die strukturierten prompt4py-Templates gerendert. |

## Häufige Muster

### Den Wald drucken, nicht eine flache Liste

Der Rückgabewert ist bereits der Baum. `str(node)` rendert einen Knoten mit Nachkommen samt Linienzügen, Typen, Wahrscheinlichkeiten, Gender und Offsets.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### Ein vertikales Paket aktivieren

Pakete stapeln sich nach Namen; die an das Modell gesendete Typentabelle enthält exakt die Vereinigung Ihrer Auswahl.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### Eine Taxonomie aus JSON definieren

Jeder Nutzer — auch wer nie einen Code-Editor öffnet — kann das Erkennungsvokabular mit einem JSON-Dokument erweitern. Eigene Typen sind gleichberechtigte Bürger neben den eingebauten.

```python
from jevner import EntityTaxonomy, JevNER

taxonomy = EntityTaxonomy.from_json("""
{
  "entity_types": {
    "BRAND": "A brand or trademark name (e.g. Apple, Nike)."
  },
  "entity_packs": {
    "ECOM": ["PERSON", "ORG", "BRAND"]
  },
  "default_packs": ["CORE", "ECOM"]
}
""")

ner = JevNER(taxonomy=taxonomy)
forest = ner("Tim Cook announced that Apple launched the new iPhone while Nike sponsored it.")
```

`from_json` akzeptiert einen JSON-String oder ein bereits geparstes Dict. Unbekannte Typreferenzen, leere Beschreibungen und unregistrierte Default-Pakete werfen eine `ValueError` mit dem Feldpfad des Übeltäters.

### Auf kalibrierte Konfidenz routen

Jede Wahrscheinlichkeit stammt von einem auf Kalibrierung trainierten Entscheidungsmodell — Schwellen meinen also, was sie sagen:

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 heißt wirklich „in 80 % der Fälle richtig“
```

Wiederholte Aufrufe auf demselben Text treffen den Cache in Millisekunden — praktisch für RAG-Nachfragen und Dokument-Wiederholäufe.

## Entscheidungsmodelle

JevNER läuft auf jedem im Gateway registrierten System-One-Entscheidungsmodell — Modelle, die typisierte Fragen (Noul Ja/Nein, Choice Auswahl) mit kalibrierten Wahrscheinlichkeiten beantworten, statt Text zu generieren. Standard `baseten/mammoth-1`; `jev-latest` und 20+ weitere wechseln mit einem Argument:

| | |
| --- | --- |
| Für NER verifiziert | `baseten/mammoth-1` (Standard), `jev-latest`, `jev-1.13.0`, `jev-1.14.0` |
| Am günstigsten | `baseten/joey-1`, `respan/span-01-lite`, `perplexity/pplx-decider-flash-v1.2` — ab $0.01/M Eingabe-Tokens |
| Ebenfalls verfügbar | Cloudflare-Clef-Familie, Nace.AI Drex, LiquidAI d1, Microsoft Decision-1, Upstage Solar Decide, Inception Mercury Decide, OpenAI GPT-6 Luna Decisions, `jaredpalmer/kev-4b` und mehr |

Vollständiger Katalog mit Preisen und Modellvorbehalten: [Unterstützte Entscheidungsmodelle](../docs/models.md).

## Ökosystem und Vergleiche

JevNER ist eine Referenzanwendung der Entscheidungsmodell-Bewegung (System One) — das Ökosystem aus Modellen und Engines, die typisierte, kalibrierte Urteile statt Prosa zurückgeben: die TypeSafe-**Jev**-Familie, das aggregierte Gateway von [EasyJev](https://www.easyjev.online/), offene Engines wie [laya](https://github.com/NandhaKishorM/laya) (nicht-autoregressive Decision-Engine, 100+ Sprachen) und [kev](https://github.com/jaredpalmer/kev) (offene Entscheidungsmodelle auf Qwen) sowie Adapter wie [AnyJev](https://github.com/nokia-applied-research/AnyJev). Die Karte: [Das Entscheidungsmodell-Ökosystem](../docs/ecosystem.md).

Sie wählen einen NER-Stack? [JevNER vs GLiNER vs spaCy vs gehostete NER-APIs](../docs/alternatives.md) — ein ehrlicher Vergleich, einschließlich wann ein lokaler [GLiNER](https://github.com/urchade/GLiNER)-Checkpoint oder eine [spaCy](https://github.com/explosion/spaCy)-Pipeline die bessere Wahl ist.

## Öffentliche API

Das Root-Paket stellt bewusst nur acht Namen bereit:

```python
from jevner import (
    Entity,          # eine erkannte Entität: text, types, start, end
    EntityNode,      # Baumknoten: entity + children
    EntityPack,      # benanntes Bündel von Entitätstypen (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # offenes Typsystem: Registry + from_json + resolve
    EntityType,      # Label + Beschreibung als Wertobjekt
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # der Erkennungsdienst
    Person,          # Entity + gender + gender_probability
)
```

## Entwurfsgrenzen

JevNER besitzt die Erkennungspipeline und ihre Verträge. Bewusst nicht geboten werden:

- lokale Modellinferenz jeglicher Art;
- Modell-Hosting, Fine-Tuning oder Gewichtsverteilung;
- Trainingsdaten, Annotationswerkzeuge oder Evaluations-Harnesses;
- UI-Komponenten; und
- Re-Exports von Gateway-SDK-Internas.

Diese Grenze hält das SDK eine dünne, überprüfbare Pipeline, während Hosting und Produkterfahrung bei der Plattform bleiben.

## Sponsor

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="EasyJev-Logo" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**Alle System-One-Modelle. Ein Endpunkt. Perfekt kompatibel mit [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python).**

EasyJev bündelt die System-One-Entscheidungsmodelle — Jev, Decider, Drex, d1, Clef, kev und mehr von allen Anbietern — hinter einem einzigen Endpunkt, an den sich das offizielle [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) ohne jede Änderung anstecken lässt: `base_url` und `api_key` aufs Gateway richten und pro Anfrage ein beliebiges Modell wählen; bestehende Namen (`jev-latest`, `jev-1.14.0`, …) funktionieren unverändert weiter. Urteile kommen in Millisekunden zurück, Ausgabe-Tokens werden auf jedem Modell mit $0 abgerechnet, und die Anmeldung läuft nur per E-Mail-Code — das Konto entsteht bei der Erstanmeldung inklusive 1 $ Startguthaben. Aufladen per Karte (Visa / Mastercard), Apple Pay, Google Pay oder WeChat Pay — Einmalzahlung, kein Abo.

[Registrieren &rarr;](https://www.easyjev.online/) &middot; [Modellplatz](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [Preise](https://www.easyjev.online/#/pricing) &middot; [AGB](https://www.easyjev.online/#/tos) &middot; [Datenschutz](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## Entwicklung

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # Stil und statische Prüfungen
uv run mypy jevner                          # Typprüfung
uv run python tests/test_local.py           # Offline-Algorithmustests (12 Prüfungen)
uv run python tests/test_e2e.py             # Ende-zu-Ende (benötigt TYPESAFE_API_KEY, kostet echtes Geld)
uv run python tests/test_languages.py       # 24-Sprachen-Sweep (kostet echtes Geld)
uv run python tests/bench_quick.py          # Kompaktes Performance-Experiment (kostet echtes Geld)
```

## Beitragen

Beiträge sollten überprüfbares Pipeline-Verhalten verbessern: Detection-Recall, Typing-Präzision, Verschachtelungstreue, Taxonomie-Validierung, Cache-Verhalten oder Anforderungsbudget-Sicherheit. Halten Sie die öffentliche API schlank und fügen Sie für neues Verhalten unabhängig lauffähige Tests hinzu.

## Lizenz

[MIT](../LICENSE)
