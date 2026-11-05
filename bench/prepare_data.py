"""Benchmark data preparation: 12 public NER datasets, frozen samples.

Datasets: MSRA, Resume, CLUENER, Weibo, CoNLL03, WNUT17, MIT-Restaurant,
and 5 CrossNER domains (ai/literature/music/politics/science) — the
standard zero-shot Chinese/English NER benchmark mix.

Pipeline per dataset:
1. Download the public HuggingFace mirror (cached under bench/data/,
   gitignored; each dataset keeps its own license).
2. Decode into explicit-offset JSONL items
   {"id", "text", "entities": [[surface, label, start, end], ...]}
   with text[start:end] == surface asserted.
3. Clean (strip, drop -DOCSTART-, cap 510 chars — BERT-style baseline
   input limit, kept for cross-pipeline comparability).
4. Freeze fixed-seed samples: test 200 (random.Random(2026)) and dev 50
   (random.Random(2027)) drawn from DISJOINT source splits (test sample
   from the dataset's test split where one exists), so dev-set tuning
   never leaks into final numbers.

Label conventions per dataset live in bench/schemas_v3/<name>.json
(annotation-guideline-derived descriptions; committed).

Usage:
    uv run python bench/prepare_data.py [--test 200] [--dev 50] [--seed 2026]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
SETS = HERE / "sets"
SCHEMAS_V3 = HERE / "schemas_v3"
MAX_CHARS = 510  # BERT-style baselines cannot read longer inputs

# Public HuggingFace endpoints: (dataset name -> {pool: URL}).
SOURCES: dict[str, dict[str, str]] = {
    "msra": {
        "dev_pool": "https://huggingface.co/datasets/PassbyGrocer/msra-ner/resolve/main/data/train-00000-of-00001.parquet",
        "test_pool": "https://huggingface.co/datasets/PassbyGrocer/msra-ner/resolve/main/data/test-00000-of-00001.parquet",
    },
    "resume": {
        "dev_pool": "https://huggingface.co/datasets/PassbyGrocer/resume-ner/resolve/main/data/validation-00000-of-00001.parquet",
        "test_pool": "https://huggingface.co/datasets/PassbyGrocer/resume-ner/resolve/main/data/test-00000-of-00001.parquet",
    },
    "weibo": {
        "dev_pool": "https://huggingface.co/datasets/PassbyGrocer/weibo-ner/resolve/main/data/validation-00000-of-00001.parquet",
        "test_pool": "https://huggingface.co/datasets/PassbyGrocer/weibo-ner/resolve/main/data/test-00000-of-00001.parquet",
    },
    "cluener": {
        "dev_pool": "https://huggingface.co/datasets/xusenlin/clue-ner/resolve/main/data/train-00000-of-00001-39fc4e004146ece2.parquet",
        "test_pool": "https://huggingface.co/datasets/xusenlin/clue-ner/resolve/main/data/validation-00000-of-00001-bc5663be3b7ff2cd.parquet",
    },
    "conll": {
        "dev_pool": "https://huggingface.co/datasets/tomaarsen/conll2003/resolve/main/data/validation-00000-of-00001.parquet",
        "test_pool": "https://huggingface.co/datasets/tomaarsen/conll2003/resolve/main/data/test-00000-of-00001.parquet",
    },
    "wnut": {
        "dev_pool": "https://huggingface.co/datasets/tner/wnut2017/resolve/main/dataset/valid.json",
        "test_pool": "https://huggingface.co/datasets/tner/wnut2017/resolve/main/dataset/test.json",
        "labels": "https://huggingface.co/datasets/tner/wnut2017/resolve/main/dataset/label.json",
    },
    "mitres": {
        "dev_pool": "https://huggingface.co/datasets/tner/mit_restaurant/resolve/main/dataset/valid.json",
        "test_pool": "https://huggingface.co/datasets/tner/mit_restaurant/resolve/main/dataset/test.json",
        "labels": "https://huggingface.co/datasets/tner/mit_restaurant/resolve/main/dataset/label.json",
    },
}
CROSSNER_DOMAINS = ("ai", "literature", "music", "politics", "science")
for dom in CROSSNER_DOMAINS:
    SOURCES[f"crossner_{dom}"] = {
        "dev_pool": f"https://huggingface.co/api/datasets/DFKI-SLT/cross_ner/parquet/{dom}/validation/0.parquet",
        "test_pool": f"https://huggingface.co/api/datasets/DFKI-SLT/cross_ner/parquet/{dom}/test/0.parquet",
    }

# Tag id -> tag name for the BIO-encoded parquet datasets.
MSRA_TAGS = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
RESUME_TAGS = [
    "O",
    "B-CONT",
    "I-CONT",
    "B-EDU",
    "I-EDU",
    "B-LOC",
    "I-LOC",
    "B-NAME",
    "I-NAME",
    "B-ORG",
    "I-ORG",
    "B-PRO",
    "I-PRO",
    "B-RACE",
    "I-RACE",
    "B-TITLE",
    "I-TITLE",
]
WEIBO_TAGS = [
    "O",
    "B-GPE.NAM",
    "I-GPE.NAM",
    "B-GPE.NOM",
    "I-GPE.NOM",
    "B-LOC.NAM",
    "I-LOC.NAM",
    "B-LOC.NOM",
    "I-LOC.NOM",
    "B-ORG.NAM",
    "I-ORG.NAM",
    "B-ORG.NOM",
    "I-ORG.NOM",
    "B-PER.NAM",
    "I-PER.NAM",
    "B-PER.NOM",
    "I-PER.NOM",
]
CONLL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
CROSSNER_TYPES = [
    "academicjournal",
    "album",
    "algorithm",
    "astronomicalobject",
    "award",
    "band",
    "book",
    "chemicalcompound",
    "chemicalelement",
    "conference",
    "country",
    "discipline",
    "election",
    "enzyme",
    "event",
    "field",
    "literarygenre",
    "location",
    "magazine",
    "metrics",
    "misc",
    "musicalartist",
    "musicalinstrument",
    "musicgenre",
    "organisation",
    "person",
    "politicalparty",
    "politician",
    "product",
    "programlang",
    "protein",
    "researcher",
    "scientist",
    "song",
    "task",
    "theory",
    "university",
    "writer",
]
CROSSNER_TAGS = ["O"] + [f"{p}-{t}" for t in CROSSNER_TYPES for p in ("B", "I")]

BIO_TAGS: dict[str, list[str]] = {
    "msra": MSRA_TAGS,
    "resume": RESUME_TAGS,
    "weibo": WEIBO_TAGS,
    "conll": CONLL_TAGS,
    **{f"crossner_{dom}": CROSSNER_TAGS for dom in CROSSNER_DOMAINS},
}
DATASET_ORDER = list(BIO_TAGS) + ["cluener", "wnut", "mitres"]


def download(url: str) -> Path:
    """Fetch a dataset file into bench/data/ (cached across runs).

    Cache keys derive from the FULL path (scheme/host/resolve/main
    stripped): identical file names across dataset repos must never
    collide — a shared cache name once made one dataset decode
    another's file.
    """
    parts = [
        p for p in url.split("/") if p not in ("https:", "", "huggingface.co", "resolve", "main")
    ]
    target = DATA / "_".join(parts)
    if target.exists():
        return target
    print(f"  downloading: {'/'.join(parts[-3:])}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp, open(target, "wb") as out:
                out.write(resp.read())
            return target
        except Exception as exc:  # noqa: BLE001 - one-time data prep, report and retry
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {exc}")
    raise AssertionError("unreachable")


def norm_label(raw: str) -> str:
    """Normalize a raw tag to a lowercase schema label (per, gpe_nam, ...)."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def tokens_to_entities(
    text: str, offsets: list[tuple[int, int]], labels: list[str]
) -> list[list[object]]:
    """Decode a BIO tag sequence over token offsets into entity tuples.

    cur holds the RAW tag type; normalization happens at emission, so the
    I- continuation check compares like with like.
    """
    entities: list[list[object]] = []
    cur: list[object] | None = None
    for (s, e), tag in zip(offsets, labels, strict=True):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
            if cur:
                entities.append(cur)
            cur = [tag[2:], s, e]
        elif tag.startswith("I-"):
            cur[2] = e
        else:
            if cur:
                entities.append(cur)
            cur = None
    if cur:
        entities.append(cur)
    return [[text[s:e], norm_label(lab), s, e] for lab, s, e in entities]


def parquet_tags(path: Path, column: str, fallback: list[str]) -> list[str]:
    """Read a BIO tag-name table from the parquet's HuggingFace metadata.

    Datasets-ETL parquet files embed the ClassLabel names under
    schema.metadata[b"huggingface"]; deriving the table from the file
    itself removes every hardcoded-mapping risk. Falls back to the
    static table when metadata is absent.

    Args:
        path: Local parquet file.
        column: Tag column name (ner_tags).
        fallback: Static tag table used when no metadata exists.

    Returns:
        Tag id -> tag name list.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    blob = schema.metadata.get(b"huggingface") if schema.metadata else None
    if blob:
        info = json.loads(blob.decode())
        feat = info.get("info", {}).get("features", {}).get(column, {})
        names = feat.get("feature", {}).get("names") or feat.get("names")
        if names:
            return list(names)
    return fallback


def bio_to_items(
    name: str, df: pd.DataFrame, tags: list[str], sep: str, path: Path | None = None
) -> list[dict]:
    """Decode BIO rows (tokens + ner_tag ids) into explicit-offset items."""
    if path is not None:
        tags = parquet_tags(path, "ner_tags", tags)
    items: list[dict] = []
    for i, row in enumerate(df.itertuples()):
        tokens = list(row.tokens)
        labels = [tags[t] for t in row.ner_tags]
        text = ""
        offsets: list[tuple[int, int]] = []
        for j, tok in enumerate(tokens):
            if j and sep:
                text += sep
            offsets.append((len(text), len(text) + len(tok)))
            text += tok
        entities = tokens_to_entities(text, offsets, labels)
        for surface, _lab, s, e in entities:
            assert text[s:e] == surface, f"offset mismatch in {name}-{i}"
        items.append({"id": f"{name}-{i}", "text": text, "entities": entities})
    return items


def cluener_to_items(name: str, df: pd.DataFrame) -> list[dict]:
    """Decode CLUENER rows (text + explicit entity offsets) into items."""
    items: list[dict] = []
    for i, row in enumerate(df.itertuples()):
        entities = [
            [e["entity"], norm_label(e["label"]), int(e["start_offset"]), int(e["end_offset"])]
            for e in row.entities
        ]
        for surface, _lab, s, e in entities:
            assert row.text[s:e] == surface, (
                f"offset mismatch in {name}-{i}: {row.text[s:e]!r} != {surface!r}"
            )
        items.append({"id": f"{name}-{i}", "text": row.text, "entities": entities})
    return items


def tner_to_items(name: str, path: Path, label_path: Path) -> list[dict]:
    """Decode tner json rows ({"tokens", "tags"} + id->tag map) into items."""
    id2label = {v: k for k, v in json.loads(label_path.read_text(encoding="utf-8")).items()}
    items: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = list(row["tokens"])
        labels = [id2label[t] for t in row["tags"]]
        text = ""
        offsets: list[tuple[int, int]] = []
        for j, tok in enumerate(tokens):
            if j:
                text += " "
            offsets.append((len(text), len(text) + len(tok)))
            text += tok
        entities = tokens_to_entities(text, offsets, labels)
        for surface, _lab, s, e in entities:
            assert text[s:e] == surface, f"offset mismatch in {name}-{i}"
        items.append({"id": f"{name}-{i}", "text": text, "entities": entities})
    return items


def load_pool(name: str, pool: str) -> list[dict]:
    """Decode one source pool for a dataset (downloads as needed)."""
    urls = SOURCES[name]
    if name == "cluener":
        return cluener_to_items(name, pd.read_parquet(download(urls[pool])))
    if name.startswith("crossner_"):
        path = download(urls[pool])
        return bio_to_items(name, pd.read_parquet(path), CROSSNER_TAGS, " ", path)
    if name in ("wnut", "mitres"):
        return tner_to_items(name, download(urls[pool]), download(urls["labels"]))
    path = download(urls[pool])
    return bio_to_items(name, pd.read_parquet(path), BIO_TAGS[name], "", path)


def clean(items: list[dict]) -> list[dict]:
    """Keep meaningful, length-capped sentences (baseline input limit)."""
    return [
        it
        for it in items
        if it["text"].strip() and it["text"] != "-DOCSTART-" and len(it["text"]) <= MAX_CHARS
    ]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8"
    )


def load_language_cases() -> list[dict]:
    """Import LANGUAGE_CASES from tests/test_languages.py and freeze them."""
    spec = importlib.util.spec_from_file_location(
        "test_languages", HERE.parent / "tests" / "test_languages.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [
        {"id": lang, "text": text, "expected": expected}
        for lang, _continent, text, expected in mod.LANGUAGE_CASES
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--dev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    SETS.mkdir(parents=True, exist_ok=True)

    total_ent = 0
    for name in DATASET_ORDER:
        print(f"[{name}]")
        # Sampling order matters for reproducibility: test pool (seed) first,
        # then dev pool (seed+1); pools come from disjoint source splits.
        test_items = clean(load_pool(name, "test_pool"))
        dev_items = clean(load_pool(name, "dev_pool"))
        test = random.Random(args.seed).sample(test_items, min(args.test, len(test_items)))
        dev = random.Random(args.seed + 1).sample(dev_items, min(args.dev, len(dev_items)))
        write_jsonl(SETS / f"{name}_test.jsonl", test)
        write_jsonl(SETS / f"{name}_dev.jsonl", dev)
        labels = sorted({e[1] for it in test for e in it["entities"]})
        n_ent = sum(len(it["entities"]) for it in test)
        total_ent += n_ent
        print(
            f"  test={len(test)} ({n_ent} entities, {len(labels)} types)  dev={len(dev)}  "
            f"types={labels}"
        )

    languages = load_language_cases()
    write_jsonl(SETS / "languages.jsonl", languages)

    meta: dict[str, dict] = {}
    for name in DATASET_ORDER:
        schema = json.loads((SCHEMAS_V3 / f"{name}.json").read_text(encoding="utf-8"))
        meta[name] = {
            "types": schema["types"],
            "source": "public HuggingFace mirrors; annotation-guideline-derived schemas",
            "seed": args.seed,
            "test_size": args.test,
            "dev_size": args.dev,
            "max_chars": MAX_CHARS,
        }
    meta["languages"] = {
        "source": "tests/test_languages.py LANGUAGE_CASES (frozen)",
        "cases": len(languages),
    }
    (SETS / "schemas.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[languages] cases={len(languages)}")
    print(f"total test entities: {total_ent}")
    print("Frozen sets written to bench/sets/.")


if __name__ == "__main__":
    sys.exit(main())
