"""One-time benchmark data preparation for the JevNER frozen baseline.

Downloads three public Chinese NER datasets (MSRA / Resume / CLUENER)
from HuggingFace parquet mirrors, decodes them into explicit-offset
JSONL items, and freezes fixed-seed dev/test samples into bench/sets/.
Also freezes the 24-language sweep cases from tests/test_languages.py
as a multilingual recall suite.

Item format (one JSON object per line)::

    {"id": "msra-123", "text": "...", "entities": [["北京", "loc", 0, 2], ...]}

entity tuples are [surface, label, start, end] with end exclusive and
text[start:end] == surface (asserted during preparation).

Dev/test come from disjoint source splits (test pool from the dataset's
test split where available) so dev-set tuning never leaks into the
final numbers. Re-run to re-freeze; changing --seed changes the frozen
samples and invalidates comparability with previous baselines.

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
MAX_CHARS = 300  # benchmark sentences are short; cap cost per item

# Public HuggingFace parquet mirrors of the standard Chinese NER datasets.
SOURCES: dict[str, dict[str, str]] = {
    "msra": {
        "dev_pool": "PassbyGrocer/msra-ner/data/train-00000-of-00001.parquet",
        "test_pool": "PassbyGrocer/msra-ner/data/test-00000-of-00001.parquet",
    },
    "resume": {
        "dev_pool": "PassbyGrocer/resume-ner/data/validation-00000-of-00001.parquet",
        "test_pool": "PassbyGrocer/resume-ner/data/test-00000-of-00001.parquet",
    },
    "cluener": {
        "dev_pool": "xusenlin/clue-ner/data/train-00000-of-00001-39fc4e004146ece2.parquet",
        "test_pool": "xusenlin/clue-ner/data/validation-00000-of-00001-bc5663be3b7ff2cd.parquet",
    },
}

# Tag id -> label name for the BIO-encoded parquet datasets.
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

# Model-facing type specs per dataset (English prompts resolve most
# stably; example text stays in its original language). Counters target
# the confusions observed in the 2026-11-05 baseline run.
SCHEMAS: dict[str, dict[str, dict]] = {
    "msra": {
        "per": {
            "description": "A specific person's name.",
            "examples": ["张伟", "温家宝"],
            "counter_examples": ["职务称谓如「总书记」「教授」（不是人名）"],
            "min_chars": 2,
        },
        "loc": {
            "description": "A location name: a country, province, city, or district.",
            "examples": ["北京", "长江流域"],
            "counter_examples": ["机构名如「北京大学」（那是 org）"],
            "min_chars": 2,
        },
        "org": {
            "description": "An organization name: a company, institution, agency, or school.",
            "examples": ["北京大学", "信息产业部"],
            "counter_examples": ["地名如「北京」（那是 loc）"],
            "min_chars": 2,
        },
    },
    "resume": {
        "name": {
            "description": "A person's name.",
            "examples": ["邵辉"],
            "min_chars": 2,
        },
        "cont": {
            "description": "A nationality or continent.",
            "examples": ["中国", "美国", "欧美"],
            "counter_examples": ["民族如「汉族」（那是 race）", "地名如「北京」（那是 loc）"],
        },
        "edu": {
            "description": "An educational institution: a school or university name.",
            "examples": ["北京大学", "浙江大学"],
            "counter_examples": ["公司名如「阿里巴巴」（那是 org）"],
            "min_chars": 2,
        },
        "loc": {
            "description": "A location name: a place, city, or region.",
            "examples": ["北京", "上海"],
            "counter_examples": [
                "国籍如「中国」（那是 cont）",
                "学校名、公司名、工作单位都不是 loc",
            ],
            "min_chars": 2,
        },
        "org": {
            "description": "An organization: a company, government body, or association.",
            "examples": ["阿里巴巴", "中央电视台"],
            "counter_examples": ["大学与学校名（那是 edu）"],
            "min_chars": 2,
        },
        "pro": {
            "description": "A professional field or qualification studied or held.",
            "examples": ["软件工程", "会计", "市场营销"],
            "counter_examples": ["职位头衔如「总经理」「工程师」（那是 title）"],
        },
        "race": {
            "description": "An ethnicity.",
            "examples": ["汉族", "回族", "蒙古族"],
            "counter_examples": ["国籍如「中国」（那是 cont）"],
        },
        "title": {
            "description": "A job title or position held at an organization.",
            "examples": ["总经理", "董事长", "工程师"],
            "counter_examples": ["专业领域如「软件工程」（那是 pro）"],
        },
    },
    "cluener": {
        "address": {
            "description": "A street or postal address.",
            "examples": ["北京市朝阳区望京街10号"],
            "counter_examples": ["景点名如「故宫」（那是 scene）", "城市名单独出现时不是 address"],
        },
        "book": {
            "description": "A book title.",
            "examples": ["红楼梦", "三体"],
            "counter_examples": ["影视作品名（那是 movie）"],
            "include_brackets": True,
        },
        "company": {
            "description": "A company name.",
            "examples": ["腾讯", "字节跳动"],
            "counter_examples": [
                "政府机构（那是 government）",
                "学校、社团等（那是 organization）",
            ],
            "min_chars": 2,
        },
        "game": {
            "description": "A game name.",
            "examples": ["王者荣耀", "原神"],
        },
        "government": {
            "description": "A government body, agency, or public institution.",
            "examples": ["国家发改委", "教育部"],
            "counter_examples": ["公司名（那是 company）"],
        },
        "job": {
            "description": "A job position or profession held.",
            "examples": ["产品经理", "教师"],
            "counter_examples": ["职级头衔如「董事长」（那是 position）"],
        },
        "movie": {
            "description": "A movie or film title.",
            "examples": ["流浪地球", "泰坦尼克号"],
            "counter_examples": ["书名（那是 book）"],
            "include_brackets": True,
        },
        "name": {
            "description": "A person's name.",
            "examples": ["王小婷"],
            "min_chars": 2,
        },
        "organization": {
            "description": (
                "An organization other than a company or government body: "
                "an NGO, university, hospital, or club."
            ),
            "examples": ["北京航空航天大学", "红十字会"],
            "counter_examples": ["公司名（那是 company）", "政府机构（那是 government）"],
        },
        "position": {
            "description": "A job title or rank.",
            "examples": ["董事长", "高级工程师"],
            "counter_examples": ["职业品类如「产品经理」（那是 job）"],
        },
        "scene": {
            "description": "A scenic spot, tourist attraction, or venue.",
            "examples": ["故宫", "迪士尼乐园"],
            "counter_examples": ["街道地址（那是 address）"],
            "min_chars": 2,
        },
    },
}


def download(rel: str) -> Path:
    """Fetch a HuggingFace parquet into bench/data/ (cached; ~30 MB total).

    Args:
        rel: Full repo-relative path, e.g.
            "PassbyGrocer/msra-ner/data/train-00000-of-00001.parquet".

    Returns:
        The local parquet path under bench/data/.
    """
    parts = rel.split("/")
    url = (
        f"https://huggingface.co/datasets/{parts[0]}/{parts[1]}/resolve/main/{'/'.join(parts[2:])}"
    )
    target = DATA / "_".join(parts)
    if target.exists():
        print(f"  cached: {target.name}")
        return target
    print(f"  downloading: {url}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(target, "wb") as out:
                out.write(resp.read())
            return target
        except Exception as exc:  # noqa: BLE001 - one-time data prep, report and retry
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {exc}")
    raise AssertionError("unreachable")


def norm_label(raw: str) -> str:
    """Normalize a raw tag to a lowercase schema label (loc / per / org ...)."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def bio_to_items(name: str, df: pd.DataFrame, tags: list[str], sep: str) -> list[dict]:
    """Decode BIO rows (tokens + ner_tag ids) into explicit-offset items."""
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
        entities: list[list[object]] = []
        cur: list[object] | None = None
        for (s, e), tag in zip(offsets, labels, strict=True):
            # cur stores the RAW tag type; normalization happens at emission,
            # so the I- continuation check compares like with like.
            if tag.startswith("B-") or (
                tag.startswith("I-") and (cur is None or cur[0] != tag[2:])
            ):
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
        items.append(
            {
                "id": f"{name}-{i}",
                "text": text,
                "entities": [[text[s:e], norm_label(lab), s, e] for lab, s, e in entities],
            }
        )
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


def clean(items: list[dict]) -> list[dict]:
    """Keep meaningful, length-capped sentences."""
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

    meta: dict[str, dict] = {}
    for name, pools in SOURCES.items():
        print(f"[{name}]")
        if name == "cluener":
            dev_pool = cluener_to_items(f"{name}_dev", pd.read_parquet(download(pools["dev_pool"])))
            test_pool = cluener_to_items(name, pd.read_parquet(download(pools["test_pool"])))
            tags_used = sorted({e[1] for it in test_pool for e in it["entities"]})
        else:
            tags = MSRA_TAGS if name == "msra" else RESUME_TAGS
            sep = ""  # both Chinese datasets are character-tokenized (no separator)
            dev_pool = bio_to_items(
                f"{name}_dev", pd.read_parquet(download(pools["dev_pool"])), tags, sep
            )
            test_pool = bio_to_items(name, pd.read_parquet(download(pools["test_pool"])), tags, sep)
            tags_used = sorted({e[1] for it in test_pool for e in it["entities"]})
        dev_items = clean(dev_pool)
        test_items = clean(test_pool)
        test = random.Random(args.seed).sample(test_items, min(args.test, len(test_items)))
        dev = random.Random(args.seed + 1).sample(dev_items, min(args.dev, len(dev_items)))
        write_jsonl(SETS / f"{name}_test.jsonl", test)
        write_jsonl(SETS / f"{name}_dev.jsonl", dev)
        n_ent = sum(len(it["entities"]) for it in test)
        print(
            f"  test={len(test)} ({n_ent} entities, {len(tags_used)} types)  "
            f"dev={len(dev)}  types={tags_used}"
        )
        meta[name] = {
            "types": SCHEMAS[name],
            "source": {
                k: f"https://huggingface.co/datasets/{v.split('/')[0]}/{v.split('/')[1]}"
                for k, v in pools.items()
            },
            "seed": args.seed,
            "test_size": len(test),
            "dev_size": len(dev),
            "max_chars": MAX_CHARS,
        }

    languages = load_language_cases()
    write_jsonl(SETS / "languages.jsonl", languages)
    meta["languages"] = {
        "source": "tests/test_languages.py LANGUAGE_CASES (frozen)",
        "cases": len(languages),
    }
    (SETS / "schemas.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[languages] cases={len(languages)}")
    print("Frozen sets written to bench/sets/.")


if __name__ == "__main__":
    sys.exit(main())
