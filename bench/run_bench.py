"""Run the frozen JevNER benchmark: strict-F1 per dataset plus the
multilingual recall sweep, with usage accounting.

Each dataset runs with EXACTLY its own label set as the schema
(entity_packs=[] + the dataset's EntityType objects), so gold labels
and prediction labels live in the same namespace and strict F1 is an
exact (start, end, type) triple match. Nested predictions (inner
entities) count as predictions — against flat gold they are precision
hits or misses like any other span. Multi-type entities count once,
by their primary type.

Writes a JSON report to bench/results/<tag>.json and prints a summary.

Usage:
    uv run --env-file .env python bench/run_bench.py --tag baseline
    uv run --env-file .env python bench/run_bench.py \
        --datasets msra --limit 15 --no-languages --tag smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from jevner import JevNER
from jevner.entities import EntityNode, EntityType
from jevner.ner import DEFAULT_MODEL

HERE = Path(__file__).parent
SETS = HERE / "sets"
RESULTS = HERE / "results"
CHUNK = 32
"""Items submitted concurrently per wave; the instance semaphore
(max_workers) still bounds the actual in-flight request rate."""


def load_items(name: str, split: str, limit: int | None) -> list[dict]:
    """Load a frozen set (languages.jsonl has no split suffix)."""
    path = SETS / ("languages.jsonl" if name == "languages" else f"{name}_{split}.jsonl")
    items = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return items[:limit] if limit else items


def build_types(specs: dict[str, dict]) -> list[EntityType]:
    """Build EntityType objects from a dataset's schema specs."""
    out: list[EntityType] = []
    for label, spec in sorted(specs.items()):
        out.append(
            EntityType(
                label,
                spec["description"],
                examples=tuple(spec.get("examples", ())),
                counter_examples=tuple(spec.get("counter_examples", ())),
                min_chars=int(spec.get("min_chars", 1)),
                include_brackets=bool(spec.get("include_brackets", False)),
            )
        )
    return out


def flatten(nodes: list[EntityNode]) -> list[EntityNode]:
    """Collect a forest into a flat node list (inner entities included)."""
    out: list[EntityNode] = []
    for n in nodes:
        out.append(n)
        out.extend(flatten(list(n.children)))
    return out


def prf(tp: int, pred: int, gold: int) -> tuple[float, float, float]:
    """Precision / recall / F1 with zero-division guards."""
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


async def run_dataset(
    ner: JevNER,
    name: str,
    items: list[dict],
    types: list[EntityType],
    flatten_fragments: bool = False,
) -> dict[str, Any]:
    """Recognize every item under the dataset's own schema; strict-F1 score."""
    gold_total = 0
    pred_total = 0
    tp_total = 0
    per_type: dict[str, dict[str, int]] = {}
    failed: list[str] = []
    for start in range(0, len(items), CHUNK):
        chunk = items[start : start + CHUNK]
        results = await asyncio.gather(
            *(
                ner.arecognize(
                    it["text"],
                    with_gender=False,
                    entity_packs=[],
                    entity_types=types,
                    flatten_fragments=flatten_fragments,
                )
                for it in chunk
            ),
            return_exceptions=True,
        )
        for it, res in zip(chunk, results, strict=True):
            gold = {(s, e, lab) for _surface, lab, s, e in it["entities"]}
            if isinstance(res, BaseException):
                failed.append(it["id"])
                preds: set[tuple[int, int, str]] = set()
            else:
                preds = {(n.entity.start, n.entity.end, n.entity.type.label) for n in flatten(res)}
            hit = gold & preds
            gold_total += len(gold)
            pred_total += len(preds)
            tp_total += len(hit)
            for lab in {t[2] for t in gold} | {t[2] for t in preds}:
                d = per_type.setdefault(lab, {"gold": 0, "pred": 0, "tp": 0})
                d["gold"] += sum(1 for t in gold if t[2] == lab)
                d["pred"] += sum(1 for t in preds if t[2] == lab)
                d["tp"] += sum(1 for t in hit if t[2] == lab)
        done = min(start + CHUNK, len(items))
        print(f"  [{name}] {done}/{len(items)}")
    precision, recall, f1 = prf(tp_total, pred_total, gold_total)
    per_type_out: dict[str, Any] = {}
    for lab in sorted(per_type):
        d = per_type[lab]
        p, r, f = prf(d["tp"], d["pred"], d["gold"])
        per_type_out[lab] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            **d,
        }
    return {
        "items": len(items),
        "failed_items": failed,
        "gold": gold_total,
        "pred": pred_total,
        "tp": tp_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_type": per_type_out,
    }


async def run_languages(ner: JevNER, items: list[dict]) -> dict[str, Any]:
    """Multilingual recall sweep: expected entity substring must surface."""
    failed: list[str] = []
    for it in items:
        forest = await ner.arecognize(it["text"], with_gender=False)
        texts = {n.entity.text for n in flatten(forest)}
        if not any(it["expected"] in t or t in it["expected"] for t in texts):
            failed.append(it["id"])
    return {"cases": len(items), "passed": len(items) - len(failed), "failed": failed}


async def amain(args: argparse.Namespace) -> int:
    schemas = json.loads((SETS / "schemas.json").read_text(encoding="utf-8"))
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE.parent
    )
    ner = JevNER(
        max_workers=args.max_workers,
        typing_strategy=args.typing_strategy,
        latency_first=args.latency_first,
    )
    report: dict[str, Any] = {
        "tag": args.tag,
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": git.stdout.strip() or "?",
            "model": DEFAULT_MODEL,
            "split": args.split,
            "limit": args.limit,
            "max_workers": args.max_workers,
            "typing_strategy": args.typing_strategy,
            "flatten": args.flatten,
            "latency_first": args.latency_first,
        },
        "datasets": {},
        "languages": None,
    }
    wall0 = time.perf_counter()
    for name in args.datasets:
        items = load_items(name, args.split, args.limit)
        types = build_types(schemas[name]["types"])
        print(f"[{name}] {len(items)} items, {len(types)} types")
        report["datasets"][name] = await run_dataset(ner, name, items, types, args.flatten)
        d = report["datasets"][name]
        print(
            f"  -> P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
            f"(gold={d['gold']} pred={d['pred']} tp={d['tp']} "
            f"failed_items={len(d['failed_items'])})"
        )
    if args.languages:
        items = load_items("languages", args.split, args.limit)
        print(f"[languages] {len(items)} cases")
        report["languages"] = await run_languages(ner, items)
        print(
            f"  -> passed {report['languages']['passed']}/{report['languages']['cases']}"
            f"  failed: {report['languages']['failed']}"
        )
    report["wall_seconds"] = round(time.perf_counter() - wall0, 1)
    report["usage"] = ner.usage.as_dict()
    await ner.aclose()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"wall={report['wall_seconds']}s  usage={report['usage']}")
    print(f"report: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["msra", "resume", "cluener"])
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument(
        "--typing-strategy", choices=["flat", "compete"], default="flat", help="A3 A/B switch"
    )
    ap.add_argument(
        "--latency-first", action="store_true", help="skip detection on short texts (fast path)"
    )
    ap.add_argument("--flatten", action="store_true", help="flatten_fragments mode (A7 A/B switch)")
    ap.add_argument("--no-languages", dest="languages", action="store_false")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
