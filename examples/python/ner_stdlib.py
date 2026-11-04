"""JevNER's underlying API, in pure Python standard library — no SDK, no pip install.

A toy NER loop over one sentence, showing the wire protocol the `jevner`
package is built on:

1. Noul question per word  -> "is this word part of an entity?" (probability)
2. Choice question per hit -> "which entity type is it?" (calibrated distribution)

Run:
    export TYPESAFE_API_KEY=...      # EasyJev key (required)
    export TYPESAFE_BASE_URL=...     # optional, default https://api.easyjev.online
    python ner_stdlib.py

This is the reference example: it is verified to run against the live gateway.
For real NER — windowing, byte budgets, caching, taxonomy, nested trees — use
the SDK (`pip install jevner`); that logic is the package.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.easyjev.online").rstrip("/")
API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
MODEL = os.environ.get("JEVNER_MODEL", "baseten/mammoth-1")

TEXT = "Maria Curie visited Paris and met Doctor Strange."


def system_one(state: dict, questions: dict) -> dict:
    """POST one batch of typed questions against shared state."""
    request = urllib.request.Request(
        BASE_URL + "/v1/systemone",
        data=json.dumps({"state": state, "model": MODEL, "questions": questions}).encode(),
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
            # The gateway rejects requests without a User-Agent.
            "User-Agent": "jevner-example/0.1 (+https://github.com/vortezwohl/JevNER)",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    if not API_KEY:
        sys.exit("set TYPESAFE_API_KEY first")

    # Whitespace tokens keep this demo dependency-free; the SDK ships a real tokenizer.
    words = [w for w in re.split(r"[^\w]+", TEXT, flags=re.UNICODE) if w]

    # Stage 1: detection — one Noul per word, all in a single request.
    detect = system_one(
        {"TEXT": TEXT},
        {
            f"w{i}": {
                "type": "noul",
                "instructions": {
                    "INPUT": {"NAME": word},
                    "TASK": "Is this word part of a named entity in the text?",
                },
                "criteria": {"true": "part of a named entity", "false": "not an entity"},
            }
            for i, word in enumerate(words)
        },
    )
    hits = [
        (word, detect["answers"][f"w{i}"]["noul"])
        for i, word in enumerate(words)
        if detect["answers"][f"w{i}"]["noul"] >= 0.5
    ]

    # Stage 2: typing — one Choice per detected word, one more request.
    typing = (
        system_one(
            {"TEXT": TEXT},
            {
                f"s{i}": {
                    "type": "choice",
                    "instructions": {
                        "INPUT": {"NAME": word},
                        "TASK": "Which entity type fits this span best?",
                    },
                    "criteria": {
                        "PERSON": "a named person",
                        "GPE": "a city, state, or country",
                        "NONE": "none of the above",
                    },
                }
                for i, (word, _noul) in enumerate(hits)
            },
        )
        if hits
        else None
    )

    print(f"text  : {TEXT}")
    print(f"model : {detect['model']}")
    for i, (word, noul) in enumerate(hits):
        answer = typing["answers"][f"s{i}"] if typing else None
        type_line = f"{answer['choice']} @ {answer['confidence']:.2f}" if answer else "?"
        print(f"{word:16s} noul={noul:.2f}  {type_line}")


if __name__ == "__main__":
    main()
