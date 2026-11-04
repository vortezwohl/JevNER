# Examples: the JevNER wire protocol from any language

Every JevNER call bottoms out in one plain-JSON REST request:
`POST /v1/systemone` with a Bearer key, a shared `state`, and a batch of typed
`questions` (Noul yes/no, Choice selection). These examples run that protocol
directly — no SDK — so the API is usable from whatever stack you ship.

| Example | Run | Dependencies | Verified |
| --- | --- | --- | --- |
| [Python (stdlib)](python/ner_stdlib.py) | `python ner_stdlib.py` | Python 3.10+, standard library only | end to end against the live gateway |
| [TypeScript](typescript/ner.ts) | `npx tsx ner.ts` | Node 18+ (native fetch) | syntax-checked; see network note |
| [Go](go/ner.go) | `go run ner.go` | Go standard library only | see network note |
| [curl](curl/ner.sh) | `./ner.sh` | curl + python (for pretty output) | see network note |

All of them read the same environment:

```shell
export TYPESAFE_API_KEY=...                    # EasyJev key
export TYPESAFE_BASE_URL=https://api.easyjev.online   # optional, this is the default
export JEVNER_MODEL=baseten/mammoth-1          # optional, any gateway decision model
```

## What the examples do

A toy two-stage NER loop over one sentence — the same shape as the real SDK,
in miniature:

1. **Detection** — one *Noul* question per word ("is this word part of a named
   entity?"), all sent in a single request; keep words with probability ≥ 0.5.
2. **Typing** — one *Choice* question per detected word ("is it a PERSON, a GPE,
   or NONE?"), returning a calibrated distribution over types.

Whitespace tokenization keeps the examples dependency-free. The Python SDK ships
the real thing: a bundled Qwen3 tokenizer with byte offsets, sliding windows sized
to the gateway's 65,535-byte body cap, a 512-option batch guard, three-layer LRU
caching, open JSON taxonomies, and nested entity-tree assembly.

```shell
pip install jevner
```

## Network note

Every example sends a `User-Agent` header — the gateway rejects requests without
one. If you still see `ECONNRESET` during the TLS handshake (observed with some
`curl` and Node builds behind aggressive CDN/WAF edges), the edge is filtering the
TLS fingerprint of that particular client build; switching network, using a proxy
(`HTTPS_PROXY`), or running the Python example (which passes everywhere it was
tested) are the known workarounds.
