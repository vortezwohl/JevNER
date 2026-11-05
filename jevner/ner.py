"""JevNER core: sliding-window named entity recognition with no local models.

This module implements the jevner SDK pipeline. The overall structure
mirrors DeepLoTX BertNER (sliding windows + cross-window merging +
gender augmentation), but a System One gateway model replaces all
local BERT inference, and a two-stage "token detection + span typing"
design replaces BertNER fixed n-gram candidates so any language works:

1. Detection: the bundled Qwen3 tokenizer splits each window into a
   token sequence; every token becomes one Noul question ("is this
   token part of a named entity?") submitted in batches, with all
   batches of a window in flight at once. This mirrors the base model
   token-level prediction in BertNER, is language-agnostic, and is
   free of fixed n-gram length assumptions.
2. Typing: contiguous hit tokens reassemble into entity regions; every
   contiguous sub-span within a region (nesting included) becomes one
   Choice question (labels = enabled entity types + NONE) submitted in
   batches. Nested entities are covered naturally.
3. Cross-window merging: repeated recognitions at the same occurrence
   (identical start/end) merge into one, taking the union of types and
   the max probability per type; both sides of a nested pair survive,
   and the relation is derived downstream by entity_tree() from offset
   containment.
4. PERSON entities are deduplicated by text and batch-judged for
   gender (Choice: MALE/FEMALE/UNKNOWN), with results written to an
   in-process LRU cache to avoid duplicate requests.

Concurrency architecture (what callers and maintainers need first):
- All gateway I/O runs as coroutines on AsyncTypeSafeClient, on ONE
  private event loop owned by the instance (a single daemon thread —
  replacing the former ThreadPool). A single asyncio.Semaphore
  (max_workers) bounds in-flight requests; the sync __call__ facade
  and the async arecognize() entry both submit onto that same loop, so
  the HTTP client never crosses event loops (httpx AsyncClient is not
  loop-safe).
- Within a window, detection batches are independent questions and all
  fly concurrently; gray-zone recheck rounds are independent resamples
  of the same token set, so they also fly concurrently — the max over
  concurrent samples equals the old sequential-rounds result.
- Error semantics are preserved from the thread-pool era: detection
  failures propagate to the caller; typing-batch and window failures
  in the concurrent path are logged and skipped as empty results.
- What callers can rely on: __call__ stays synchronous and may be
  invoked from any thread (concurrent calls interleave on the private
  loop under the shared semaphore); awaitables should use arecognize.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
import traceback
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar, cast

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    TypeSafeAPIError,
)
from vortezwohl.cache import LRUCache
from vortezwohl.iter import batched

from jevner.disk_cache import DiskCache
from jevner.entities import (
    DEFAULT_TAXONOMY,
    MIXED_DESCRIPTION,
    MIXED_LABEL,
    NONE_DESCRIPTION,
    NONE_LABEL,
    PARTIAL_DESCRIPTION,
    PARTIAL_LABEL,
    PERSON_LABEL,
    VETO_LABELS,
    Entity,
    EntityNode,
    EntityPack,
    EntityTaxonomy,
    EntityType,
    Gender,
    Person,
    entity_tree,
)
from jevner.prompt import (
    render_detection_state,
    render_gender_state,
    render_nomination_state,
    render_typing_state,
)
from jevner.tokenizer import BundledTokenizer, strip_span

logger = logging.getLogger("jevner")

T = TypeVar("T")
"""Type variable for private-loop submission helpers."""


@dataclass
class Usage:
    """Cumulative gateway usage for one JevNER instance.

    The counters update on the private event loop as requests complete
    and cache lookups answer; reading from other threads is safe
    (individual int/float reads are atomic under the GIL). Retries
    inside the SDK client are not visible here and are not counted.

    Attributes:
        requests: system_one calls completed.
        questions: questions sent across all requests.
        cache_hits: questions answered from local caches instead of
            the gateway (a detection window hit counts every skipped
            token question; typing and gender hits count one per
            span/name).
        input_tokens: input tokens reported by the API (0 when the API
            does not report usage).
        output_tokens: output tokens reported by the API.
        seconds: wall time spent inside system_one calls (excluding
            semaphore queueing).
    """

    requests: int = 0
    questions: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return a plain-dict snapshot (for logging and reports)."""
        return {
            "requests": self.requests,
            "questions": self.questions,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_seconds": round(self.seconds, 3),
        }


DEFAULT_MODEL = "baseten/mammoth-1"
"""Default model: Mammoth 1 hosted on BaseTen (routed through the same
System One gateway).

Uses exactly the same system_one protocol (Noul/Choice) as the jev
family on the gateway; only the model field differs. Verified
protocol-compatible on the NER pipeline with comparable quality.
"""

LONG_CONTEXT_HEADER = {"TypeSafe-Danger-Zone-Long-Context": "true"}
"""Long-context request header for jev-family models.

Requests carrying this header get both per-request and per-question
contexts raised to 262k tokens (defaults: 64k/32k). Typing states
embed the full type table, so extra-long windows (64k+) depend on
it; for ordinary windows it is a no-op safety net.
"""

REQUEST_MAX_BYTES = 65_535
"""Hard cap on a single system_one request body (server-side limit).

The request body = state JSON + all question JSON. Oversized
requests are rejected by the server (observed as 500s / gateway
melt-downs) and the error never points at the cause — so the design
guarantees no request ever exceeds it: windows are cut from a byte
budget (REQUEST_STATE_BUDGET), and typing criteria carry labels only
(descriptions appear once, in the state type table).
"""

BYTES_PER_TOKEN_WORST = 4
"""Worst-case UTF-8 bytes per token.

Chinese costs roughly 3 bytes per token (one token per character);
English subwords average about 4. Use 4 as the uniform safe upper
bound when converting the window byte budget into token window size.
"""

REQUEST_STATE_BUDGET = 48_000
"""Byte budget for the window text inside a request body.

Out of the 65,535-byte body: subtract the typing type table (~2KB),
the question batch (~150-250 bytes per question at up to 64 per
batch, about 16KB), and JSON structural overhead. Detection and
typing share the same window, so the budget follows the tighter of
the two (typing).
"""

DEFAULT_WINDOW_SIZE = REQUEST_STATE_BUDGET // BYTES_PER_TOKEN_WORST
"""Default sliding-window size in tokens: byte budget / worst-case
bytes-per-token = 12k.

The window follows the tighter of two caps:
- Request body <= 65,535 bytes (hard limit, see REQUEST_MAX_BYTES):
  Chinese UTF-8 costs 3 bytes per character and English about 4
  bytes per token, so a 12k-token window stays within budget even at
  48KB of text plus batch questions and structural overhead;
- jev per-question context of 32k tokens: 12k leaves ample headroom.

Very long documents (millions of tokens) are handled by sliding
windows + window-level parallelism: 12k tokens per window, 50%
adjacent overlap, windows submitted concurrently on the private
event loop.
"""

DEFAULT_STRIDE_DIVISOR = 2
"""Default stride factor: stride = window_size // 2 (50% overlap).

Token-level detection is insensitive to window boundaries
(overlapping windows guarantee a boundary-crossing entity is fully
visible on at least one side), so 50% overlap suffices — half the
requests of the BertNER 25% step.
"""

DEFAULT_MAX_WORKERS = 16
"""Default in-flight request limit (the asyncio semaphore width).

The bottleneck is gateway round-trips and rate limiting, not local
CPU; this bounds how many requests (windows, detection batches,
recheck batches, typing batches, gender batches) may be in flight at
once on the private event loop. 0 or 1 serializes requests. Gateway
load test (2026-11-05): 16 and 32 concurrent requests completed with
zero errors; throughput saturates around 24, so values beyond that
add queueing without speed.
"""

DETECT_CACHE_CAPACITY = 128
"""In-process LRU capacity for detection results (keyed by window
hash)."""

TYPING_CACHE_CAPACITY = 4096
"""In-process LRU capacity for typing results (keyed by window hash +
span)."""

DEFAULT_PROB_THRESHOLD = 0.5
"""Default probability floor for an entity primary (most relevant)
type."""

SECONDARY_TYPE_THRESHOLD = 0.25
"""Probability floor for secondary entity types.

Choice probabilities are a competitive distribution across options;
the primary type (the selected option) must pass prob_threshold, and
other types at or above this floor ride along as secondary types,
all ranked by probability — one entity can hold several types at
once (a country is both GPE and LOC) without forcing a single one.
0.25 empirically: fragment spans in large batches show secondary
probabilities of 0.1-0.2, and the raised floor keeps noise types out."""

REQUEST_MAX_OPTIONS = 512
"""Cap on total options per request (enforced by the gateway backend;
exceeding it returns 422).

A Choice question carries (enabled types + NONE) options, so the
per-batch question count is computed as 512 / options per question
(e.g. CORE 20 options -> 25 questions per batch), keeping every
request within the limit under any taxonomy (custom multi-type
included).
"""

TYPING_BATCH_SIZE = 64
"""Upper bound on typing questions per request (further narrowed by
the option budget).

Typing demands fine semantic judgment; large batches dilute the
model NONE verdicts on fragment spans (in-batch interference), so
small batches are markedly more stable."""

DETECT_BATCH_SIZE = 128
"""Detection questions per request.

Server-side cost of extra questions in one request is ~zero (the
decision model answers the whole batch in one forward pass), so the
batch size trades only request-body bytes against REQUEST COUNT —
and request count is what the gateway's per-second ceiling meters.
128 keeps the batch well inside the 65KB body budget; the former 64
was set when per-question latency was assumed to scale with batch
size. Quality is guarded by the frozen dev benchmark (in-batch score
dilution was observed at 256 under the old schema; re-verified at
128 under the current one)."""

DETECT_PROB_THRESHOLD = 0.25
"""Noul decision threshold for token detection.

Detection prefers recall over precision: fragments and pseudo-spans
are rejected at the source by the typing stage NONE veto (plus the
prob_threshold primary floor), so a loose detection layer only adds
candidate spans — never fake entities. Empirically, frequent Chinese
single characters (digits inside drug names) can score as low as 0.3
in one round; a high threshold drops whole words."""

DETECT_GAP_TOLERANCE = 2
"""Missed-token tolerance during region reassembly.

In finely segmented scripts (Thai, Arabic), single-character tokens
such as tone marks and affixes are often individually voted down;
the tolerance bridges a few missed tokens so entity regions do not
break apart. Bridged non-entity fragments are filtered by the
typing stage NONE veto.
"""

DETECT_RECHECK_LOW = 0.15
"""Lower bound of the detection recheck gray zone.

Tokens scoring in [DETECT_RECHECK_LOW, 0.5) enter the recheck queue
for three independent rounds, keeping the maximum — the live API
has per-request sampling variance, and Chinese medical tokens can
score 0.2-0.4 in a single round. Max-of-three sharply cuts misses.
"""

DETECT_RECHECK_ROUNDS = 3
"""Recheck rounds for gray-zone tokens (1 first pass + 3 samples,
take the maximum)."""

DETECT_NEIGHBORHOOD = 128
"""Tokens of context each side of a detection batch's request state.

Every batch of a window used to re-send the whole window text (up to
48KB), dominating input-token cost on long documents (~188 batches
per full window). Detection questions carry their immediate LEFT/
RIGHT neighbors inline, so a bounded neighborhood is sufficient
context — and the neighborhood hashes into a window-independent
cache key, making the 50% window overlap nearly free."""

TYPING_MARGIN = 160
"""Characters of context margin around typing states and cache keys.

The typing request state is the source slice covering all candidate
spans plus this margin (short texts degenerate to the whole text);
per-span cache keys use the same margin around each span so that the
same span in different windows hashes identically."""

EDGE_SCORE_MIN = 0.40
"""Detection-score floor for a sub-span's edge tokens (A5b).

Detection already paid for every token's Noul score; an entity's edge
tokens score high, so a candidate sub-span whose first or last token
scored below this floor is almost surely a bad boundary. Free
precision: junk spans never reach a typing question."""

ZH_EDGE_STOP = frozenset(
    "的了是在与及把被着过吗呢吧啊也都就还这那我你他她它或而但对为从向于等之其该各每某再更很太最让跟同以及已又便则"
)
"""Single CJK characters that never open or close an entity name (A5a)."""

EN_EDGE_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "nor",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "these",
        "those",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "my",
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "than",
        "then",
        "also",
        "not",
        "including",
        "such",
        "while",
        "when",
        "where",
    }
)
"""Latin words that never open or close an entity name (A5a).

Prepositions are forbidden at either edge (entity-internal
prepositions like "Bank of America" are untouched — only the EDGE
unit is checked)."""

TYPING_CLUSTER_CHARS = 2000
"""Maximum source-character spread of one typing batch (S2).

Typing batches are position-clustered: a batch closes when its spans
spread wider than this, so each request state stays a local slice
instead of covering every span of a long document."""

NOMINATION_ROUNDS = 3
"""Maximum span-competition nomination rounds (each round removes its
picks and re-asks, surfacing nested entities the previous round's
single answer overshadowed)."""

NOMINATE_TOP = 5
"""Options accepted per nomination round (ranked by probability)."""

NOMINATE_MIN = 0.10
"""Probability floor for a nominated option to be accepted."""

NOMINATE_POOL_MAX = 63
"""Unique surface-form cap for window-wide nomination pools.

At or below it the window nominates as ONE option pool (maximum
competition); above it nomination switches per region so distinct
picks are not capped by the round budget on entity-dense long docs."""

LATENCY_FIRST_MAX_TOKENS = 64
"""Token cap for the latency_first fast path.

The fast path flat-types every viable sub-span of the text in ONE
parallel wave (no detection round trip), so its request count grows
quadratically with token count; at 64 tokens it stays within about
two concurrency waves on the default semaphore. Texts above the cap
silently use the normal pipeline."""
"""Unique surface-form cap for window-wide nomination pools.

At or below it the window nominates as ONE option pool (maximum
competition); above it nomination switches per region so distinct
picks are not capped by the round budget on entity-dense long docs."""
"""Probability floor for a nominated option to be accepted."""

COMPETE_MIN_SPANS = 8
"""Minimum candidate count for competition typing; below it flat
typing every span is cheaper than nominate-then-verify."""

NOMINATE_INSTRUCTIONS: dict[str, str] = {
    "TASK": "Pick the option that is exactly one complete named entity of a type in ENTITY_TYPES.",
    "RULES": (
        "The option must be the complete entity by itself: not a fragment of a "
        "longer name, and not a name with extra words attached (verbs, "
        "prepositions, titles, or modifiers). If no option qualifies, choose NONE."
    ),
}
"""Instructions for span-competition nomination questions (A3)."""

MAX_ENTITY_TOKENS = 12
"""Maximum token length of a sub-span inside an entity region.

Named entities longer than this are extremely rare; sub-span length
is truncated here.
"""

GENDER_BATCH_SIZE = 256
"""Gender questions per request (tiny payloads; no in-batch
interference)."""

GENDER_CACHE_CAPACITY = 1024
"""In-process LRU capacity for gender results."""

GENDER_CRITERIA: dict[str, str] = {
    "MALE": "A name conventionally borne by men.",
    "FEMALE": "A name conventionally borne by women.",
    "UNKNOWN": "No clear gender signal, or a name commonly used for any gender.",
}
"""Option descriptions for the gender Choice questions."""


class _Cache(Protocol):
    """A typed protocol view over vortezwohl LRUCache.

    vortezwohl ships no type annotations; this structural protocol
    pins the two methods this SDK actually depends on, keeping the
    jevner side type-safe. Keys and values are Any; call sites narrow
    them.
    """

    def read(self, key: object) -> object | None: ...

    def write(self, key: object, value: object) -> object | None: ...


def _type_fingerprint(entity_types: tuple[EntityType, ...]) -> str:
    """Hash the full type specs (labels, descriptions, examples, counters).

    Cache keys for typing answers carry this fingerprint so that a
    schema change — new examples, a reworded description, different
    packs — invalidates cached judgments instead of silently reusing
    answers obtained under the old type table.
    """
    blob = repr([(t.label, t.description, t.examples, t.counter_examples) for t in entity_types])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_EDGE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:['’\-.&][A-Za-z0-9]+)*"
    r"|[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]"
)
"""One CJK character or one Latin word/number — the edge unit for span pruning."""


def _is_cjk_char(ch: str) -> bool:
    """True for CJK ideographs, kana, and hangul (entity-text scripts)."""
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def bad_span_edge(text: str) -> bool:
    """True when a span opens or closes on a function word (A5a).

    Entity names never begin or end on a stopword in any script:
    Chinese particles (的了是在...) and Latin function words /
    prepositions (of, the, in...) at an edge mean the span grabbed
    glue. Entity-INTERNAL words are untouched — only the edge units
    are checked, so "Bank of America" and "Ministry of Education"
    survive.

    Args:
        text: The candidate span text.

    Returns:
        True when either edge unit is a stopword (span should be
        dropped before typing).
    """
    units = _EDGE_TOKEN_RE.findall(text)
    if not units:
        return False
    for unit in (units[0], units[-1]):
        if len(unit) == 1 and _is_cjk_char(unit):
            if unit in ZH_EDGE_STOP:
                return True
        elif unit.lower() in EN_EDGE_STOP:
            return True
    return False


def _encode_ranked(ranked: tuple[tuple[EntityType, float], ...]) -> list[list[object]]:
    """Serialize a ranked type tuple to a JSON-safe nested list."""
    return [
        [
            [
                t.label,
                t.description,
                list(t.examples),
                list(t.counter_examples),
                t.min_chars,
                t.include_brackets,
            ],
            p,
        ]
        for t, p in ranked
    ]


def _decode_ranked(blob: list[list[Any]]) -> tuple[tuple[EntityType, float], ...]:
    """Rebuild a ranked type tuple from its JSON-safe nested list."""
    return tuple(
        (
            EntityType(s[0], s[1], tuple(s[2]), tuple(s[3]), int(s[4]), bool(s[5])),
            float(p),
        )
        for s, p in blob
    )


def _flatten_fragments(source: str, entities: list[Entity]) -> list[Entity]:
    """Drop same-type fragments and merge adjacent same-type pairs (A7).

    The standard flat-F1 convention: an entity fully inside another
    entity of the SAME primary type is a boundary fragment ("Tim"
    inside "Tim Cook"), and same-type neighbors separated only by
    non-alphanumerics split one name ("Tim" + "Cook" over a space).
    The default nested output deliberately keeps both sides; this
    opt-in mode serves flat gold standards and review flows.

    Args:
        source: The full source text (gap inspection between spans).
        entities: Merged entities (nesting still flat at this stage).

    Returns:
        The flattened entity list, sorted by (start, end).
    """
    kept: list[Entity] = []
    for e in sorted(entities, key=lambda x: (x.start, -x.end)):
        contained = any(
            o is not e
            and o.type == e.type
            and o.start <= e.start
            and e.end <= o.end
            and (o.end - o.start) > (e.end - e.start)
            for o in entities
        )
        if not contained:
            kept.append(e)
    merged: list[Entity] = []
    for e in sorted(kept, key=lambda x: (x.start, x.end)):
        if merged:
            prev = merged[-1]
            gap = source[prev.end : e.start]
            if prev.type == e.type and prev.end <= e.start and not any(ch.isalnum() for ch in gap):
                merged[-1] = Entity(
                    text=source[prev.start : e.end],
                    types=prev.types,
                    start=prev.start,
                    end=e.end,
                )
                continue
        merged.append(e)
    return merged


class _ResponseLike(Protocol):
    """Anything carrying per-question answers the pipeline reads."""

    @property
    def nouls(self) -> Mapping[str, Any]: ...

    @property
    def choices(self) -> Mapping[str, Any]: ...


@dataclass
class _MergedResponse:
    """A bisected-request recombination: the union of two half answers."""

    nouls: dict[str, Any]
    choices: dict[str, Any]


_OVERSIZE_MARKERS: tuple[str, ...] = (
    "max_tokens",
    "too large",
    "too many",
    "request entity",
    "payload",
    "413",
    "context_length",
)


def _looks_oversize(message: str) -> bool:
    """True when an API error message smells like a size-limit rejection.

    Best-effort marker matching: gateway error wording varies by model
    and revision; anything unrecognized re-raises untouched.
    """
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERSIZE_MARKERS)


class JevNER:
    """A sliding-window multilingual entity recognizer over gateway models.

    Usage mirrors BertNER: construct once, call repeatedly. Credentials
    default to the TYPESAFE_API_KEY / TYPESAFE_BASE_URL environment
    variables and may also be passed explicitly.

    Concurrency: every gateway request is a coroutine on one private
    event loop (a single daemon thread), bounded by an in-flight
    semaphore of max_workers. The synchronous call operator works from
    any thread; async callers should await arecognize().

    Usage accounting: the `usage` attribute accumulates requests,
    questions, cache hits, token counts, and API seconds since
    construction — the basis for cost/latency measurement and A/B
    comparison of pipeline changes.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        typing_strategy: str = "compete",
        disk_cache_path: str | Path | None = None,
        latency_first: bool = False,
        taxonomy: EntityTaxonomy | None = None,
    ) -> None:
        """Initialize the recognizer.

        Args:
            model_name: Gateway model name; defaults to
                'baseten/mammoth-1' (Mammoth 1 on BaseTen; any model
                registered on the gateway works, e.g. 'jev-latest').
            api_key: Gateway API key; falls back to the
                TYPESAFE_API_KEY environment variable.
            base_url: API root; falls back to TYPESAFE_BASE_URL.
            timeout: Per-request timeout in seconds; SDK default when
                None.
            window_size: Sliding-window size in tokens (default 12k,
                derived from the request-body byte budget).
            max_workers: In-flight request limit (the semaphore
                width on the private event loop); 0 or 1 serializes
                requests.
            typing_strategy: "compete" (default) first shortlists
                candidates by span competition (options compete as
                Choice options; rounds remove picks and re-ask to
                surface nested entities), then flat-types the
                shortlist; "flat" types every candidate sub-span
                independently — lower latency, more questions.
            disk_cache_path: Optional sqlite file persisting gateway
                answers across processes (repeat runs cost zero
                requests); None keeps everything in-process only.
            latency_first: Skip the token-detection round trip for
                sentence-scale inputs (<= LATENCY_FIRST_MAX_TOKENS
                tokens): every locally viable sub-span is flat-typed
                in one parallel wave, cutting the serial chain to a
                single round trip. Typing runs flat (compete is
                ignored on the fast path); texts above the token cap
                use the normal pipeline.
            taxonomy: Entity type system (type/pack registries and
                default packs); None uses the built-in system. Consumer
                users can pass EntityTaxonomy.from_json(...) results.

        Raises:
            ValueError: window_size is not positive.
            TypeSafeError: The API key is missing or invalid.
        """
        if window_size < 1:
            raise ValueError(f"window_size ({window_size}) must be positive.")
        # Retry: 3 exponential-backoff attempts on 408/429/5xx and connection errors
        # (the default 30s total budget is too short — transient 500/520s
        # get abandoned early); uncapped, bounded by max_retries
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            timeout=timeout,
            retry=RetryPolicy(max_retries=3, backoff_initial=1.0, backoff_max=8.0, timeout=None),
        )
        self._tokenizer = BundledTokenizer()
        self._window_size = window_size
        self._model_name = model_name
        # Optional persistent answer cache (opt-in; LRU caches sit in front).
        self._disk: DiskCache | None = DiskCache(disk_cache_path) if disk_cache_path else None
        # Entity type system: type/pack registries + default packs, the single
        # source of type semantics; users extend or override via JSON
        self._taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        # In-flight request limit: one asyncio.Semaphore bounds windows,
        # detection batches, recheck batches, typing batches, and gender
        # batches alike — the global rate the old thread pool provided.
        self._max_workers = max(1, max_workers)
        if typing_strategy not in ("flat", "compete"):
            raise ValueError(f'typing_strategy ({typing_strategy!r}) must be "flat" or "compete".')
        self._typing_strategy = typing_strategy
        self._latency_first = bool(latency_first)
        # Private event loop: started lazily on first use, one daemon thread.
        # All coroutines (and the AsyncTypeSafeClient) live on it; both the
        # sync facade and arecognize submit work there, so the HTTP client
        # never crosses event loops (httpx AsyncClient is not loop-safe).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        # Three in-process LRU caches: detection (window -> hit tokens), typing
        # (window + span -> type tuples), gender (name -> judgment).
        # Repeated text / overlapping windows / repeated names all hit cache.
        # Single-threaded under the event loop: no await between read and
        # write, so plain dict-based LRU access needs no locking.
        # Per-key single-flight guard: overlapping windows ask identical
        # detection batches CONCURRENTLY — the first request computes, the
        # rest await the shared future instead of racing the cache.
        self._inflight: dict[object, asyncio.Future[dict[int, float]]] = {}
        self._detect_cache: _Cache = LRUCache(capacity=DETECT_CACHE_CAPACITY)
        self._typing_cache: _Cache = LRUCache(capacity=TYPING_CACHE_CAPACITY)
        self._gender_cache: _Cache = LRUCache(capacity=GENDER_CACHE_CAPACITY)
        # Cumulative usage accounting (requests / questions / cache hits /
        # tokens / API seconds), updated on the private loop; see Usage.
        self.usage = Usage()
        logger.debug(
            "JevNER initialized with model: %s, window_size: %d, max_workers: %d.",
            model_name,
            window_size,
            self._max_workers,
        )

    # ------------------------------------------------------------------
    # Private-loop plumbing
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the private event loop thread once and return its loop.

        All async work (and the shared AsyncTypeSafeClient) must run on
        this single loop: httpx AsyncClient connections are not safe to
        use across event loops, so both the sync facade and arecognize
        submit coroutines here instead of running their own loops.

        Returns:
            The private event loop (running in its daemon thread).
        """
        with self._loop_lock:
            # Gate on is_closed(), never is_running(): the loop thread signals
            # `started` before entering run_forever, so is_running() has a legal
            # False window here and would spuriously spawn a second loop.
            # is_closed() is monotonic (a loop never reopens), and submissions
            # via call_soon_threadsafe queue fine on an open not-yet-running loop.
            if self._loop is None or self._loop.is_closed():
                started = threading.Event()

                def _run_loop() -> None:
                    loop = asyncio.new_event_loop()
                    self._loop = loop
                    # Created here (on the loop thread) so the first acquire
                    # binds it to exactly this loop.
                    self._semaphore = asyncio.Semaphore(self._max_workers)
                    started.set()
                    try:
                        loop.run_forever()
                    finally:
                        loop.close()

                self._loop_thread = threading.Thread(
                    target=_run_loop, name="jevner-event-loop", daemon=True
                )
                self._loop_thread.start()
                if not started.wait(10.0):
                    raise RuntimeError("JevNER private event loop failed to start.")
        assert self._loop is not None and not self._loop.is_closed()
        return self._loop

    def _disk_key(self, key: tuple[Any, ...]) -> tuple[Any, ...]:
        """Prefix a cache key with the model name for the disk layer.

        Answers from one model are never reused for another; the LRU
        layer needs no prefix (one instance runs exactly one model).
        """
        return ("model", self._model_name, *key)

    def _submit(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the private loop; block the caller for the result.

        Safe from any thread; concurrent submissions interleave as
        coroutines on the loop under the shared semaphore.
        """
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def _system_one(
        self,
        *,
        state: str,
        questions: Mapping[str, Noul | Choice],
        extra_headers: Mapping[str, str] | None = None,
    ) -> _ResponseLike:
        """One gateway request under the shared in-flight semaphore.

        The semaphore (max_workers) is the single rate limiter for every
        request this instance issues, mirroring the global bound the old
        thread pool provided.

        Oversize self-healing (S5c): a size-limit rejection (token or
        payload cap, best-effort matched) splits the question set in
        half and both halves retry recursively — a runtime safety net
        behind the static request budgets, so one unexpectedly large
        state degrades into more requests instead of a hard failure.
        """
        assert self._semaphore is not None, "private loop must exist before any request"
        async with self._semaphore:
            start = time.perf_counter()
            try:
                response = await self._client.system_one(
                    state=state, questions=questions, extra_headers=extra_headers
                )
            except TypeSafeAPIError as exc:
                # Anything else re-raises; an oversize-looking rejection with
                # more than one question falls through to the bisect below.
                if len(questions) < 2 or not _looks_oversize(str(exc)):
                    raise
            else:
                usage = self.usage
                usage.requests += 1
                usage.questions += len(questions)
                usage.seconds += time.perf_counter() - start
                if response.usage.input_tokens is not None:
                    usage.input_tokens += response.usage.input_tokens
                if response.usage.output_tokens is not None:
                    usage.output_tokens += response.usage.output_tokens
                return response
        # Reached only on the oversize path (the semaphore is released —
        # the recursive halves acquire their own slots).
        items = list(questions.items())
        mid = len(items) // 2
        first = await self._system_one(
            state=state, questions=dict(items[:mid]), extra_headers=extra_headers
        )
        second = await self._system_one(
            state=state, questions=dict(items[mid:]), extra_headers=extra_headers
        )
        logger.warning(
            "JevNER request exceeded a size limit; bisected %d questions into %d + %d.",
            len(questions),
            mid,
            len(questions) - mid,
        )
        return _MergedResponse(
            nouls={**dict(first.nouls), **dict(second.nouls)},
            choices={**dict(first.choices), **dict(second.choices)},
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def _detect_batch(
        self,
        source: str,
        tokens: list[tuple[int, int, str]],
        pairs: list[tuple[int, tuple[int, int, str]]],
    ) -> dict[int, float]:
        """Ask one detection batch (one Noul per token); returns index -> score.

        Each token question carries its neighbors: whether an isolated
        single character belongs to an entity is context-dependent (a
        lone surname is undecidable; with neighbors it is), giving
        detection region vision. Failures propagate to the caller — a
        failed detection batch leaves regions unknowable, so the whole
        call aborts rather than returning silently partial entities.

        The request state is a NEIGHBORHOOD of the batch (tokens
        [first - DETECT_NEIGHBORHOOD, last + DETECT_NEIGHBORHOOD])
        rather than the whole window: every batch of a window re-sent
        the full window text, which dominated input-token cost on long
        documents (~188 batches x up to 48KB). Token questions carry
        their immediate LEFT/RIGHT neighbors inline, so a bounded
        neighborhood is sufficient context for the judgment.

        Answers cache by (neighborhood hash, relative positions) —
        window-independent, so the 50% window overlap and repeated
        passages hit cache instead of re-asking.

        Args:
            source: The full source text (token offsets are global;
                the neighborhood is sliced from it).
            tokens: All window tokens (neighbor lookup).
            pairs: (index, token) pairs for this batch.

        Returns:
            Token index -> Noul score for the answered questions.
        """
        first, last = pairs[0][0], pairs[-1][0]
        lo = max(0, first - DETECT_NEIGHBORHOOD)
        hi = min(len(tokens), last + 1 + DETECT_NEIGHBORHOOD)
        nb_text = source[tokens[lo][0] : tokens[hi - 1][1]]
        nb_hash = hashlib.sha256(nb_text.encode("utf-8")).hexdigest()
        rel_positions = tuple(i - lo for i, _tok in pairs)
        cache_key = ("nb", nb_hash, rel_positions)
        cached = self._detect_cache.read(cache_key)
        if cached is None and self._disk is not None:
            blob = self._disk.read(self._disk_key(cache_key))
            if blob is not None:
                cached = {int(k): float(v) for k, v in blob.items()}
                self._detect_cache.write(cache_key, cached)
        if cached is not None:
            self.usage.cache_hits += len(pairs)
            cached_rel = cast("dict[int, float]", cached)
            return {i: cached_rel[i - lo] for i, _tok in pairs}
        # Single-flight: an in-flight identical batch (overlapping window)
        # shares its computation instead of issuing a duplicate request.
        existing = self._inflight.get(cache_key)
        if existing is not None:
            # REL-keyed shared result: both windows map through their own lo.
            shared = await asyncio.shield(existing)
            self.usage.cache_hits += len(pairs)
            return {i: shared[i - lo] for i, _tok in pairs if (i - lo) in shared}
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[int, float]] = loop.create_future()
        self._inflight[cache_key] = future
        try:
            rel_result = await self._detect_batch_ask(nb_text, tokens, pairs, lo, cache_key)
        except BaseException as exc:
            future.set_exception(exc)
            del self._inflight[cache_key]
            raise
        future.set_result(rel_result)
        del self._inflight[cache_key]
        return {i: rel_result[i - lo] for i, _tok in pairs if (i - lo) in rel_result}

    async def _detect_batch_ask(
        self,
        nb_text: str,
        tokens: list[tuple[int, int, str]],
        pairs: list[tuple[int, tuple[int, int, str]]],
        lo: int,
        cache_key: tuple[object, ...],
    ) -> dict[int, float]:
        """Issue one detection batch request and write it to the caches."""
        state = render_detection_state(nb_text)
        questions: dict[str, Noul] = {
            f"tok_{i - lo}": Noul(
                instructions={
                    "TOKEN": tok[2].strip(),
                    "POSITION": i - lo,
                    "LEFT": tokens[i - 1][2].strip() if i > 0 else "",
                    "RIGHT": tokens[i + 1][2].strip() if i + 1 < len(tokens) else "",
                }
            )
            for i, tok in pairs
        }
        response = await self._system_one(
            state=state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
        )
        rel_scores: dict[int, float] = {}
        for i, _tok in pairs:
            answer = response.nouls.get(f"tok_{i - lo}")
            if answer is None:
                logger.warning("JevNER got no detection answer for tok_%d, skipped.", i)
                continue
            rel_scores[i - lo] = float(answer.noul)
        if rel_scores:
            self._detect_cache.write(cache_key, rel_scores)
            if self._disk is not None:
                self._disk.write(
                    self._disk_key(cache_key), {str(k): v for k, v in rel_scores.items()}
                )
        # REL-keyed: single-flight waiters map through their own window-local lo.
        return rel_scores

    async def _detect_regions(
        self, source: str, window_text: str, tokens: list[tuple[int, int, str]], base: int = 0
    ) -> tuple[
        list[tuple[int, int, str]], list[int], list[list[tuple[int, int, str]]], dict[int, float]
    ]:
        """Detect entity token regions within one window.

        Submits one Noul question per token ("is this token part of a
        named entity?"), filters hits by threshold, and reassembles
        them into regions by positional adjacency. This mirrors the
        token-level prediction of the BertNER base model and treats
        every tokenization shape alike. Results are cached at two
        levels (whole-window fast path keyed by window hash; per-batch
        neighborhood answers keyed window-independently); repeated
        windows and the 50% window overlap cost zero requests.

        Concurrency: every batch of the first pass is independent, so
        all fly at once (the former sequential loop cost one round
        trip per 64 tokens — ~188 serialized requests on a full 12k
        window). The recheck rounds resample the same gray-zone token
        set, so they also fly concurrently; taking the max over the
        concurrent samples is exactly the old sequential-rounds result
        (only tokens that would have passed early are re-asked).

        Args:
            source: The full source text (token offsets are global).
            window_text: The window text (an exact source substring;
                keys the whole-window fast path).
            tokens: Window tokens with offsets anchored to the source.
            base: This window's first token index in the full-source token
                sequence; detection batches align on ABSOLUTE positions
                (boundaries at (base+i) % DETECT_BATCH_SIZE == 0), so
                overlapping windows produce identical neighborhoods and
                the neighborhood cache dedups the 50% overlap.

        Returns:
            (all window tokens, hit token indices, strict contiguous
            regions, per-token Noul scores); tokens carry source-global
            character offsets. Scores ride along so the typing stage
            can use edge-token confidence as a free boundary prior.
        """
        if not tokens:
            return [], [], [], {}
        cache_key = hashlib.sha256(window_text.encode("utf-8")).hexdigest()
        cached = self._detect_cache.read(cache_key)
        if cached is not None:
            # A window hit skips every token question in the window.
            self.usage.cache_hits += len(tokens)
            return cast(
                "tuple[list[tuple[int, int, str]], list[int],"
                " list[list[tuple[int, int, str]]], dict[int, float]]",
                cached,
            )
        # First pass: all batches independent -> all in flight at once.
        # Batch boundaries sit at absolute token indices that are multiples
        # of DETECT_BATCH_SIZE (not at every window's local zero), so a
        # token questioned in two overlapping windows lands in the same
        # batch, same neighborhood, same cache key exactly once.
        first_pass: list[list[tuple[int, tuple[int, int, str]]]] = []
        pending_batch: list[tuple[int, tuple[int, int, str]]] = []
        for i, tok in enumerate(tokens):
            if pending_batch and (base + i) % DETECT_BATCH_SIZE == 0:
                first_pass.append(pending_batch)
                pending_batch = []
            pending_batch.append((i, tok))
        if pending_batch:
            first_pass.append(pending_batch)
        scores: dict[int, float] = {}
        first_results: list[dict[int, float]] = list(
            await asyncio.gather(
                *(self._detect_batch(source, tokens, pairs) for pairs in first_pass)
            )
        )
        for batch_scores in first_results:
            scores.update(batch_scores)
        # Gray-zone recheck (DETECT_RECHECK_ROUNDS independent resamples of the
        # same token set, keep max): the live API has per-request sampling
        # variance, and the same state is stable across rounds. Rounds need no
        # ordering — every (round x batch) task flies concurrently.
        gray = [
            i for i, score in scores.items() if DETECT_RECHECK_LOW <= score < DETECT_PROB_THRESHOLD
        ]
        if gray:
            recheck_tasks: list[Coroutine[Any, Any, dict[int, float]]] = []
            for _ in range(DETECT_RECHECK_ROUNDS):
                for raw_batch in batched(gray, DETECT_BATCH_SIZE):
                    indices = [i for i in raw_batch if i is not None]
                    if not indices:
                        continue
                    recheck_tasks.append(
                        self._detect_batch(source, tokens, [(i, tokens[i]) for i in indices])
                    )
            recheck_results: list[dict[int, float]] = list(await asyncio.gather(*recheck_tasks))
            for resample in recheck_results:
                for i, score in resample.items():
                    scores[i] = max(scores[i], score)
        hits = sorted(i for i, score in scores.items() if score >= DETECT_PROB_THRESHOLD)
        # Reassemble hits into strictly contiguous regions (strategy 1 of 2).
        # Strict contiguity keeps adjacent entities from gluing together:
        # each becomes its own region, no swallowed preposition spans.
        regions: list[list[tuple[int, int, str]]] = []
        current: list[tuple[int, int, str]] = []
        prev: int | None = None
        for i in hits:
            if current and prev is not None and i == prev + 1:
                current.append(tokens[i])
            else:
                if current:
                    regions.append(current)
                current = [tokens[i]]
            prev = i
        if current:
            regions.append(current)
        logger.debug("JevNER detected %d strict region(s).", len(regions))
        result = (tokens, hits, regions, scores)
        self._detect_cache.write(cache_key, result)
        return result

    @staticmethod
    def _bridge_regions(
        tokens: list[tuple[int, int, str]],
        hits: list[int],
        regions: list[list[tuple[int, int, str]]],
    ) -> list[list[tuple[int, int, str]]]:
        """Bridge hit tokens with gap tolerance into a second region set.

        Strict contiguity slices intra-word tokens out of finely
        segmented scripts (Thai, Arabic), where tone marks and affixes
        are individually voted down, breaking word-internal entities
        into fragments. Bridged regions tolerate a few missed tokens
        (DETECT_GAP_TOLERANCE) and merge the crossed tokens' character
        ranges in. The two region sets union in the typing stage:
        strict regions keep boundaries, bridged regions keep
        word-internal entities whole, and glued junk spans are
        filtered out by the typing NONE veto.

        Args:
            tokens: All window tokens.
            hits: Hit token indices (ascending).
            regions: Strict regions (unused; kept for signature
                symmetry with the former call site).

        Returns:
            Bridged regions; unioned with the strict regions.
        """
        bridged: list[list[tuple[int, int, str]]] = []
        current: list[tuple[int, int, str]] = []
        last_hit: int | None = None
        for i in hits:
            missed = i - last_hit - 1 if last_hit is not None else 0
            if current and last_hit is not None and missed <= DETECT_GAP_TOLERANCE:
                # Contiguous (missed == 0) extends the region; misses within tolerance
                # bridge the missed tokens into the region
                current.extend(tokens[last_hit + 1 : i])
                current.append(tokens[i])
            else:
                # Misses beyond tolerance: flush the current region and start a new one
                if current:
                    bridged.append(current)
                current = [tokens[i]]
            last_hit = i
        if current:
            bridged.append(current)
        return bridged

    # ------------------------------------------------------------------
    # Typing
    # ------------------------------------------------------------------

    async def _type_window(
        self,
        source: str,
        window_text: str,
        tokens: list[tuple[int, int, str]],
        base: int,
        entity_types: tuple[EntityType, ...],
        prob_threshold: float,
        skip_detection: bool = False,
    ) -> list[Entity]:
        """Run detection and typing for one window; returns its entities.

        Detection yields the union of two region strategies (strict
        contiguity + gap bridging); every contiguous sub-span within
        each region (at most MAX_ENTITY_TOKENS tokens long, nesting
        included) becomes one Choice question batched for typing.
        Typing batches fly concurrently under the shared semaphore.

        The typing request state is the source slice covering all
        candidate spans plus a fixed margin (not the whole window) —
        span judgments are local, and the slice keeps request bodies
        small on long windows. Answers cache by (span context slice,
        span text, threshold, full type fingerprint): the key depends
        only on the span and its schema, never the window, so
        overlapping windows and repeated runs hit cache.

        Typing supports multiple types: Choice probabilities are a
        competitive distribution; the selected option must clear
        prob_threshold as the primary type, and remaining types at or
        above SECONDARY_TYPE_THRESHOLD ride along, all ranked by
        probability — a single type is never forced.

        Args:
            source: The full source text (token offsets are global;
                spans slice from it, fixing multi-window offsets).
            window_text: The window text (an exact source substring;
                keys the detection fast path).
            tokens: Window tokens with offsets anchored to the source.
            entity_types: The enabled entity types for this pass.
            prob_threshold: Primary-type probability floor.

        Returns:
            Entities recognized in this window (not yet merged across
            windows), offsets anchored to the source.
        """
        if skip_detection:
            # Fast path: the whole window is one candidate region; no
            # detection round trip, so no edge-score prior either.
            hits: list[int] = []
            strict_regions = [tokens]
            scores: dict[int, float] = {}
            edge_score_min = 0.0
        else:
            _, hits, strict_regions, scores = await self._detect_regions(
                source, window_text, tokens, base
            )
            edge_score_min = EDGE_SCORE_MIN
        # Token offset pair -> Noul score: the free boundary prior for span
        # pruning (an entity's edge tokens score high; weak edges are junk).
        score_by_offset = {(tok[0], tok[1]): scores.get(idx, 0.0) for idx, tok in enumerate(tokens)}
        all_regions: list[list[tuple[int, int, str]]] = list(strict_regions)
        for region in self._bridge_regions(tokens, hits, strict_regions):
            if region not in all_regions:
                all_regions.append(region)
        if not all_regions:
            return []
        # Generate every contiguous sub-span in each region; edge punctuation
        # stripping moves offsets with it so span text stays an exact substring.
        # Offsets index the SOURCE: region token offsets are source-global, and
        # slicing window_text here would clamp garbage on every window that
        # does not start at source position 0.
        spans: list[tuple[str, int, int]] = []
        region_ids: list[int] = []
        seen: set[tuple[str, int, int]] = set()
        for rid, region in enumerate(all_regions):
            n = len(region)
            for i in range(n):
                for length in range(1, min(MAX_ENTITY_TOKENS, n - i) + 1):
                    raw_start = region[i][0]
                    raw_end = region[i + length - 1][1]
                    text, start, end = strip_span(source, raw_start, raw_end)
                    key = text.strip()
                    # Skip blank, duplicate, or punctuation-only candidates
                    if not key or (text, start, end) in seen:
                        continue
                    if not any(ch.isalnum() for ch in key):
                        continue
                    # Local pruning (A5): stopword edges are almost never entity
                    # boundaries, and a sub-span whose edge token scored weakly
                    # in detection is a bad boundary — both cost zero API calls.
                    if bad_span_edge(text):
                        continue
                    if (
                        score_by_offset.get((region[i][0], region[i][1]), 0.0) < edge_score_min
                        or score_by_offset.get(
                            (region[i + length - 1][0], region[i + length - 1][1]), 0.0
                        )
                        < edge_score_min
                    ):
                        continue
                    seen.add((text, start, end))
                    spans.append((text, start, end))
                    region_ids.append(rid)
        if not spans:
            return []
        # Request state: the source slice covering all spans ± margin (equal to
        # the whole text for short inputs). Shared by nomination and typing.
        cover_lo = min(sp[1] for sp in spans)
        cover_hi = max(sp[2] for sp in spans)
        typing_text = source[
            max(0, cover_lo - TYPING_MARGIN) : min(len(source), cover_hi + TYPING_MARGIN)
        ]
        # Competition typing (A3): spans compete as Choice options first; the
        # shortlist goes through the same flat typing below. Skipped for small
        # candidate sets where flat typing everything is cheaper.
        if self._typing_strategy == "compete" and not skip_detection:
            keep = await self._compete_nominate(typing_text, spans, region_ids, entity_types)
            if not any(keep):
                return []
            spans = [sp for sp, k in zip(spans, keep, strict=True) if k]
        # Typing cache hits: skip spans already judged (window overlap / repeats).
        # Keys are window-independent (span context slice + text + threshold +
        # full type fingerprint): the same span under a different type table —
        # different packs, a schema A/B, a changed description or example —
        # is a different question and must not reuse answers.
        fingerprint = _type_fingerprint(entity_types)

        def _span_cache_key(text: str, start: int, end: int) -> tuple[object, ...]:
            ctx = source[max(0, start - TYPING_MARGIN) : min(len(source), end + TYPING_MARGIN)]
            return (
                "type",
                hashlib.sha256(ctx.encode("utf-8")).hexdigest(),
                text,
                prob_threshold,
                fingerprint,
            )

        uncached: list[tuple[int, str]] = []
        cached_entities: dict[int, tuple[tuple[EntityType, float], ...]] = {}
        for i, (text, start, end) in enumerate(spans):
            cache_key = _span_cache_key(text, start, end)
            hit = self._typing_cache.read(cache_key)
            if hit is None and self._disk is not None:
                blob = self._disk.read(self._disk_key(cache_key))
                if blob is not None:
                    hit = _decode_ranked(blob)
                    self._typing_cache.write(cache_key, hit)
            if hit is not None:
                self.usage.cache_hits += 1
                cached_entities[i] = cast("tuple[tuple[EntityType, float], ...]", hit)
            else:
                uncached.append((i, text))
        entities: list[Entity] = []
        # Criteria carry labels only: descriptions already appear once in the
        # state ENTITY_TYPES table (repeating them per question balloons a
        # question from ~150 bytes to ~2KB; 64 per batch ~= 128KB, past the
        # 65,535-byte hard cap — the observed stable 500s came from exactly this).
        # Labels match the type-table keys; the model resolves them by lookup.
        criteria: dict[str, str | None] = {t.label: None for t in entity_types}
        criteria[NONE_LABEL] = NONE_DESCRIPTION
        criteria[PARTIAL_LABEL] = PARTIAL_DESCRIPTION
        criteria[MIXED_LABEL] = MIXED_DESCRIPTION
        # Answer decoding maps labels against the RESOLVED types of this call,
        # never the taxonomy registry: types passed via entity_types (or a
        # custom taxonomy) are legitimate questions whose labels are not
        # registered, and the model must answer from the criteria labels anyway.
        label_map: dict[str, EntityType] = {t.label: t for t in entity_types}

        async def _type_batch(
            pairs: list[tuple[int, str]],
            batch_state: str,
        ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
            """One typing batch request: map span index -> ranked type tuple;
            an empty tuple means NONE."""
            questions: dict[str, Choice] = {
                f"type_{i}": Choice(instructions={"SPAN": cand}, criteria=criteria)
                for i, cand in pairs
            }
            response = await self._system_one(
                state=batch_state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
            )
            batch: dict[int, tuple[tuple[EntityType, float], ...]] = {}
            for i, _cand in pairs:
                answer = response.choices.get(f"type_{i}")
                if answer is None:
                    logger.warning("JevNER got no typing answer for type_%d, skipped.", i)
                    batch[i] = ()
                    continue
                # Veto: NONE (not an entity), PARTIAL (a truncated
                # fragment), or MIXED (a name with glue words) all kill the
                # span — no secondary types collected, otherwise fragments
                # slip in through low-probability secondary probabilities
                # bypassing the veto (observed with 0.25-0.5 fragment noise)
                if answer.choice in VETO_LABELS:
                    batch[i] = ()
                    continue
                # The primary type (selected option) must clear prob_threshold;
                # others at or above SECONDARY_TYPE_THRESHOLD ride along as secondary
                best_per_type: dict[EntityType, float] = {}
                primary_prob = float(answer.probabilities.get(answer.choice, 0.0))
                if primary_prob < prob_threshold:
                    batch[i] = ()
                    continue
                primary = label_map.get(answer.choice)
                if primary is None:
                    logger.warning(
                        "JevNER got unexpected typing label %r, treated as NONE.", answer.choice
                    )
                    batch[i] = ()
                    continue
                best_per_type[primary] = primary_prob
                for label, prob in answer.probabilities.items():
                    # Veto labels are not entity types; excluded from collection
                    if label in VETO_LABELS or label == answer.choice:
                        continue
                    if prob < SECONDARY_TYPE_THRESHOLD:
                        continue
                    etype = label_map.get(label)
                    if etype is None:
                        continue
                    if etype not in best_per_type or prob > best_per_type[etype]:
                        best_per_type[etype] = float(prob)
                batch[i] = tuple(sorted(best_per_type.items(), key=lambda kv: -kv[1]))
            return batch

        # Typing batches run concurrently under the semaphore; batch size narrows
        # by option budget: options per question = types + 1 (NONE), questions per
        # batch <= REQUEST_MAX_OPTIONS / options per question, capped by
        # TYPING_BATCH_SIZE (the gateway caps total options per request at 512; 422
        # beyond). batched() pads the last batch with None; filter them out.
        options_per_question = len(entity_types) + len(VETO_LABELS)
        batch_limit = min(TYPING_BATCH_SIZE, REQUEST_MAX_OPTIONS // options_per_question)
        uncached_sorted = sorted(uncached, key=lambda pair: spans[pair[0]][1])
        batches: list[list[tuple[int, str]]] = []
        current_batch: list[tuple[int, str]] = []
        cluster_lo = 0
        for i, text in uncached_sorted:
            _t, start, end = spans[i]
            lo = max(0, start - TYPING_MARGIN)
            hi = min(len(source), end + TYPING_MARGIN)
            if current_batch and (
                len(current_batch) >= max(1, batch_limit) or hi - cluster_lo > TYPING_CLUSTER_CHARS
            ):
                batches.append(current_batch)
                current_batch = []
                cluster_lo = lo
            if not current_batch:
                cluster_lo = lo
            current_batch.append((i, text))
        if current_batch:
            batches.append(current_batch)
        # One state per batch: the slice covering that batch's spans ± margin —
        # request states stay local even when spans scatter across a long doc.
        batch_states: list[str] = []
        for batch in batches:
            lo = max(0, spans[batch[0][0]][1] - TYPING_MARGIN)
            hi = min(len(source), spans[batch[-1][0]][2] + TYPING_MARGIN)
            batch_states.append(render_typing_state(source[lo:hi], entity_types))
        if len(batches) <= 1 or self._max_workers <= 1:
            batch_results = [
                await _type_batch(pairs, bstate)
                for pairs, bstate in zip(batches, batch_states, strict=True)
            ]
        else:
            # Per-batch resilience (thread-pool-era semantics): a failed batch
            # logs its traceback and yields {} (spans fall back to NONE).
            async def _guarded_batch(
                pairs: list[tuple[int, str]],
                bstate: str,
            ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
                try:
                    return await _type_batch(pairs, bstate)
                except Exception as exc:
                    logger.warning(
                        "JevNER typing batch failed: %s\n%s", exc, traceback.format_exc()
                    )
                    return {}

            batch_results = list(
                await asyncio.gather(
                    *(
                        _guarded_batch(pairs, bstate)
                        for pairs, bstate in zip(batches, batch_states, strict=True)
                    )
                )
            )
        for pairs, result in zip(batches, batch_results, strict=True):
            for i, _text in pairs:
                _t, start, end = spans[i]
                span_key = _span_cache_key(_t, start, end)
                ranked = result.get(i, ())
                self._typing_cache.write(span_key, ranked)
                if self._disk is not None:
                    self._disk.write(self._disk_key(span_key), _encode_ranked(ranked))
        # Combine: cache hits + this round; NONE (empty tuple) yields no entity
        all_ranked: dict[int, tuple[tuple[EntityType, float], ...]] = dict(cached_entities)
        for result in batch_results:
            all_ranked.update(result)
        for i, (text, start, end) in enumerate(spans):
            ranked = all_ranked.get(i, ())
            if ranked:
                entities.append(Entity(text=text, types=ranked, start=start, end=end))
        return entities

    async def _compete_nominate(
        self,
        typing_text: str,
        spans: list[tuple[str, int, int]],
        region_ids: list[int],
        entity_types: tuple[EntityType, ...],
    ) -> list[bool]:
        """Shortlist candidate spans by span competition, region by region (A3).

        Candidate span texts become the options of a nomination Choice
        ("which option is exactly one complete entity?"): probability
        mass is shared, so a complete name beats its own fragments —
        fragments no longer pass on a lonely 0.5 the way independent
        questions allowed. Each round removes its picks and re-asks,
        which surfaces nested entities the previous single answer
        overshadowed ("江苏" after "江苏南京").

        Nomination runs PER REGION: region option pools stay small
        (a whole-window pool capped the picks per window and broke
        recall on entity-dense long documents), regions nominate
        concurrently, and regions too small to be worth a nomination
        pass through to flat typing directly. The shortlist then goes
        through ordinary flat typing (vetoes, thresholds, multi-type
        collection) unchanged.

        Args:
            typing_text: The typing-neighborhood state text.
            spans: Candidate (text, start, end) triples.
            region_ids: Region index per span (aligned with spans).
            entity_types: The enabled entity types for this pass.

        Returns:
            A keep mask aligned with spans (True = keep for typing).
        """
        # Hybrid pool sizing: a window whose UNIQUE surface forms fit one
        # nomination question competes window-wide (maximum competition beat
        # per-region pools on short texts); larger windows nominate per region
        # (a window-wide pool caps distinct picks and breaks recall on
        # entity-dense long documents).
        if len({sp[0] for sp in spans}) <= NOMINATE_POOL_MAX:
            picked = await self._nominate_region(
                typing_text, list(range(len(spans))), spans, entity_types
            )
            picked_set = set(picked)
            return [i in picked_set for i in range(len(spans))]
        by_region: dict[int, list[int]] = {}
        for idx, rid in enumerate(region_ids):
            by_region.setdefault(rid, []).append(idx)
        keep = [False] * len(spans)
        tasks = []
        for indices in by_region.values():
            if len(indices) < COMPETE_MIN_SPANS:
                for i in indices:
                    keep[i] = True
            else:
                tasks.append(self._nominate_region(typing_text, indices, spans, entity_types))
        results: list[list[int]] = list(await asyncio.gather(*tasks))
        for picked in results:
            for i in picked:
                keep[i] = True
        return keep

    async def _nominate_region(
        self,
        typing_text: str,
        indices: list[int],
        spans: list[tuple[str, int, int]],
        entity_types: tuple[EntityType, ...],
    ) -> list[int]:
        """Nominate inside one region's option pool; returns kept span indices."""
        options: dict[str, None] = {}
        for i in indices:
            options.setdefault(spans[i][0])
        picked_texts: set[str] = set()
        for _round in range(NOMINATION_ROUNDS):
            keys = [k for k in options if k not in picked_texts]
            if not keys:
                break
            probs = await self._nominate_ask(typing_text, keys, entity_types)
            ranked = sorted(
                ((k, v) for k, v in probs.items() if v >= NOMINATE_MIN),
                key=lambda kv: -kv[1],
            )
            fresh = [k for k, _v in ranked[:NOMINATE_TOP] if k not in picked_texts]
            if not fresh:
                break
            picked_texts.update(fresh)
        return [i for i in indices if spans[i][0] in picked_texts]

    async def _nominate_ask(
        self,
        typing_text: str,
        keys: list[str],
        entity_types: tuple[EntityType, ...],
    ) -> dict[str, float]:
        """One nomination round: every option chunk asked concurrently.

        Args:
            typing_text: The typing-neighborhood state text.
            keys: Candidate option texts for this round.
            entity_types: The enabled entity types (fingerprint keys the
                answer cache together with the option set).

        Returns:
            Option text -> nomination probability (missing options
            never qualified and stay absent).
        """
        cache_key = (
            "nom",
            hashlib.sha256(typing_text.encode("utf-8")).hexdigest(),
            tuple(keys),
            _type_fingerprint(entity_types),
        )
        cached = self._typing_cache.read(cache_key)
        if cached is None and self._disk is not None:
            blob = self._disk.read(self._disk_key(cache_key))
            if blob is not None:
                cached = {str(k): float(v) for k, v in blob.items()}
                self._typing_cache.write(cache_key, cached)
        if cached is not None:
            self.usage.cache_hits += 1
            return cast("dict[str, float]", cached)
        state = render_nomination_state(typing_text, entity_types)
        # Options per question <= 63 (+NONE): keeps every question inside the
        # per-question option budget with headroom for the gateway's 512 total.
        chunk_size = min(len(keys), REQUEST_MAX_OPTIONS // 8)
        chunks = [
            [k for k in chunk if k is not None] for chunk in batched(keys, max(1, chunk_size))
        ]
        chunks = [chunk for chunk in chunks if chunk]

        async def _ask(chunk_keys: list[str]) -> dict[str, float]:
            criteria: dict[str, str | None] = {k: None for k in chunk_keys}
            criteria[NONE_LABEL] = None
            response = await self._system_one(
                state=state,
                questions={"nom": Choice(instructions=NOMINATE_INSTRUCTIONS, criteria=criteria)},
                extra_headers=LONG_CONTEXT_HEADER,
            )
            answer = response.choices.get("nom")
            if answer is None:
                logger.warning("JevNER got no nomination answer, chunk skipped.")
                return {}
            return {k: float(answer.probabilities.get(k, 0.0)) for k in chunk_keys}

        results: list[dict[str, float]] = list(await asyncio.gather(*(_ask(c) for c in chunks)))
        merged: dict[str, float] = {}
        for result in results:
            merged.update(result)
        self._typing_cache.write(cache_key, merged)
        if self._disk is not None:
            self._disk.write(self._disk_key(cache_key), merged)
        return merged

    def _combine(self, entities: list[Entity], deduplicate: bool) -> list[Entity]:
        """Merge entities across windows (per-occurrence dedup + type union).

        Rules:

        - Same occurrence (identical start/end): repeated recognitions
          from overlapping windows or batches merge into one entity,
          taking the union of types and the max probability per type;
        - Different occurrences: kept independently — spans in a
          containment relation are nested entities (e.g. GPE "江苏"
          [16:18] inside GPE "江苏南京" [16:20]); the merge stage never
          absorbs or drops either side, and relations are derived
          entirely downstream by entity_tree() from offset containment.

        Fragment noise (detection-variance artifacts like PERSON single-character fragments)
        is not intercepted here — the typing stage NONE veto filters
        it at the source.

        Args:
            entities: Flat entity list from all windows.
            deduplicate: Whether to merge same-occurrence duplicates
                (type union).

        Returns:
            Merged entities sorted by (start, end); each entity types
            ranked by probability, text an exact source substring.
        """
        if not deduplicate:
            return sorted(entities, key=lambda e: (e.start, e.end, e.text))
        # Group by occurrence: same-position judgments across windows/batches merge
        # into one (type union, max probability per type); distinct positions stay independent
        by_span: dict[tuple[int, int], list[Entity]] = {}
        for e in entities:
            by_span.setdefault((e.start, e.end), []).append(e)
        merged: list[Entity] = []
        for (start, end), group in sorted(by_span.items()):
            best_per_type: dict[EntityType, float] = {}
            text = group[0].text
            for e in group:
                for etype, prob in e.types:
                    if etype not in best_per_type or prob > best_per_type[etype]:
                        best_per_type[etype] = prob
            ranked = tuple(sorted(best_per_type.items(), key=lambda kv: -kv[1]))
            merged.append(Entity(text=text, types=ranked, start=start, end=end))
        return merged

    # ------------------------------------------------------------------
    # Gender
    # ------------------------------------------------------------------

    async def _recognize_genders(self, names: list[str]) -> dict[str, tuple[Gender, float]]:
        """Batch-judge name gender (with the in-process cache).

        Args:
            names: Name list (duplicates allowed; results keyed).

        Returns:
            Mapping from name to (gender, probability).
        """
        results: dict[str, tuple[Gender, float]] = {}
        pending: list[str] = []
        for name in names:
            cached = self._gender_cache.read(name)
            if cached is None and self._disk is not None:
                blob = self._disk.read(self._disk_key(("gender", name)))
                if blob is not None:
                    cached = (Gender(blob[0]), float(blob[1]))
                    self._gender_cache.write(name, cached)
            if cached is not None:
                self.usage.cache_hits += 1
                results[name] = cast("tuple[Gender, float]", cached)
            elif name not in pending:
                pending.append(name)
        if not pending:
            return results
        state = render_gender_state()
        for raw_batch in batched(pending, GENDER_BATCH_SIZE):
            batch = [n for n in raw_batch if n is not None]
            if not batch:
                continue
            questions: dict[str, Choice] = {
                f"gender_{i}": Choice(instructions={"NAME": name}, criteria=GENDER_CRITERIA)
                for i, name in enumerate(batch)
            }
            response = await self._system_one(state=state, questions=questions)
            for i, name in enumerate(batch):
                answer = response.choices.get(f"gender_{i}")
                if answer is None:
                    logger.warning("JevNER got no answer for gender_%d, skipped.", i)
                    continue
                prob = float(answer.probabilities.get(answer.choice, 0.0))
                try:
                    gender = Gender(answer.choice)
                except ValueError:
                    logger.warning("JevNER got unexpected gender label %r, dropped.", answer.choice)
                    continue
                pair = (gender, prob)
                self._gender_cache.write(name, pair)
                if self._disk is not None:
                    self._disk.write(self._disk_key(("gender", name)), [gender.value, prob])
                results[name] = pair
        return results

    # ------------------------------------------------------------------
    # Recognition entry points
    # ------------------------------------------------------------------

    async def _arecognize(
        self,
        s: str,
        *,
        with_gender: bool,
        entity_packs: Iterable[EntityPack | str] | None,
        entity_types: Iterable[EntityType] | None,
        prob_threshold: float,
        fast_mode: bool,
        window_size: int | None,
        stride: int | None,
        deduplicate: bool,
        flatten_fragments: bool,
        latency_first: bool | None,
    ) -> list[EntityNode]:
        """Async recognition core (the whole pipeline); parameter meanings
        and exceptions are documented on __call__."""
        if not isinstance(s, str) or not s.strip():
            return []
        assert 0.0 <= prob_threshold <= 1.0, (
            f"prob_threshold ({prob_threshold}) must be within [0, 1]."
        )
        # Type resolution: packs + explicit types (taxonomy is the sole source;
        # PERSON and other fixed semantic labels live in the taxonomy)
        types = self._taxonomy.resolve(entity_packs, entity_types)
        ws = window_size if window_size is not None else self._window_size
        st = stride if stride is not None else max(1, ws // DEFAULT_STRIDE_DIVISOR)
        if ws < 1 or st < 1:
            raise ValueError(f"window_size ({ws}) and stride ({st}) must be positive.")
        # Windows are cut on source token offsets; window text is sliced from the
        # source character range (space-prefixed tokens overlap; re-joining would
        # duplicate chars), so token offsets stay source-anchored — spans and
        # entities carry source-global offsets, nesting derives from containment
        if fast_mode:
            windows = [(s, self._tokenizer.window_tokens(s), 0)]
        else:
            windows = self._tokenizer.windows(s, ws, st)
        # Request-body guard: window text dominates the body; over budget fails fast
        # (an oversized request returns an opaque 500 rather than an actionable
        # error, so this raises with actionable guidance)
        for window_text, _toks, _base in windows:
            if len(window_text.encode("utf-8")) > REQUEST_STATE_BUDGET:
                raise ValueError(
                    "A sliding window's UTF-8 size exceeds the request body budget "
                    f"({REQUEST_STATE_BUDGET} bytes of {REQUEST_MAX_BYTES} allowed). "
                    f"Use a smaller window_size (current {ws} tokens)."
                )
        # Window-level concurrency: each window runs detection + typing
        # independently as a coroutine (batches already concurrent under the
        # shared semaphore). Sequential (or single-window) runs preserve the
        # thread-pool-era error semantics: failures propagate; the concurrent
        # path logs and skips failed windows as empty results.
        # latency_first: windows within the token cap skip the detection
        # round trip entirely (flat typing of every viable sub-span).
        # None on the call inherits the constructor's latency_first.
        lf = self._latency_first if latency_first is None else latency_first
        skips = [
            lf and not fast_mode and len(toks) <= LATENCY_FIRST_MAX_TOKENS
            for _wt, toks, _base in windows
        ]
        if len(windows) <= 1 or self._max_workers <= 1:
            window_entities = [
                await self._type_window(
                    s, window_text, toks, base, types, prob_threshold, skip_detection=skip
                )
                for (window_text, toks, base), skip in zip(windows, skips, strict=True)
            ]
        else:

            async def _guarded_window(
                window_text: str,
                toks: list[tuple[int, int, str]],
                base: int,
                skip: bool,
            ) -> list[Entity]:
                try:
                    return await self._type_window(
                        s,
                        window_text,
                        toks,
                        base,
                        types,
                        prob_threshold,
                        skip_detection=skip,
                    )
                except Exception as exc:
                    logger.warning("JevNER window failed: %s\n%s", exc, traceback.format_exc())
                    return []

            window_entities = list(
                await asyncio.gather(
                    *(
                        _guarded_window(wt, toks, base, skip)
                        for (wt, toks, base), skip in zip(windows, skips, strict=True)
                    )
                )
            )
        entities: list[Entity] = [e for ents in window_entities for e in ents]
        merged = self._combine(entities, deduplicate)
        # Per-type conventions (A6): minimum entity length and bracket
        # inclusion. Applied before the PERSON upgrade so bracket extension
        # rebuilds plain Entity objects only.
        min_chars_map = {t.label: t.min_chars for t in types}
        brackets_map = {t.label: t.include_brackets for t in types}
        refined: list[Entity] = []
        for e in merged:
            if len(e.text) < min_chars_map.get(e.type.label, 1):
                continue
            if (
                brackets_map.get(e.type.label, False)
                and e.start > 0
                and e.end < len(s)
                and s[e.start - 1] in "《〈"
                and s[e.end] in "》〉"
            ):
                # Book-title conventions: enclosing brackets belong to the entity.
                e = Entity(
                    text=s[e.start - 1 : e.end + 1],
                    types=e.types,
                    start=e.start - 1,
                    end=e.end + 1,
                )
            refined.append(e)
        merged = refined
        # Flat-F1 convention mode (A7): drop same-type fragments, merge
        # same-type neighbors split by delimiters. Default keeps nesting.
        if flatten_fragments:
            merged = _flatten_fragments(s, merged)
        if with_gender:
            # Upgrade PERSON entities to Person (with gender), then build the tree;
            # the PERSON label always exists in a taxonomy (fixed semantic anchor)
            names = [e.text for e in merged if e.type.label == PERSON_LABEL]
            genders = await self._recognize_genders(names)
            merged = [
                (
                    Person(
                        text=e.text,
                        types=e.types,
                        start=e.start,
                        end=e.end,
                        gender=genders[e.text][0],
                        gender_probability=genders[e.text][1],
                    )
                    if e.type.label == PERSON_LABEL and e.text in genders
                    else e
                )
                for e in merged
            ]
        # Rebuild the nesting tree by offset containment: the return value carries the structure
        return entity_tree(merged)

    def __call__(
        self,
        s: str,
        *,
        with_gender: bool = True,
        entity_packs: Iterable[EntityPack | str] | None = None,
        entity_types: Iterable[EntityType] | None = None,
        prob_threshold: float = DEFAULT_PROB_THRESHOLD,
        fast_mode: bool = False,
        window_size: int | None = None,
        stride: int | None = None,
        deduplicate: bool = True,
        flatten_fragments: bool = False,
        latency_first: bool | None = None,
    ) -> list[EntityNode]:
        """Recognize named entities in a text; returns the nesting forest.

        Full pipeline: source-anchored tokens -> sliding windows ->
        per-token Noul detection (gray-zone multiround recheck) ->
        region sub-span Choice typing (NONE veto + multi-type
        collection) -> per-occurrence merge -> PERSON upgrade -> tree
        building by offset containment.

        Blocks until the whole pipeline finishes (requests interleave
        with other calls on the instance's private event loop). Async
        callers should use arecognize instead.

        Args:
            s: The text to process, in any language.
            with_gender: Whether PERSON entities get a gender judgment.
            entity_packs: Packs (objects or names) to enable; None
                uses the taxonomy default packs.
            entity_types: Extra explicit types (objects or labels),
                unioned with the packs.
            prob_threshold: Primary-type probability floor in [0, 1].
            fast_mode: True treats the whole text as one window
                directly (must fit gateway limits); False enables the
                sliding-window long-text algorithm.
            window_size: Override of the instance window size (tokens).
            stride: Override of the slide step; defaults to
                window_size // 2.
            deduplicate: Whether to merge cross-window duplicates per
                occurrence (type union).
            latency_first: Skip the detection round trip on
                sentence-scale inputs (see the constructor docstring);
                one serial round trip total. None inherits the
                constructor setting.
            flatten_fragments: Flat-F1 convention mode (default False):
                drop entities contained by a same-type entity and merge
                same-type neighbors split by delimiters — serves flat
                gold standards; the default output keeps nesting.

        Returns:
            The entity nesting forest: top-level (uncontained)
            EntityNode list sorted by (start, end); node.entity holds
            the entity itself (PERSON entities carry gender when
            with_gender=True) and node.children holds entities fully
            inside its span. Nesting rides in the return value itself
            — no downstream post-processing.

        Raises:
            AssertionError: prob_threshold outside [0, 1].
            ValueError: An unregistered pack name or type label, empty
                type resolution, invalid window parameters, or a
                window exceeding the request byte budget.
            TypeSafeError: Request construction failed (e.g. missing
                credentials).
            TypeSafeAPIError: The API returned an error response.
            TypeSafeAPIConnectionError: Network failure.
        """
        return self._submit(
            self._arecognize(
                s,
                with_gender=with_gender,
                entity_packs=entity_packs,
                entity_types=entity_types,
                prob_threshold=prob_threshold,
                fast_mode=fast_mode,
                window_size=window_size,
                stride=stride,
                deduplicate=deduplicate,
                flatten_fragments=flatten_fragments,
                latency_first=latency_first,
            )
        )

    async def arecognize(
        self,
        s: str,
        *,
        with_gender: bool = True,
        entity_packs: Iterable[EntityPack | str] | None = None,
        entity_types: Iterable[EntityType] | None = None,
        prob_threshold: float = DEFAULT_PROB_THRESHOLD,
        fast_mode: bool = False,
        window_size: int | None = None,
        stride: int | None = None,
        deduplicate: bool = True,
        flatten_fragments: bool = False,
        latency_first: bool | None = None,
    ) -> list[EntityNode]:
        """Async twin of __call__: the same pipeline, awaitable.

        The work still runs on this instance's private event loop
        (submitted and awaited via wrap_future), so the HTTP client
        never crosses event loops — safe to call from any caller
        loop. Parameters, return value, and exceptions are identical
        to __call__.
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._arecognize(
                s,
                with_gender=with_gender,
                entity_packs=entity_packs,
                entity_types=entity_types,
                prob_threshold=prob_threshold,
                fast_mode=fast_mode,
                window_size=window_size,
                stride=stride,
                deduplicate=deduplicate,
                flatten_fragments=flatten_fragments,
                latency_first=latency_first,
            ),
            loop,
        )
        return await asyncio.wrap_future(future)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _aclose_client(self) -> None:
        """Close the async HTTP client on its own loop."""
        await self._client.aclose()

    def close(self) -> None:
        """Release the HTTP client and stop the private event-loop thread.

        Idempotent, safe from sync and async contexts alike. Async code
        may prefer await aclose() (which also skips the loop-thread join).

        The instance must not be used afterwards (the underlying client
        is closed); create a new JevNER instead.
        """
        if self._disk is not None:
            self._disk.close()
        with self._loop_lock:
            loop = self._loop
            thread = self._loop_thread
            self._loop = None
            self._loop_thread = None
            self._semaphore = None
        if loop is not None and not loop.is_closed():
            # Close the client on its own loop (blocking until drained — the
            # submission queues legally even mid-startup), then stop the loop
            # and join its thread.
            asyncio.run_coroutine_threadsafe(self._aclose_client(), loop).result()
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None:
                thread.join(timeout=5.0)
        elif self._loop_running_in_this_thread():
            # Loop never started (or already closed), and THIS thread runs an
            # event loop (async caller using the sync close): asyncio.run
            # forbids nesting, and fire-and-forget scheduling on the caller's
            # loop could be dropped if that loop exits immediately — so close
            # the unused client on a throwaway thread's own loop.
            closer = threading.Thread(
                target=lambda: asyncio.run(self._aclose_client()), daemon=True
            )
            closer.start()
            closer.join(timeout=5.0)
        else:
            # Sync caller, loop never started (or already closed): close the
            # client on a throwaway loop — an unused or already-closed httpx
            # client holds no loop-bound state.
            asyncio.run(self._aclose_client())

    @staticmethod
    def _loop_running_in_this_thread() -> bool:
        """True when an event loop is running in the calling thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    async def aclose(self) -> None:
        """Async close: release the client and stop the loop without
        blocking the caller's event loop on the loop-thread join."""
        if self._disk is not None:
            self._disk.close()
        with self._loop_lock:
            loop = self._loop
            self._loop = None
            self._loop_thread = None
            self._semaphore = None
        if loop is not None and not loop.is_closed():
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._aclose_client(), loop))
            loop.call_soon_threadsafe(loop.stop)
        else:
            # No private loop ever ran (or it already closed): close directly
            # on the caller's loop — the client is unused or already closed.
            await self._client.aclose()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the context manager and release resources."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the async context manager and release resources."""
        await self.aclose()
