"""JevNER core: sliding-window named entity recognition with no local models.

This module implements the jevner SDK pipeline. The overall structure
mirrors DeepLoTX BertNER (sliding windows + cross-window merging +
gender augmentation), but a System One gateway model replaces all
local BERT inference, and a two-stage "token detection + span typing"
design replaces BertNER fixed n-gram candidates so any language works:

1. Detection: the bundled Qwen3 tokenizer splits each window into a
   token sequence; every token becomes one Noul question ("is this
   token part of a named entity?") submitted in batches and answered
   in parallel. This mirrors the base model token-level prediction in
   BertNER, is language-agnostic, and is free of fixed n-gram length
   assumptions.
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

What callers and maintainers need to know first:
- This class loads no local neural weights; the only external
  dependency is the gateway API.
- Choice probabilities are a competitive distribution normalized
  across options, so NONE probability naturally carries the
  "not an entity" confidence, and prob_threshold filters on the
  selected type own probability.
- Detection and typing questions are batched by their own sizes so a
  single request never exceeds the gateway request-body and
  total-option hard limits.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from typing import Protocol, Self, cast

from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient
from vortezwohl.cache import LRUCache
from vortezwohl.concurrent import ThreadPool
from vortezwohl.iter import batched

from jevner.entities import (
    DEFAULT_TAXONOMY,
    NONE_LABEL,
    PERSON_LABEL,
    Entity,
    EntityNode,
    EntityPack,
    EntityTaxonomy,
    EntityType,
    Gender,
    Person,
    entity_tree,
)
from jevner.prompt import render_detection_state, render_gender_state, render_typing_state
from jevner.tokenizer import BundledTokenizer, strip_span

logger = logging.getLogger("jevner")

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
adjacent overlap, windows submitted concurrently on a thread pool.
"""

DEFAULT_STRIDE_DIVISOR = 2
"""Default stride factor: stride = window_size // 2 (50% overlap).

Token-level detection is insensitive to window boundaries
(overlapping windows guarantee a boundary-crossing entity is fully
visible on at least one side), so 50% overlap suffices — half the
requests of the BertNER 25% step.
"""

DEFAULT_MAX_WORKERS = 8
"""Default thread count for window-level parallelism.

The bottleneck is gateway round-trips and rate limiting, not local
CPU; 8-way concurrency compresses multi-window latency within the
rate limits. 0 or 1 degrades to sequential execution.
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
included)."""

TYPING_BATCH_SIZE = 64
"""Upper bound on typing questions per request (further narrowed by
the option budget).

Typing demands fine semantic judgment; large batches dilute the
model NONE verdicts on fragment spans (in-batch interference), so
small batches are markedly more stable."""

DETECT_BATCH_SIZE = 64
"""Detection questions per request.

Aligned with the typing batch: detection suffers the same in-batch
interference — at 256 per batch, two-character Chinese place names
had their Noul scores diluted into misses, while 64 stays stable."""

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


class JevNER:
    """A sliding-window multilingual entity recognizer over gateway models.

    Usage mirrors BertNER: construct once, call repeatedly.
    Credentials default to the TYPESAFE_API_KEY / TYPESAFE_BASE_URL
    environment variables and may also be passed explicitly.
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
            max_workers: Window-level concurrency; 0 or 1 runs
                sequentially.
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
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            timeout=timeout,
            retry=RetryPolicy(max_retries=3, backoff_initial=1.0, backoff_max=8.0, timeout=None),
        )
        self._tokenizer = BundledTokenizer()
        self._window_size = window_size
        # Entity type system: type/pack registries + default packs, the single
        # source of type semantics; users extend or override via JSON
        self._taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        # Window-level thread pool: windows (long documents) submit detection and
        # typing requests concurrently. Thread safety: httpx2.Client pools connections.
        self._max_workers = max(1, max_workers)
        self._pool = ThreadPool(max_workers=self._max_workers)
        # Three in-process LRU caches: detection (window -> hit tokens), typing
        # (window + span -> type tuples), gender (name -> judgment).
        # Repeated text / overlapping windows / repeated names all hit cache.
        self._detect_cache: _Cache = LRUCache(capacity=DETECT_CACHE_CAPACITY)
        self._typing_cache: _Cache = LRUCache(capacity=TYPING_CACHE_CAPACITY)
        self._gender_cache: _Cache = LRUCache(capacity=GENDER_CACHE_CAPACITY)
        logger.debug(
            "JevNER initialized with model: %s, window_size: %d, max_workers: %d.",
            model_name,
            window_size,
            self._max_workers,
        )

    def _detect_regions(
        self, window_text: str, tokens: list[tuple[int, int, str]]
    ) -> tuple[list[tuple[int, int, str]], list[int], list[list[tuple[int, int, str]]]]:
        """Detect entity token regions within one window.

        Submits one Noul question per token ("is this token part of a
        named entity?"), filters hits by threshold, and reassembles
        them into regions by positional adjacency. This mirrors the
        token-level prediction of the BertNER base model and treats
        every tokenization shape alike. Results are cached in-process
        by window-text hash; repeated windows cost zero requests.

        Args:
            window_text: The window text (an exact source substring;
                used to render the state).
            tokens: Window tokens with offsets anchored to the source.

        Returns:
            (all window tokens, hit token indices, strict contiguous
            regions); tokens carry source-global character offsets.
        """
        if not tokens:
            return [], [], []
        cache_key = hashlib.sha256(window_text.encode("utf-8")).hexdigest()
        cached = self._detect_cache.read(cache_key)
        if cached is not None:
            return cast(
                "tuple[list[tuple[int, int, str]], list[int], list[list[tuple[int, int, str]]]]",
                cached,
            )
        state = render_detection_state(window_text)
        # Per-token Noul scores; gray-zone tokens (DETECT_RECHECK_LOW..HIGH)
        # get a second recheck round; passing the threshold in either round counts
        # (the live API has per-request sampling variance; the same state is
        # stable across rounds, and rechecks cut token misses sharply).
        scores: dict[int, float] = {}
        recheck: list[int] = []
        indexed: list[tuple[int, tuple[int, int, str]]] = list(enumerate(tokens))
        for raw_batch in batched(indexed, DETECT_BATCH_SIZE):
            pairs = [p for p in raw_batch if p is not None]
            if not pairs:
                continue
            # Each token question carries its neighbors: whether an isolated single
            # character belongs to an entity is context-dependent ("吴" alone is
            # undecidable; with neighbors it is), giving detection region vision.
            questions: dict[str, Noul] = {
                f"tok_{i}": Noul(
                    instructions={
                        "TOKEN": tok[2].strip(),
                        "POSITION": i,
                        "LEFT": tokens[i - 1][2].strip() if i > 0 else "",
                        "RIGHT": tokens[i + 1][2].strip() if i + 1 < len(tokens) else "",
                    }
                )
                for i, tok in pairs
            }
            response = self._client.system_one(
                state=state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
            )
            for i, _tok in pairs:
                answer = response.nouls.get(f"tok_{i}")
                if answer is None:
                    logger.warning("JevNER got no detection answer for tok_%d, skipped.", i)
                    continue
                scores[i] = float(answer.noul)
                if DETECT_RECHECK_LOW <= scores[i] < DETECT_PROB_THRESHOLD:
                    recheck.append(i)
        # Gray-zone multiround recheck (DETECT_RECHECK_ROUNDS independent samples, keep max)
        for _ in range(DETECT_RECHECK_ROUNDS):
            pending = [
                i
                for i, score in scores.items()
                if DETECT_RECHECK_LOW <= score < DETECT_PROB_THRESHOLD
            ]
            if not pending:
                break
            for raw_batch in batched(pending, DETECT_BATCH_SIZE):
                indices = [i for i in raw_batch if i is not None]
                if not indices:
                    continue
                questions = {
                    f"tok_{i}": Noul(
                        instructions={
                            "TOKEN": tokens[i][2].strip(),
                            "POSITION": i,
                            "LEFT": tokens[i - 1][2].strip() if i > 0 else "",
                            "RIGHT": tokens[i + 1][2].strip() if i + 1 < len(tokens) else "",
                        }
                    )
                    for i in indices
                }
                response = self._client.system_one(
                    state=state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
                )
                for i in indices:
                    answer = response.nouls.get(f"tok_{i}")
                    if answer is None:
                        continue
                    scores[i] = max(scores[i], float(answer.noul))
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
        result = (tokens, hits, regions)
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

    def _type_window(
        self,
        window_text: str,
        tokens: list[tuple[int, int, str]],
        entity_types: tuple[EntityType, ...],
        prob_threshold: float,
    ) -> list[Entity]:
        """Run detection and typing for one window; returns its entities.

        Detection yields the union of two region strategies (strict
        contiguity + gap bridging); every contiguous sub-span within
        each region (at most MAX_ENTITY_TOKENS tokens long, nesting
        included) becomes one Choice question batched for typing.

        Typing supports multiple types: Choice probabilities are a
        competitive distribution; the selected option must clear
        prob_threshold as the primary type, and remaining types at or
        above SECONDARY_TYPE_THRESHOLD ride along, all ranked by
        probability — a single type is never forced.

        Args:
            window_text: The window text (an exact source substring;
                used to render the state).
            tokens: Window tokens with offsets anchored to the source.
            entity_types: The enabled entity types for this pass.
            prob_threshold: Primary-type probability floor.

        Returns:
            Entities recognized in this window (not yet merged across
            windows), offsets anchored to the source.
        """
        _, hits, strict_regions = self._detect_regions(window_text, tokens)
        all_regions: list[list[tuple[int, int, str]]] = list(strict_regions)
        for region in self._bridge_regions(tokens, hits, strict_regions):
            if region not in all_regions:
                all_regions.append(region)
        if not all_regions:
            return []
        # Generate every contiguous sub-span in each region; edge punctuation
        # stripping moves offsets with it so span text stays an exact substring
        spans: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()
        for region in all_regions:
            n = len(region)
            for i in range(n):
                for length in range(1, min(MAX_ENTITY_TOKENS, n - i) + 1):
                    raw_start = region[i][0]
                    raw_end = region[i + length - 1][1]
                    text, start, end = strip_span(window_text, raw_start, raw_end)
                    key = text.strip()
                    # Skip blank, duplicate, or punctuation-only candidates
                    if not key or (text, start, end) in seen:
                        continue
                    if not any(ch.isalnum() for ch in key):
                        continue
                    seen.add((text, start, end))
                    spans.append((text, start, end))
        if not spans:
            return []
        # Typing cache hits: skip spans already judged (window overlap / repeats)
        window_hash = hashlib.sha256(window_text.encode("utf-8")).hexdigest()
        uncached: list[tuple[int, str]] = []
        cached_entities: dict[int, tuple[tuple[EntityType, float], ...]] = {}
        for i, (text, _start, _end) in enumerate(spans):
            cache_key = (window_hash, text, prob_threshold)
            hit = self._typing_cache.read(cache_key)
            if hit is not None:
                cached_entities[i] = cast("tuple[tuple[EntityType, float], ...]", hit)
            else:
                uncached.append((i, text))
        entities: list[Entity] = []
        state = render_typing_state(window_text, entity_types)
        # Criteria carry labels only: descriptions already appear once in the
        # state ENTITY_TYPES table (repeating them per question balloons a
        # question from ~150 bytes to ~2KB; 64 per batch ~= 128KB, past the
        # 65,535-byte hard cap — the observed stable 500s came from exactly this).
        # Labels match the type-table keys; the model resolves them by lookup.
        criteria: dict[str, str | None] = {t.label: None for t in entity_types}
        criteria[NONE_LABEL] = None

        def _type_batch(
            pairs: list[tuple[int, str]],
        ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
            """One typing batch request: map span index -> ranked type tuple;
            an empty tuple means NONE."""
            questions: dict[str, Choice] = {
                f"type_{i}": Choice(instructions={"SPAN": cand}, criteria=criteria)
                for i, cand in pairs
            }
            response = self._client.system_one(
                state=state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
            )
            batch: dict[int, tuple[tuple[EntityType, float], ...]] = {}
            for i, _cand in pairs:
                answer = response.choices.get(f"type_{i}")
                if answer is None:
                    logger.warning("JevNER got no typing answer for type_%d, skipped.", i)
                    batch[i] = ()
                    continue
                # NONE veto: when the model judges a span a non-entity,
                # no secondary types are collected — otherwise fragments slip in
                # through low-probability secondary types bypassing the veto
                # (observed with 0.25-0.5 fragment noise)
                if answer.choice == NONE_LABEL:
                    batch[i] = ()
                    continue
                # The primary type (selected option) must clear prob_threshold;
                # others at or above SECONDARY_TYPE_THRESHOLD ride along as secondary
                best_per_type: dict[EntityType, float] = {}
                primary_prob = float(answer.probabilities.get(answer.choice, 0.0))
                if primary_prob < prob_threshold:
                    batch[i] = ()
                    continue
                primary = self._taxonomy._type_of(answer.choice)
                best_per_type[primary] = primary_prob
                for label, prob in answer.probabilities.items():
                    # NONE is the veto option, not an entity type; excluded from collection
                    if label == NONE_LABEL or label == answer.choice:
                        continue
                    if prob < SECONDARY_TYPE_THRESHOLD:
                        continue
                    etype = self._taxonomy._type_of(label)
                    if etype not in best_per_type or prob > best_per_type[etype]:
                        best_per_type[etype] = float(prob)
                batch[i] = tuple(sorted(best_per_type.items(), key=lambda kv: -kv[1]))
            return batch

        # Typing batches run in parallel; batch size narrows by option budget:
        # options per question = types + 1 (NONE), questions per batch <=
        # REQUEST_MAX_OPTIONS / options per question, capped by TYPING_BATCH_SIZE
        # (the gateway caps total options per request at 512; 422 beyond).
        # batched() pads the last batch with None; filter them out.
        options_per_question = len(entity_types) + 1
        batch_limit = min(TYPING_BATCH_SIZE, REQUEST_MAX_OPTIONS // options_per_question)
        batches = [
            [item for item in batch if item is not None]
            for batch in batched(uncached, max(1, batch_limit))
        ]
        batches = [batch for batch in batches if batch]
        if len(batches) <= 1 or self._max_workers <= 1:
            batch_results = [_type_batch(pairs) for pairs in batches]
        else:
            # ThreadPool.map unpacks Iterable params with *param, so wrap each
            # batch in a 1-tuple to hand _type_batch the full list
            batch_results = []
            for result in self._pool.map(_type_batch, [(list(b),) for b in batches]):
                if result.error is not None:
                    logger.warning(
                        "JevNER typing batch failed: %s\n%s", result.error, result.traceback
                    )
                    batch_results.append({})
                else:
                    batch_results.append(
                        cast("dict[int, tuple[tuple[EntityType, float], ...]]", result.returns)
                    )
        for pairs, result in zip(batches, batch_results, strict=True):
            for i, text in pairs:
                self._typing_cache.write((window_hash, text, prob_threshold), result.get(i, ()))
        # Combine: cache hits + this round; NONE (empty tuple) yields no entity
        all_ranked: dict[int, tuple[tuple[EntityType, float], ...]] = dict(cached_entities)
        for result in batch_results:
            all_ranked.update(result)
        for i, (text, start, end) in enumerate(spans):
            ranked = all_ranked.get(i, ())
            if ranked:
                entities.append(Entity(text=text, types=ranked, start=start, end=end))
        return entities

    def _combine(self, entities: list[Entity], deduplicate: bool) -> list[Entity]:
        """Merge entities across windows (per-occurrence dedup + type union).

        Rules:

        - Same occurrence (identical start/end): repeated recognitions
          from overlapping windows or batches merge into one entity,
          taking the union of types and the max probability per type;
        - Different occurrences: kept independently — spans in a
          containment relation are nested entities (e.g. GPE "福建"
          [16:18] inside GPE "福建福州" [16:20]); the merge stage never
          absorbs or drops either side, and relations are derived
          entirely downstream by entity_tree() from offset containment.

        Fragment noise (detection-variance artifacts like PERSON "吴")
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

    def _recognize_genders(self, names: list[str]) -> dict[str, tuple[Gender, float]]:
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
            if cached is not None:
                results[name] = cast("tuple[Gender, float]", cached)
            elif name not in pending:
                pending.append(name)
        if not pending:
            return results
        state = render_gender_state()
        for raw_batch in batched(pending, DETECT_BATCH_SIZE):
            batch = [n for n in raw_batch if n is not None]
            if not batch:
                continue
            questions: dict[str, Choice] = {
                f"gender_{i}": Choice(instructions={"NAME": name}, criteria=GENDER_CRITERIA)
                for i, name in enumerate(batch)
            }
            response = self._client.system_one(state=state, questions=questions)
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
                results[name] = pair
        return results

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
    ) -> list[EntityNode]:
        """Recognize named entities in a text; returns the nesting forest.

        Full pipeline: source-anchored tokens -> sliding windows ->
        per-token Noul detection (gray-zone multiround recheck) ->
        region sub-span Choice typing (NONE veto + multi-type
        collection) -> per-occurrence merge -> PERSON upgrade -> tree
        building by offset containment.

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
            windows = [(s, self._tokenizer.window_tokens(s))]
        else:
            windows = self._tokenizer.windows(s, ws, st)
        # Request-body guard: window text dominates the body; over budget fails fast
        # (an oversized request returns an opaque 500 rather than an actionable
        # error, so this raises with actionable guidance)
        for window_text, _toks in windows:
            if len(window_text.encode("utf-8")) > REQUEST_STATE_BUDGET:
                raise ValueError(
                    "A sliding window's UTF-8 size exceeds the request body budget "
                    f"({REQUEST_STATE_BUDGET} bytes of {REQUEST_MAX_BYTES} allowed). "
                    f"Use a smaller window_size (current {ws} tokens)."
                )
        # Window-level parallelism: each window runs detection + typing
        # independently (batches already parallel); sequential config or single window runs in order
        if len(windows) <= 1 or self._max_workers <= 1:
            window_entities = [
                self._type_window(window_text, toks, types, prob_threshold)
                for window_text, toks in windows
            ]
        else:
            window_entities = []
            jobs_args = [
                (window_text, toks, types, prob_threshold) for window_text, toks in windows
            ]
            for result in self._pool.map(self._type_window, [(a,) for a in jobs_args]):
                if result.error is not None:
                    logger.warning("JevNER window failed: %s\n%s", result.error, result.traceback)
                    window_entities.append([])
                else:
                    window_entities.append(cast("list[Entity]", result.returns))
        entities: list[Entity] = [e for ents in window_entities for e in ents]
        merged = self._combine(entities, deduplicate)
        if with_gender:
            # Upgrade PERSON entities to Person (with gender), then build the tree;
            # the PERSON label always exists in a taxonomy (fixed semantic anchor)
            names = [e.text for e in merged if e.type.label == PERSON_LABEL]
            genders = self._recognize_genders(names)
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

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

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
