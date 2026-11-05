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
import threading
import traceback
from collections.abc import Coroutine, Iterable, Mapping
from typing import Any, Protocol, Self, TypeVar, cast

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    SystemOneResponse,
)
from vortezwohl.cache import LRUCache
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

T = TypeVar("T")
"""Type variable for private-loop submission helpers."""

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

DEFAULT_MAX_WORKERS = 8
"""Default in-flight request limit (the asyncio semaphore width).

The bottleneck is gateway round-trips and rate limiting, not local
CPU; this bounds how many requests (windows, detection batches,
recheck batches, typing batches, gender batches) may be in flight at
once on the private event loop. 0 or 1 serializes requests.
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

    Usage mirrors BertNER: construct once, call repeatedly. Credentials
    default to the TYPESAFE_API_KEY / TYPESAFE_BASE_URL environment
    variables and may also be passed explicitly.

    Concurrency: every gateway request is a coroutine on one private
    event loop (a single daemon thread), bounded by an in-flight
    semaphore of max_workers. The synchronous call operator works from
    any thread; async callers should await arecognize().
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
            max_workers: In-flight request limit (the semaphore
                width on the private event loop); 0 or 1 serializes
                requests.
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
        # Entity type system: type/pack registries + default packs, the single
        # source of type semantics; users extend or override via JSON
        self._taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        # In-flight request limit: one asyncio.Semaphore bounds windows,
        # detection batches, recheck batches, typing batches, and gender
        # batches alike — the global rate the old thread pool provided.
        self._max_workers = max(1, max_workers)
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
        self._detect_cache: _Cache = LRUCache(capacity=DETECT_CACHE_CAPACITY)
        self._typing_cache: _Cache = LRUCache(capacity=TYPING_CACHE_CAPACITY)
        self._gender_cache: _Cache = LRUCache(capacity=GENDER_CACHE_CAPACITY)
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
    ) -> SystemOneResponse:
        """One gateway request under the shared in-flight semaphore.

        The semaphore (max_workers) is the single rate limiter for every
        request this instance issues, mirroring the global bound the old
        thread pool provided.
        """
        assert self._semaphore is not None, "private loop must exist before any request"
        async with self._semaphore:
            return await self._client.system_one(
                state=state, questions=questions, extra_headers=extra_headers
            )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def _detect_batch(
        self,
        state: str,
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

        Args:
            state: The rendered detection state (window text).
            tokens: All window tokens (neighbor lookup).
            pairs: (index, token) pairs for this batch.

        Returns:
            Token index -> Noul score for the answered questions.
        """
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
        response = await self._system_one(
            state=state, questions=questions, extra_headers=LONG_CONTEXT_HEADER
        )
        batch: dict[int, float] = {}
        for i, _tok in pairs:
            answer = response.nouls.get(f"tok_{i}")
            if answer is None:
                logger.warning("JevNER got no detection answer for tok_%d, skipped.", i)
                continue
            batch[i] = float(answer.noul)
        return batch

    async def _detect_regions(
        self, window_text: str, tokens: list[tuple[int, int, str]]
    ) -> tuple[list[tuple[int, int, str]], list[int], list[list[tuple[int, int, str]]]]:
        """Detect entity token regions within one window.

        Submits one Noul question per token ("is this token part of a
        named entity?"), filters hits by threshold, and reassembles
        them into regions by positional adjacency. This mirrors the
        token-level prediction of the BertNER base model and treats
        every tokenization shape alike. Results are cached in-process
        by window-text hash; repeated windows cost zero requests.

        Concurrency: every batch of the first pass is independent, so
        all fly at once (the former sequential loop cost one round
        trip per 64 tokens — ~188 serialized requests on a full 12k
        window). The recheck rounds resample the same gray-zone token
        set, so they also fly concurrently; taking the max over the
        concurrent samples is exactly the old sequential-rounds result
        (only tokens that would have passed early are re-asked).

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
        # First pass: all batches independent -> all in flight at once.
        indexed: list[tuple[int, tuple[int, int, str]]] = list(enumerate(tokens))
        first_pass: list[list[tuple[int, tuple[int, int, str]]]] = []
        for raw_batch in batched(indexed, DETECT_BATCH_SIZE):
            pairs = [p for p in raw_batch if p is not None]
            if pairs:
                first_pass.append(pairs)
        scores: dict[int, float] = {}
        first_results: list[dict[int, float]] = list(
            await asyncio.gather(
                *(self._detect_batch(state, tokens, pairs) for pairs in first_pass)
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
                        self._detect_batch(state, tokens, [(i, tokens[i]) for i in indices])
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
        Typing batches fly concurrently under the shared semaphore.

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
        _, hits, strict_regions = await self._detect_regions(window_text, tokens)
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

        async def _type_batch(
            pairs: list[tuple[int, str]],
        ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
            """One typing batch request: map span index -> ranked type tuple;
            an empty tuple means NONE."""
            questions: dict[str, Choice] = {
                f"type_{i}": Choice(instructions={"SPAN": cand}, criteria=criteria)
                for i, cand in pairs
            }
            response = await self._system_one(
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

        # Typing batches run concurrently under the semaphore; batch size narrows
        # by option budget: options per question = types + 1 (NONE), questions per
        # batch <= REQUEST_MAX_OPTIONS / options per question, capped by
        # TYPING_BATCH_SIZE (the gateway caps total options per request at 512; 422
        # beyond). batched() pads the last batch with None; filter them out.
        options_per_question = len(entity_types) + 1
        batch_limit = min(TYPING_BATCH_SIZE, REQUEST_MAX_OPTIONS // options_per_question)
        batches = [
            [item for item in batch if item is not None]
            for batch in batched(uncached, max(1, batch_limit))
        ]
        batches = [batch for batch in batches if batch]
        if len(batches) <= 1 or self._max_workers <= 1:
            batch_results = [await _type_batch(pairs) for pairs in batches]
        else:
            # Per-batch resilience (thread-pool-era semantics): a failed batch
            # logs its traceback and yields {} (spans fall back to NONE).
            async def _guarded_batch(
                pairs: list[tuple[int, str]],
            ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
                try:
                    return await _type_batch(pairs)
                except Exception as exc:
                    logger.warning(
                        "JevNER typing batch failed: %s\n%s", exc, traceback.format_exc()
                    )
                    return {}

            batch_results = list(
                await asyncio.gather(*(_guarded_batch(pairs) for pairs in batches))
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
            if cached is not None:
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
        # Window-level concurrency: each window runs detection + typing
        # independently as a coroutine (batches already concurrent under the
        # shared semaphore). Sequential (or single-window) runs preserve the
        # thread-pool-era error semantics: failures propagate; the concurrent
        # path logs and skips failed windows as empty results.
        if len(windows) <= 1 or self._max_workers <= 1:
            window_entities = [
                await self._type_window(window_text, toks, types, prob_threshold)
                for window_text, toks in windows
            ]
        else:

            async def _guarded_window(
                window_text: str, toks: list[tuple[int, int, str]]
            ) -> list[Entity]:
                try:
                    return await self._type_window(window_text, toks, types, prob_threshold)
                except Exception as exc:
                    logger.warning("JevNER window failed: %s\n%s", exc, traceback.format_exc())
                    return []

            window_entities = list(
                await asyncio.gather(*(_guarded_window(wt, toks) for wt, toks in windows))
            )
        entities: list[Entity] = [e for ents in window_entities for e in ents]
        merged = self._combine(entities, deduplicate)
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
