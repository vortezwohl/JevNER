"""JevNER 主实现：无本地模型的滑动窗口命名实体识别。

本模块实现 jevner SDK 的核心算法。整体结构与 DeepLoTX BertNER
同构（滑动窗口 + 跨窗口合并 + 性别增强），但用 TypeSafe 的 System One
模型 jev 替代全部本地 BERT 推理，并用"词元检测 + 区间分型"两阶段替代
BertNER 的固定 n-gram 候选，以兼容全球语种：

1. 检测阶段：随包 Qwen3 分词器把窗口切为词元序列，每个词元作为
   一个 Noul 问题（"该词元是否为命名实体组成部分"）批量提交，
   jev 并行回答；对应 BertNER 底座模型的 token 级预测，语言无关，
   不受固定 n-gram 长度假设的约束；
2. 分型阶段：连续命中词元重组为实体区间，区间内生成全部连续子
   span（含嵌套），每个子 span 作为一个 Choice 问题（标签为启用
   实体类型 + NONE）批量提交，jev 并行分型；嵌套实体（如
   "中华人民共和国外交部"内的"中华人民共和国"）天然被覆盖；
3. 跨窗口合并：同类型包含吸收（概率最高者优先，清理窗口碎片），
   不同类型的包含关系是嵌套实体，两侧都保留；
4. PERSON 实体按文本去重后批量做性别判定（Choice：
   MALE/FEMALE/UNKNOWN），结果写入进程内 LRU 缓存避免重复请求。

调用方与维护者进入本文件时最需要先知道：
- 本类不加载任何本地神经网络权重，唯一外部依赖是 TypeSafe API；
- Choice 的 probabilities 是跨选项归一化的竞争分布，NONE 的概率
  天然承担"非实体"置信度，prob_threshold 过滤基于被选中类型
  自身的概率；
- 检测与分型问题分别按各自批次大小打包，
  保证单个请求不超过 jev 的 per-request 上下文限制。
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

DEFAULT_MODEL = "jev-latest"
"""默认模型别名：自动路由到 TypeSafe 最新稳定版 jev。"""

LONG_CONTEXT_HEADER = {"TypeSafe-Danger-Zone-Long-Context": "true"}
"""jev 长上下文请求头。

携带该头的请求 per-request 与 per-question 上下文均提升至 262k
token；不携带时默认 64k/32k。分型 state 含完整类型表，超长窗口
（64k 以上）依赖该头；常规窗口下它只作保险，无副作用。
"""

REQUEST_MAX_BYTES = 65_535
"""单个 system_one 请求体（prompt）的字节硬上限（服务端限制）。

请求体 = state JSON + 全部问题 JSON。超限请求被服务端拒绝
（实测 500/网关熔断），且难以从报错定位——因此在设计上保证
任意请求不超：窗口按字节预算切分（REQUEST_STATE_BUDGET），
分型 criteria 只携带标签不携带描述（描述仅在 state 类型表出现一次）。
"""

BYTES_PER_TOKEN_WORST = 4
"""按 UTF-8 最坏情况估算的每 token 字节数。

中文一字一 token 约 3 字节，英文子词平均约 4 字节，取 4 为
统一安全上界，用于把窗口字节预算换算为 token 窗口大小。
"""

REQUEST_STATE_BUDGET = 48_000
"""窗口文本在请求体中的字节预算。

请求体 65,535 字节减去分型类型表（约 2KB）、问题批次（每问
约 150~250 字节 × 批次 64 ≈ 16KB）与 JSON 结构开销后，留给
窗口文本的安全预算。检测与分型共用同一窗口，按两者中更紧的
分型预算取值。
"""

DEFAULT_WINDOW_SIZE = REQUEST_STATE_BUDGET // BYTES_PER_TOKEN_WORST
"""默认滑动窗口大小（token 数）：字节预算 ÷ 最坏字节/token = 12k。

窗口同时受两种上限约束，取更紧者：
- 请求体字节数 ≤ 65,535（硬限制，见 REQUEST_MAX_BYTES）：中文
  UTF-8 每字 3 字节、英文每 token 约 4 字节，12k token 窗口
  最坏 48KB 文本 + 批次问题与结构开销，仍在预算内；
- jev per-question 上下文 32k token：12k 窗口留足余量。

超长文本（如百万级 token）由滑动窗口 + 窗口级并行处理：
每窗 12k token、相邻 50% 重叠，窗口间在线程池上并发提交。
"""

DEFAULT_STRIDE_DIVISOR = 2
"""默认滑动步长因子：stride = window_size // 2（50% 重叠）。

词元级检测对窗口边界不敏感（重叠窗口保证跨边界实体至少单侧完整），
50% 重叠即可，比 BertNER 的 25% 步长节省一半请求。
"""

DEFAULT_MAX_WORKERS = 8
"""窗口级并行的默认线程数。

瓶颈在 TypeSafe API 的网络往返与限流而非本地 CPU，8 路并发
可在限流内显著压缩多窗口延迟；0 或 1 时退化为串行。
"""

DETECT_CACHE_CAPACITY = 128
"""检测结果的进程内 LRU 缓存容量（按窗口文本哈希键控）。"""

TYPING_CACHE_CAPACITY = 4096
"""分型结果的进程内 LRU 缓存容量（按窗口哈希 + span 键控）。"""

DEFAULT_PROB_THRESHOLD = 0.5
"""实体主类型（最相关类型）概率的默认过滤阈值。"""

SECONDARY_TYPE_THRESHOLD = 0.25
"""实体次要类型的概率下限。

Choice 的 probabilities 是跨选项的竞争分布；主类型（被选项）必须过
prob_threshold，其余类型达到该下限即作为次要类型一并返回，与主
类型一起按概率降序排列——一个实体可以同时具有多种类型（如一个
国家既是政区 GPE 又是地点 LOC），按相关性排序而不强制单一类型。
0.25 起步：实测大批次下碎片 span 的次要类型概率常在 0.1~0.2，
抬高下限避免噪声类型混入。"""

TYPING_BATCH_SIZE = 64
"""分型阶段单请求问题数。

分型需要精细语义判定，实测大批次会稀释 jev 对碎片 span 的 NONE
判定（批内问题互相干扰），小批次显著更稳。"""

DETECT_BATCH_SIZE = 64
"""检测阶段单请求问题数。

与分型批对齐：检测同样受批内干扰影响，实测 256 大批下中文双字
地名（"福建/福州"）的 Noul 值被稀释而漏检，64 批稳定。"""

DETECT_PROB_THRESHOLD = 0.25
"""词元检测 Noul 的判定阈值。

检测阶段宁滥勿缺：碎片/伪 span 由分型阶段的 NONE 一票否决
（含 prob_threshold 主阈值）在源头过滤，检测层放宽只增加候选
span 数量、不引入伪实体。实测中文高频单字（"二甲双胍"的"二"
等数字词）单轮检测分数可低至 0.3，阈值过高会整词漏检。"""

DETECT_GAP_TOLERANCE = 2
"""区间重组允许的漏检词元数。

细碎分词语言（泰文、阿拉伯文等）中，声调符号、词缀等单字符词元
常被检测阶段单独判否；容差允许桥接少量漏检词元，避免实体区间
因个别碎片漏检而断链。桥接进来的非实体片段由分型阶段的 NONE
兜底过滤。
"""

DETECT_RECHECK_LOW = 0.15
"""词元检测复检的灰色区下界。

Noul 值落在 [DETECT_RECHECK_LOW, 0.5) 的词元进入复检队列，
做三轮独立复检取最大值——真实 API 逐请求存在采样方差，实测
中文医药词元（如"二甲双胍"的"二/甲/双/胍"）单轮检测概率可低至
0.2~0.4，单轮 + 低阈值会整词漏检，三轮取最大显著压低漏检率。
"""

DETECT_RECHECK_ROUNDS = 3
"""灰色区词元复检轮数（含首轮共 1 + 3 轮采样，取最大值）。"""

MAX_ENTITY_TOKENS = 12
"""实体区间内子 span 的最大词元长度。

超过此长度的命名实体极罕见；子 span 长度以此截断。
"""

GENDER_BATCH_SIZE = 256
"""性别判定单请求问题数（性别问题极小，大批次无批内干扰问题）。"""

GENDER_CACHE_CAPACITY = 1024
"""性别判定结果的进程内 LRU 缓存容量。"""

GENDER_CRITERIA: dict[str, str] = {
    "MALE": "A name conventionally borne by men.",
    "FEMALE": "A name conventionally borne by women.",
    "UNKNOWN": "No clear gender signal, or a name commonly used for any gender.",
}
"""性别判定 Choice 问题的选项描述。"""


class _Cache(Protocol):
    """vortezwohl LRUCache 的类型化协议视图。

    vortezwohl 未随包发布类型标注，此处以结构化协议约束本 SDK 实际
    依赖的两个方法，保证 jevner 侧的类型安全。键值均为 Any，由
    各调用点自行收窄。
    """

    def read(self, key: object) -> object | None: ...

    def write(self, key: object, value: object) -> object | None: ...


class JevNER:
    """基于 jev 的滑动窗口多语言命名实体识别器。

    用法与 BertNER 对齐：构造一次，反复调用实例。凭据默认从环境
    变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 读取，也可显式传入。
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
        """初始化识别器。

        Args:
            model_name: TypeSafe 模型名或别名，默认 'jev-latest'
                （自动路由到最新稳定版 jev）。
            api_key: TypeSafe API Key，缺省时读取 TYPESAFE_API_KEY 环境变量。
            base_url: API 根地址，缺省时读取 TYPESAFE_BASE_URL 环境变量。
            timeout: 单请求超时秒数，缺省时使用 SDK 默认值。
            window_size: 滑动窗口大小（token 数），默认 24k
                （jev per-question 32k 减余量）。
            max_workers: 窗口级并行的线程数；0 或 1 时退化为串行。
            taxonomy: 实体类型体系（类型/包注册表与默认包），
                None 时用内置体系；C 端用户可传
                EntityTaxonomy.from_json(...) 的构建结果。

        Raises:
            ValueError: window_size 非正。
            TypeSafeError: API Key 缺失或非法。
        """
        if window_size < 1:
            raise ValueError(f"window_size ({window_size}) must be positive.")
        # 重试策略：对 408/429/5xx 与连接类错误做 3 次指数退避重试
        # （默认 RetryPolicy 总预算 30s 过短，实测服务端瞬态 500/520
        # 会被过早放弃），总预算放开由 max_retries 封顶
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            timeout=timeout,
            retry=RetryPolicy(max_retries=3, backoff_initial=1.0, backoff_max=8.0, timeout=None),
        )
        self._tokenizer = BundledTokenizer()
        self._window_size = window_size
        # 实体类型体系：类型/包注册表与默认包，识别管线全部类型
        # 语义的唯一来源；用户可经 JSON 扩展或覆盖
        self._taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        # 窗口级并行线程池：多窗口（超长文本）并发提交检测与分型请求。
        # 线程安全性：httpx2.Client 底层为连接池，多线程共享安全。
        self._max_workers = max(1, max_workers)
        self._pool = ThreadPool(max_workers=self._max_workers)
        # 三级进程内 LRU 缓存：检测（窗口 → 命中词元）、分型
        # （窗口 + span → 类型元组）、性别（人名 → 性别判定）。
        # 重复文本 / 重叠窗口 / 重复人名全部命中缓存，免于重复请求。
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
        """在单个窗口内检测实体词元区间。

        逐词元提交 Noul 问题（"该词元是否为命名实体组成部分"），按
        阈值筛出命中词元并按位置严格连续性重组为区间。对应 BertNER
        底座模型的 token 级预测，对任意语言的分词形态一视同仁。
        结果按窗口文本哈希写入进程内 LRU 缓存，重复窗口零请求。

        Args:
            window_text: 窗口文本（原文的精确子串，渲染 state 用）。
            tokens: 窗口词元列表，偏移锚定在原文上。

        Returns:
            (窗口全部词元, 命中词元索引, 严格连续区间) 三元组；
            词元为 (start, end, text)，start/end 为原文全局字符偏移。
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
        # 逐词元 Noul 概率；对灰色区（DETECT_RECHECK_LOW..HIGH）的
        # 不确定词元做第二轮复检，两轮任一过阈值即命中——真实
        # API 逐请求存在采样方差，实测同一 state 多轮极稳定，
        # 复检显著降低词元级漏检率（尤其孤立短词元如"北京"）。
        scores: dict[int, float] = {}
        recheck: list[int] = []
        indexed: list[tuple[int, tuple[int, int, str]]] = list(enumerate(tokens))
        for raw_batch in batched(indexed, DETECT_BATCH_SIZE):
            pairs = [p for p in raw_batch if p is not None]
            if not pairs:
                continue
            # 每个词元问题携带左右邻近词元：孤立单字/碎片是否属于
            # 实体高度依赖上下文（"吴"单看无法判定，左邻右邻
            # "子"才成立），邻近词元让检测阶段具备区间视野。
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
        # 灰色区词元多轮复检（DETect_RECHECK_ROUNDS 轮独立采样取最大值）
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
        # 命中词元按位置严格连续性重组为区间（双策略之一）。
        # 严格连续保证相邻实体不互相粘连："在北京"与"耿爽"各自
        # 独立成区间，"北京"不会被吞进粘介词 span。
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
        """对命中词元做容差桥接，得到第二套区间（双策略之二）。

        严格连续区间会把细碎分词语言（泰文、阿拉伯文）中被检测
        判否的声调符号、词缀切出词外，导致词内实体断成碎片。桥接
        区间允许跳过漏检词元（DETECT_GAP_TOLERANCE 内）把命中词元
        连成更长区间，跨过的漏检词元字符一并并入。两套区间并集
        交给分型阶段：严格区间保边界，桥接区间保词内完整实体，
        粘连产生的垃圾 span 由分型阶段的 NONE 兜底过滤。

        Args:
            tokens: 窗口全部词元。
            hits: 命中词元的索引列表（升序）。
            regions: 严格连续区间列表。

        Returns:
            桥接区间列表；与严格区间一起构成双策略并集。
        """
        bridged: list[list[tuple[int, int, str]]] = []
        current: list[tuple[int, int, str]] = []
        last_hit: int | None = None
        for i in hits:
            missed = i - last_hit - 1 if last_hit is not None else 0
            if current and last_hit is not None and missed <= DETECT_GAP_TOLERANCE:
                # 连续（missed == 0）延长区间；漏检在容差内则把
                # 漏检词元一并桥接进区间
                current.extend(tokens[last_hit + 1 : i])
                current.append(tokens[i])
            else:
                # 漏检超出容差：落盘当前区间并另起新区间
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
        """在单个窗口内完成检测与分型，返回实体列表。

        检测产出双策略区间（严格连续 + 容差桥接）的并集，每区间内
        生成全部连续子 span（长度至多 MAX_ENTITY_TOKENS 个词元），
        子 span 之间天然包含嵌套（如"中华人民共和国外交部"内的
        "中华人民共和国"），全部作为 Choice 问题批量提交 jev 分型。

        分型支持多类型：Choice 的 probabilities 是跨选项竞争分布，
        被选项作为主类型必须过 prob_threshold，其余类型概率达到
        SECONDARY_TYPE_THRESHOLD 即作为次要类型一并保留，与主类型
        一起按概率降序排列——不强制一个实体只有一种类型。

        Args:
            window_text: 窗口文本（原文的精确子串，渲染 state 用）。
            tokens: 窗口词元列表，偏移锚定在原文上。
            entity_types: 本次识别启用的实体类型集合。
            prob_threshold: 主类型概率过滤阈值。

        Returns:
            该窗口内识别出的实体列表（未跨窗口合并），偏移锚定原文。
        """
        _, hits, strict_regions = self._detect_regions(window_text, tokens)
        all_regions: list[list[tuple[int, int, str]]] = list(strict_regions)
        for region in self._bridge_regions(tokens, hits, strict_regions):
            if region not in all_regions:
                all_regions.append(region)
        if not all_regions:
            return []
        # 区间内生成全部连续子 span；偏移剥离首尾标点后同步修正，
        # 保证 span 文本始终是原文的精确子串（嵌套关联的锚点）
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
                    # 跳过空白、重复、以及完全不含字母数字的纯标点候选
                    if not key or (text, start, end) in seen:
                        continue
                    if not any(ch.isalnum() for ch in key):
                        continue
                    seen.add((text, start, end))
                    spans.append((text, start, end))
        if not spans:
            return []
        # 分型缓存命中：跳过已判定过的 span（重叠窗口 / 重复文本复用）
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
        # criteria 只携带标签不携带描述：类型描述已在 state 的
        # ENTITY_TYPES 表中出现一次（每问重复描述会把单问题从约
        # 150 字节膨胀到约 2KB，64 问一批即约 128KB，超过服务端
        # 65,535 字节请求体硬上限——实测 [3] 稳定 500 即此原因）。
        # 标签命名与类型表键一致，模型按键查表语义判定。
        criteria: dict[str, str | None] = {t.label: None for t in entity_types}
        criteria[NONE_LABEL] = None

        def _type_batch(
            pairs: list[tuple[int, str]],
        ) -> dict[int, tuple[tuple[EntityType, float], ...]]:
            """单批分型请求：返回 span 索引到降序类型元组的映射，空元组表示 NONE。"""
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
                # NONE 一票否决：jev 判定该 span 不是实体时，
                # 不再收集任何次要类型——否则碎片 span 会借
                # 低概率次要类型绕过否决混入结果（实测 0.25~0.5
                # 的碎片噪声即从该路径漏入）
                if answer.choice == NONE_LABEL:
                    batch[i] = ()
                    continue
                # 主类型（被选项）必须过 prob_threshold；
                # 其余类型过 SECONDARY_TYPE_THRESHOLD 即为次要类型
                best_per_type: dict[EntityType, float] = {}
                primary_prob = float(answer.probabilities.get(answer.choice, 0.0))
                if primary_prob < prob_threshold:
                    batch[i] = ()
                    continue
                primary = self._taxonomy._type_of(answer.choice)
                best_per_type[primary] = primary_prob
                for label, prob in answer.probabilities.items():
                    # NONE 是否决选项不是实体类型，不进类型收集
                    if label == NONE_LABEL or label == answer.choice:
                        continue
                    if prob < SECONDARY_TYPE_THRESHOLD:
                        continue
                    etype = self._taxonomy._type_of(label)
                    if etype not in best_per_type or prob > best_per_type[etype]:
                        best_per_type[etype] = float(prob)
                batch[i] = tuple(sorted(best_per_type.items(), key=lambda kv: -kv[1]))
            return batch

        # 分型批次并行提交（每批 TYPING_BATCH_SIZE 个 span）；
        # 批内判定共享同一 state，批间在线程池上并发。
        # batched() 以 fill_value=None 补齐尾批，需过滤 None 元素。
        batches = [
            [item for item in batch if item is not None]
            for batch in batched(uncached, TYPING_BATCH_SIZE)
        ]
        batches = [batch for batch in batches if batch]
        if len(batches) <= 1 or self._max_workers <= 1:
            batch_results = [_type_batch(pairs) for pairs in batches]
        else:
            # ThreadPool.map 会对 Iterable 参数做 *param 解包，这里
            # 用单元素 tuple 包装，使 _type_batch 收到完整的批次列表
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
        # 汇总：缓存命中 + 本轮判定，NONE（空元组）不产出实体
        all_ranked: dict[int, tuple[tuple[EntityType, float], ...]] = dict(cached_entities)
        for result in batch_results:
            all_ranked.update(result)
        for i, (text, start, end) in enumerate(spans):
            ranked = all_ranked.get(i, ())
            if ranked:
                entities.append(Entity(text=text, types=ranked, start=start, end=end))
        return entities

    def _combine(self, entities: list[Entity], deduplicate: bool) -> list[Entity]:
        """跨窗口实体合并（按出现位置去重与类型并集）。

        规则：

        - 同一出现位置（start/end 相同）：跨窗口/跨批次重复识别的同
          一实体，合并为一个，多类型取并集，同类型取最高概率；
        - 不同出现位置：各自独立保留——含包含关系的实体是嵌套实体
          （如 GPE"福建"[16:18] 内嵌于 GPE"福建福州"[16:20]），合并
          阶段不做任何吸收或吞并，嵌套关联完全交由输出侧
          entity_tree() 依据偏移包含关系派生。

        碎片噪声（检测方差产生的 PERSON"吴"之类）不在合并阶段拦截——
        由分型阶段的 NONE 一票否决在源头过滤。

        Args:
            entities: 全部窗口的实体平铺列表。
            deduplicate: 是否对同位置重复实体做去重合并（类型并集）。

        Returns:
            合并后的实体列表，按 (start, end) 排序；每实体 types
            按概率降序，text 为原文精确子串。
        """
        if not deduplicate:
            return sorted(entities, key=lambda e: (e.start, e.end, e.text))
        # 按出现位置分组：同位置的跨窗口/跨批次判定合并为一个实体，
        # 类型并集、同类型取最高概率；不同位置一律独立保留
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
        """批量判定人名性别（带进程内缓存）。

        Args:
            names: 人名列表（允许重复，结果按键去重）。

        Returns:
            人名到 (性别, 概率) 的映射。
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
        """对文本执行命名实体识别，返回实体嵌套森林。

        完整管线：原文偏移词元 → 滑动窗口 → 逐词元 Noul 检测
        （灰色区多轮复检）→ 区间子 span Choice 分型（NONE 一票
        否决 + 多类型收集）→ 同位置去重合并 → PERSON 升级 Person
        → 偏移包含关系建树。

        Args:
            s: 待识别文本，任意语言。
            with_gender: 是否对 PERSON 实体附加性别判定。
            entity_packs: 启用的实体包（对象或名称字符串）列表，
                缺省时仅启用基础包 CORE。
            entity_types: 额外显式启用的实体类型，与包并集叠加。
            prob_threshold: 类型概率过滤阈值，[0, 1]。
            fast_mode: True 时整段文本作为单一窗口直接判定（要求文本
                在 jev 上下文限制内）；False 时启用滑动窗口长文本算法。
            window_size: 覆盖实例级窗口大小（token 数）。
            stride: 覆盖滑动步长，缺省为 window_size // 2。
            deduplicate: 是否对跨窗口重复实体按 (text, type) 去重。

        Returns:
            实体嵌套森林：顶层（不被任何实体包含）EntityNode 列表，
            按 (start, end) 排序；每节点的 entity 为实体本体（PERSON
            实体在 with_gender=True 时为带性别字段的 Person），
            children 为偏移落在其区间内的内层实体。嵌套关系直接
            内含于返回值，无需下游再做后处理。

        Raises:
            AssertionError: prob_threshold 不在 [0, 1] 内。
            ValueError: entity_packs 中的名称字符串不是已注册的实体包、
                实体类型解析为空或窗口参数非法。
            TypeSafeError: 请求构造失败（如凭据缺失）。
            TypeSafeAPIError: API 返回错误响应。
            TypeSafeAPIConnectionError: 网络连接失败。
        """
        if not isinstance(s, str) or not s.strip():
            return []
        assert 0.0 <= prob_threshold <= 1.0, (
            f"prob_threshold ({prob_threshold}) must be within [0, 1]."
        )
        # 实体类型解析：包选择 + 显式类型叠加（taxonomy 是语义唯一来源；
        # PERSON 之类固定语义标签在体系内注册，非 PERSON 不做性别判定）
        types = self._taxonomy.resolve(entity_packs, entity_types)
        ws = window_size if window_size is not None else self._window_size
        st = stride if stride is not None else max(1, ws // DEFAULT_STRIDE_DIVISOR)
        if ws < 1 or st < 1:
            raise ValueError(f"window_size ({ws}) and stride ({st}) must be positive.")
        # 窗口按原文 token 偏移切分，窗口文本取原文字符区间（词元
        # 文本含空白前缀时会互相交叠，join 重建会产生重复字符），
        # 词元偏移全部锚定在原文上——span 与实体因此携带原文全局
        # 偏移，嵌套关联由偏移包含关系可靠判定
        if fast_mode:
            windows = [(s, self._tokenizer.window_tokens(s))]
        else:
            windows = self._tokenizer.windows(s, ws, st)
        # 请求体字节防护：窗口文本是请求体的大头，超预算直接报错
        # （服务端 65,535 字节硬限制的超限请求是 500，报错不指向
        # 根因，这里给出可操作的指引）
        for window_text, _toks in windows:
            if len(window_text.encode("utf-8")) > REQUEST_STATE_BUDGET:
                raise ValueError(
                    "A sliding window's UTF-8 size exceeds the request body budget "
                    f"({REQUEST_STATE_BUDGET} bytes of {REQUEST_MAX_BYTES} allowed). "
                    f"Use a smaller window_size (current {ws} tokens)."
                )
        # 窗口级并行：每窗独立完成检测 + 分型（检测/分型批已在
        # 线程池上并发），窗口间再并发提交；串行配置或单窗口退化为顺序执行
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
            # PERSON 宇实升级为 Person（附带性别判定），再建树；
            # PERSON 标签由 taxonomy 保证存在（开放体系的固定语义锚点）
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
        # 按偏移包含关系重建嵌套树：返回值自身即含关联结构
        return entity_tree(merged)

    def close(self) -> None:
        """释放底层 HTTP 客户端。"""
        self._client.close()

    def __enter__(self) -> Self:
        """进入上下文管理器。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """退出上下文管理器并释放资源。"""
        self.close()
