"""随包分发的 Qwen3 分词器封装。

本模块封装随 SDK 一同打包发行的 Qwen3 tokenizer（字节级 BPE，对包括
CJK、泰文、阿拉伯文在内的上百种语言均有良好覆盖），承担 jevner 全部
本地文本处理：token 计数、长文本滑动窗口切分与词元字符定位。

调用方与维护者进入本文件时最需要先知道：
- 分词器文件位于包内 assets/qwen3_tokenizer.json，由包数据随 SDK 一起
  安装，用户无需访问 HuggingFace，也无需安装 transformers/torch；
- 候选实体不依赖固定 n-gram：检测阶段逐词元判断是否属于实体，区间
  重建与子 span 生成都基于词元的字符偏移量，对无空格文字（中文/
  泰文）与粘着语（阿拉伯文/马来文）一视同仁；
- 滑动窗口与 BertNER 同构：按 token 序列取窗口、逐窗口 decode 回
  文本，窗口边界永远落在 token 边界上，不会把多 token 实体从字符
  中间劈开。
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from tokenizers import Encoding, Tokenizer

_ASSET_PATH = Path(__file__).parent / "assets" / "qwen3_tokenizer.json"
"""随包分发的 Qwen3 tokenizer 文件路径。"""


def _is_edge_junk(ch: str) -> bool:
    """判断字符是否为应从实体边界剥离的空白或标点（货币符号等保留）。"""
    return ch.isspace() or unicodedata.category(ch)[0] == "P"


def strip_edge_punctuation(text: str) -> str:
    """剥离文本首尾的空白与标点（保留货币符号等其他字符）。

    词元偏移切片常把紧邻标点带进候选文本（如 "子豪,"），不剥离会
    污染实体文本并破坏跨 span 的包含判定。

    Args:
        text: 待清洗的候选 span 文本。

    Returns:
        清洗后的文本；可能为空字符串。
    """
    start, end = 0, len(text)
    while start < end and _is_edge_junk(text[start]):
        start += 1
    while end > start and _is_edge_junk(text[end - 1]):
        end -= 1
    return text[start:end]


def strip_span(s: str, start: int, end: int) -> tuple[str, int, int]:
    """剥离区间首尾的空白与标点，偏移随文本同步修正。

    与 strip_edge_punctuation 的区别：作用在原文区间上，返回
    (清洗后文本, 新起点, 新终点)。新偏移保证
    text == s[new_start:new_end]，使实体文本始终是原文的精确
    子串——嵌套关联由偏移包含关系判定，偏移锚定必须与文本严格一致。

    Args:
        s: 原文文本。
        start: 区间起点（含），越界时钳制到 [0, len(s)]。
        end: 区间终点（不含），越界时钳制到 [0, len(s)]。

    Returns:
        (清洗后文本, 清洗后起点, 清洗后终点) 三元组。
    """
    start = max(0, min(start, len(s)))
    end = max(0, min(end, len(s)))
    while start < end and _is_edge_junk(s[start]):
        start += 1
    while end > start and _is_edge_junk(s[end - 1]):
        end -= 1
    return s[start:end], start, end


class BundledTokenizer:
    """基于随包 Qwen3 分词器的轻量封装。

    提供 jevner 算法所需的本地能力：token 序列化、滑动窗口文本切分、
    词元字符定位。全部为纯 CPU 的确定性本地计算，不发起任何网络请求。
    """

    def __init__(self, asset_path: Path = _ASSET_PATH) -> None:
        """加载随包分发的分词器文件。

        Args:
            asset_path: tokenizer.json 文件路径，默认使用包内资产。

        Raises:
            FileNotFoundError: 分词器资产文件不存在（发行包损坏）。
            Exception: 分词器文件解析失败时由 tokenizers 库向上抛出。
        """
        if not asset_path.exists():
            raise FileNotFoundError(f"Bundled tokenizer asset not found: {asset_path}")
        self._tokenizer = Tokenizer.from_file(str(asset_path))

    def encode(self, s: str) -> Encoding:
        """将文本编码为 token 序列（不含特殊 token）。

        Args:
            s: 待编码文本。

        Returns:
            tokenizers.Encoding，含 .ids 与字符级 .offsets。
        """
        return self._tokenizer.encode(s, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        """将 token 序列解码回文本。

        Args:
            ids: token id 列表。

        Returns:
            解码后的文本（跳过特殊 token）。
        """
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def count_tokens(self, s: str) -> int:
        """统计文本的 token 数。

        Args:
            s: 待统计文本。

        Returns:
            token 数量。
        """
        return len(self.encode(s).ids)

    def window_texts(self, s: str, window_size: int, stride: int) -> list[str]:
        """按 token 序列对长文本做滑动窗口切分。

        与 BertNER 的 _slow_extract 同构：文本 token 数不超过窗口大小时
        整段作为单个窗口；超过时按 stride 步进取 token 切片并逐窗口
        decode 回文本，相邻窗口保留重叠，保证跨窗口边界的实体至少在
        一个窗口内完整出现。

        Args:
            s: 待切分文本。
            window_size: 窗口大小（token 数）。
            stride: 滑动步长（token 数），应小于 window_size 以保留重叠。

        Returns:
            窗口文本列表，按原文顺序排列。
        """
        enc = self.encode(s)
        if len(enc.ids) <= window_size:
            return [s]
        texts: list[str] = []
        for start in range(0, len(enc.ids), stride):
            chunk = enc.ids[start : start + window_size]
            if not chunk:
                break
            texts.append(self.decode(chunk))
        return texts

    def window_tokens(self, s: str) -> list[tuple[int, int, str]]:
        """返回文本全部词元的字符偏移与文本。

        词元级检测（逐词元 Noul）之后，代码侧用本结果在字符层重建
        连续实体区间，是替代固定 n-gram 的语言无关基础。

        Args:
            s: 窗口文本。

        Returns:
            (字符起点, 字符终点, 词元文本) 列表，按文本顺序排列。
        """
        enc = self.encode(s)
        return [(start, end, s[start:end]) for (start, end) in enc.offsets]

    def windows(
        self, s: str, window_size: int, stride: int
    ) -> list[tuple[str, list[tuple[int, int, str]]]]:
        """按 token 序列对长文本做滑动窗口切分，返回原文窗口。

        与 window_texts 的区别：窗口不经 decode 往返，直接取原文
        字符区间 s[c_start:c_end] 作为窗口文本，窗口内词元偏移就是
        原文全局偏移（text == s[start:end] 恒成立）。下游 span 因此
        携带原文偏移，实体嵌套关联由偏移包含关系可靠判定。

        注意词元文本不能重新拼接成窗口文本：Qwen3 对空白开头的词元
        （如 " 我"）与后续词元存在字符交叠，join 会产生重复字符，
        使检测 state 与词元错位。窗口文本必须从原文字符区间切出。

        切分策略与 window_texts 一致：token 数不超过窗口大小时
        整段作为单个窗口；超过时按 stride 步进取词元切片，窗口文本
        覆盖首个到最后词元的字符区间，相邻窗口保留重叠。

        Args:
            s: 待切分文本。
            window_size: 窗口大小（token 数）。
            stride: 滑动步长（token 数），应小于 window_size 以保留重叠。

        Returns:
            (窗口文本, 窗口词元列表) 元组列表；词元为
            (原文起点, 原文终点, 词元文本)，偏移即原文全局偏移。
        """
        enc = self.encode(s)
        offsets = enc.offsets
        if len(offsets) <= window_size:
            return [(s, [(start, end, s[start:end]) for start, end in offsets])]
        result: list[tuple[str, list[tuple[int, int, str]]]] = []
        for base in range(0, len(offsets), stride):
            chunk = offsets[base : base + window_size]
            if not chunk:
                break
            c_start, c_end = chunk[0][0], chunk[-1][1]
            window_text = s[c_start:c_end]
            result.append((window_text, [(start, end, s[start:end]) for start, end in chunk]))
        return result
