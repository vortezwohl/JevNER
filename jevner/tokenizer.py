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
    while start < end and (text[start].isspace() or unicodedata.category(text[start])[0] == "P"):
        start += 1
    while end > start and (
        text[end - 1].isspace() or unicodedata.category(text[end - 1])[0] == "P"
    ):
        end -= 1
    return text[start:end]


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
