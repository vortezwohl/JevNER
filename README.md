# JevNER

无本地模型的多语言命名实体识别 SDK：用 TypeSafe System One 模型 **jev**（"超级 BERT"，默认 `jev-latest` 自动路由最新稳定版）替代全部本地 BERT 推理，参考 [DeepLoTX BertNER](https://github.com/vortezwohl/DeepLoTX) 的长文本滑动窗口算法，随包分发 Qwen3 分词器。

## 特性

- **全球语种**：词元检测 + 区间分型两阶段算法，不依赖固定 n-gram 假设；实测覆盖亚洲/欧洲/美洲 24 语种（汉/日/韩/泰/越/印尼/马来/菲律宾/印地/阿拉伯/希伯来/土耳其 + 英法德西葡意俄荷波 + 美洲三语），无空格文字与粘着语均正确识别；
- **上下文窗口**：默认窗口 24k token（受 jev per-question 32k 上限约束），超长文本由滑动窗口 + 窗口级并行处理；
- **多类型实体**：一个实体可同时具有多种类型（国家既是政区 GPE 又是地点 LOC），`types` 按相关性（概率）降序排列，不强制单一类型；`type`/`base_probability` 属性兼容单类型访问；
- **无本地模型**：不加载任何神经网络权重，唯一推理依赖是 TypeSafe API（`jev-latest`）；
- **随包分词器**：Qwen3 字节级 BPE 分词器（覆盖 100+ 语言含 CJK）打包在包内 `assets/`，无需访问 HuggingFace，无需 transformers/torch；
- **实体包体系**：基础包 CORE（OntoNotes 十八类 + MISC）默认启用，垂直领域包 BIOMED（疾病/药物/基因/蛋白质/化学）、FINANCE（证券/指数/货币）、TECH（软件/协议/平台）按需叠加；
- **嵌套实体树**：返回值即 `list[EntityNode]` 森林，偏移包含关系派生（交叉嵌套挂最近公共外层成兄弟），`print(node)` 直接得到可读树形，无需下游后处理；
- **性别识别**：PERSON 实体自动附加性别判定（MALE/FEMALE/UNKNOWN），进程内 LRU 缓存；
- **typesafe**：全量类型标注（`py.typed`），mypy 零错误，提示词按结构化提示词设计方法论（七层语义模型）经 prompt4py 渲染，ruff 全量格式化。

## 安装

```bash
uv add jevner        # 或 pip install jevner
```

## 快速开始

```python
from jevner import JevNER

ner = JevNER()  # 凭据读取环境变量 TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# 返回值即实体嵌套森林，print 直接得到可读树形
forest = ner("你好, 我的名字是吴子豪, 来自福建福州.")
for node in forest:
    print(node)
# 吴子豪 (PERSON 1.00 | MALE 1.00) [9:12]
# 福建福州 (GPE 1.00) [16:20]
# ├── 福建 (GPE 1.00) [16:18]
# └── 福州 (GPE 1.00) [18:20]

# 垂直领域：生物医学实体包
forest = ner("患者确诊为二型糖尿病, 服用二甲双胍控制血糖.", entity_packs=["BIOMED"])
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 二甲双胍 (DRUG 1.00) [14:18]
```

返回值是 `list[EntityNode]`：顶层（不被包含的实体）节点按 `(start, end)` 排序，`node.entity` 为实体本体，`node.children` 递归内层实体。`Entity.types` 是 `(EntityType, probability)` 元组序列（概率降序，一实体可多类型）；`Entity.type` / `base_probability` 取首项；`start` / `end` 是原文全局字符偏移（恒满足 `text == source[start:end]`），嵌套即由偏移包含关系派生。

## API

```python
JevNER(model_name="jev-latest", *, api_key=None, base_url=None, timeout=None,
       window_size=24576)  # jev per-question 上限 32k 减问题余量

ner(s, *, with_gender=True, entity_packs=None, entity_types=None,
    prob_threshold=0.5, fast_mode=False, window_size=None, stride=None,
    deduplicate=True) -> list[EntityNode]
```

| 参数 | 说明 |
| --- | --- |
| `entity_packs` | 实体包列表（对象或名称字符串），缺省 `["CORE"]`；可传 `["CORE", "BIOMED"]` 叠加 |
| `entity_types` | 额外显式启用的 `EntityType`，与包并集叠加 |
| `with_gender` | 是否对 PERSON 实体附加性别判定 |
| `prob_threshold` | 主类型概率过滤阈值，[0, 1]，默认 0.5 |
| `fast_mode` | `True` 时整段单窗口直接判定（要求文本在 jev 上下文限制内） |
| `window_size` / `stride` | 滑动窗口大小与步长（token 数），默认 24k / 窗口一半 |

## 算法

```
原文 ──Qwen3 分词──> 原文偏移词元 ──滑动窗口(24k, 50% 重叠)──> 原文区间窗口
                                                        │
     检测：逐词元 Noul（"是否为命名实体组成部分"，带左右邻近词元，
           灰色区二次复检）→ 命中词元 → 双策略区间（严格连续 + 容差桥接并集）
                                                        │
     分型：区间内全部连续子 span → Choice（启用实体类型 + NONE）
           → jev 并行判定，主类型过阈值 + 次要类型（≥0.25）一并返回，
             NONE 过滤粘连/词中截断的垃圾 span
                                                        │
     合并：同出现位置合并为类型并集（同类型取最高概率）；
           同主类型交叠碎片由概率最高者吸收，嵌套两侧保留
                                                        │
     PERSON → Choice(MALE/FEMALE/UNKNOWN) 性别判定（LRU 缓存）
```

与 BertNER 的对应：本地 BERT token 级 BIO 预测 → jev 词元 Noul 检测 + 子 span Choice 分型；固定 n-gram 候选 → 检测区间驱动子 span；448 token 小窗口 → 24k token 窗口 + 窗口级并行；单类型输出 → 多类型按相关性排序输出；平铺列表 → 原文偏移锚定 + entity_tree 嵌套树；`Davlan/xlm-roberta-base-ner-hrl` → `jev-latest`。

## 测试

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # 格式与静态检查
uv run mypy jevner                          # 类型检查
uv run python tests/test_local.py           # 本地算法单测（无网络，9 项）
uv run python tests/test_e2e.py             # 端到端（需 TYPESAFE_API_KEY，6 项，产生计费）
uv run python tests/test_languages.py       # 语言覆盖扫描（24 语种，产生计费）
```