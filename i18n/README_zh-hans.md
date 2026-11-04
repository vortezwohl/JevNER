<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**零样本多语言命名实体识别。由 System One 判决模型驱动。**

*一个类型安全的 Python SDK，把判决模型变成命名实体识别器——嵌套实体树、自定义实体包、内置 Qwen3 分词器，无需训练或部署任何本地模型。*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="72" />
  </a>
  <br/>
  <strong>由 <a href="https://www.easyjev.online/">EasyJev</a> 赞助 —— 所有判决模型，一个端点。</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**免训练。任意语言。树就在返回值里。**

[English](../README.md) | 简体中文 | [繁體中文](README_zh-hant.md) | [日本語](README_ja-jp.md) | [한국어](README_ko-kr.md) | [Français](README_fr-fr.md) | [Español](README_es-es.md) | [Deutsch](README_de-de.md) | [Português](README_pt-br.md)

</div>

---

## 为什么选择 JevNER

大多数 NER 技术栈都逼你二选一：本地模型把标签集冻结在微调那一刻，或者托管 API 把真实置信度藏起来。

JevNER 走第三条路。一个 **System One 判决模型**——而不是文本生成器——阅读每一个问题并返回**校准过的概率**而非散文，SDK 再把这些判决组装成一棵可以直接打印成树的**嵌套实体森林**。

| 传统 NER 的短板 | JevNER 的做法 |
| --- | --- |
| 标签集在微调时冻结；每换一个领域就要重训一次。 | **天生零样本。** 实体类型是开放数据：任何用户（包括不会编程的人）都能在运行期通过一份普通 JSON 文档增改类型和实体包。 |
| 多语言覆盖意味着每种语言一个微调检查点。 | **一条流水线，已验证 24 种语言**——中文、泰语、阿拉伯语、希伯来语、印地语、土耳其语、印尼语、菲律宾语、越南语、日语、韩语、英语、法语、德语、西班牙语、葡萄牙语、意大利语、俄语、荷兰语、波兰语等，含低资源与无空格文字。 |
| 托管 NER API 返回扁平 span；嵌套被压平或丢弃。 | **返回值就是树。** `node.children` 持有 `node` 内包含的每个实体，由原文偏移推导（`text == source[start:end]`），嵌套是结构性的，不是猜的。 |
| softmax 分数不是可以拿来做路由的概率。 | **校准概率**来自判决模型：`0.8` 的含义是"十次里对八次"，这正是置信度门控人工复核流程真正需要的东西。 |
| 一个 span 只有一个类型，哪怕一个名字既是地点又是机构。 | **多类型实体，按概率排序。** `Entity.types` 携带所有竞争类型及其概率，按相关度降序。 |
| 每次部署都要下载模型权重。 | **完全无需本地模型。** 包内自带 Qwen3 分词器（覆盖 100+ 语言）；所有推理经 EasyJev 网关完成。 |

## 安装

- 从 [PyPI](https://pypi.org/project/jevner/) 安装

    ```shell
    pip install -U jevner
    ```

- 推荐：使用 [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- 从 [GitHub](https://github.com/vortezwohl/JevNER) 安装——抢先体验未发布功能。

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## 快速上手

```python
from jevner import JevNER

ner = JevNER()  # 凭据从 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 环境变量读取

# 返回值就是嵌套森林：直接打印即可渲染成树。
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

这就是完整的第一闭环：不下载模型、不微调、不写后处理。PERSON 实体会升级为带性别判决的 `Person`，偏移永远满足 `text == source[start:end]`，嵌套关系就装在返回值里。

> [!TIP]
> 网关路由了多个 System One 模型。默认是 `baseten/mammoth-1`；随时用 `JevNER(model_name="jev-latest")` 切换。

> 想用 TypeScript、Go 或裸 `curl` 调用？线协议就是纯 JSON REST——见 [`examples/`](../examples/README.md)。

## 核心能力

| 能力 | 你能得到什么 |
| --- | --- |
| 零样本识别 | 不训练、不标注、不微调。定义好要抽什么，直接开调。 |
| 判决模型地基 | Noul / Choice 问题携带校准概率——是/否置信度与完整选项分布，不是自由文本生成。 |
| 嵌套实体森林 | `ner(text)` 返回 `list[EntityNode]`。交叉 span 以兄弟身份挂到最近公共祖先下；嵌套由偏移推导，不做子串猜测。 |
| 校准的多类型实体 | `Entity.types` 是按相关度排序的 `(EntityType, probability)` 元组序列——一个实体可以既是 GPE *又是* LOC。 |
| 24 种语言覆盖 | 在亚洲、欧洲、美洲端到端验证，含低资源与无空格文字。 |
| JSON 开放词表 | `EntityTaxonomy.from_json` 接受三段式文档（`entity_types`、`entity_packs`、`default_packs`），可扩展或覆盖 31 个内置类型与 4 个实体包——为不写代码的用户而生。 |
| 垂直实体包 | 默认携带 `CORE`（18 个 OntoNotes 5.0 类型 + MISC）；`BIOMED`、`FINANCE`、`TECH` 按需叠加。 |
| 性别识别 | PERSON 实体升级为 `Person`，携带 `MALE` / `FEMALE` / `UNKNOWN` 与概率，进程内缓存。 |
| 内置 Qwen3 分词器 | 覆盖 100+ 语言的字节级 BPE 随包分发——不访问 HuggingFace、不需要 `transformers`、不需要 `torch`。 |
| 请求预算防护 | 窗口按网关 65,535 字节请求体上限裁切；分型批次按 512 选项上限收窄——每个请求在构造上就不会越界。 |
| 三级缓存 | 检测、分型、性别结果按窗口哈希存入 LRU 缓存；重复或交叠文本毫秒级返回。 |
| 全程类型安全 | 完整标注并携带 `py.typed`，严格模式 mypy 零错误；提示词经结构化 prompt4py 模板渲染。 |

## 常用模式

### 打印森林，而不是扁平列表

返回值已经是树。`str(node)` 用制表符线条渲染节点及其后代——类型、概率、性别、偏移一应俱全。

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### 启用垂直实体包

实体包按名叠加；发给模型的类型表就是你选择的集合的并集。

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### 用 JSON 定义词表

任何用户——包括从没打开过代码编辑器的人——都能用一份 JSON 文档扩展识别词表。自定义类型与内置类型平起平坐。

```python
from jevner import EntityTaxonomy, JevNER

taxonomy = EntityTaxonomy.from_json("""
{
  "entity_types": {
    "BRAND": "A brand or trademark name (e.g. Apple, Nike)."
  },
  "entity_packs": {
    "ECOM": ["PERSON", "ORG", "BRAND"]
  },
  "default_packs": ["CORE", "ECOM"]
}
""")

ner = JevNER(taxonomy=taxonomy)
forest = ner("Tim Cook announced that Apple launched the new iPhone while Nike sponsored it.")
```

`from_json` 接受 JSON 字符串或已解析的 dict。未知类型引用、空描述、未注册的默认包都会携带出错字段路径抛出 `ValueError`。

### 按校准置信度路由

每个概率都来自以校准为训练目标的判决模型，所以阈值是名副其实的：

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 真的是"十次对八次"
```

同一文本的重复调用毫秒级命中缓存——对 RAG 重查询和文档重跑很友好。

## 判决模型

JevNER 可以运行在网关上注册的任何 System One 判决模型上——这类模型用校准概率回答类型化问题（Noul 是/否、Choice 选择），而不是生成文本。默认 `baseten/mammoth-1`；`jev-latest` 及另外 20+ 个模型只需一个参数：

| | |
| --- | --- |
| NER 已验证 | `baseten/mammoth-1`（默认）、`jev-latest`、`jev-1.13.0`、`jev-1.14.0` |
| 最便宜 | `baseten/joey-1`、`respan/span-01-lite`、`perplexity/pplx-decider-flash-v1.2` —— 输入低至 $0.01/百万词元 |
| 同样可用 | Cloudflare Clef 家族、Nace.AI Drex、LiquidAI d1、Microsoft Decision-1、Upstage Solar Decide、Inception Mercury Decide、OpenAI GPT-6 Luna Decisions、`jaredpalmer/kev-4b` 等 |

含价格与逐模型注意事项的完整目录：[支持的判决模型](../docs/models.md)。

## 生态与对比

JevNER 是判决模型（System One）浪潮的参考应用——这个生态里的模型和引擎返回类型化的校准判决而非散文：TypeSafe **Jev** 家族、[EasyJev](https://www.easyjev.online/) 聚合网关、[laya](https://github.com/NandhaKishorM/laya)（非自回归决策引擎，100+ 语言）与 [kev](https://github.com/jaredpalmer/kev)（基于 Qwen 的开源决策模型）这样的开源引擎、以及 [AnyJev](https://github.com/nokia-applied-research/AnyJev) 这样的适配器。全景图：[判决模型生态](../docs/ecosystem.md)。

正在选型 NER 技术栈？看 [JevNER 对比 GLiNER、spaCy 与托管 NER API](../docs/alternatives.md)——一份诚实的对比，包括什么时候本地 [GLiNER](https://github.com/urchade/GLiNER) 检查点或 [spaCy](https://github.com/explosion/spaCy) 流水线才是更优解。

## 公开 API

根包刻意只暴露八个名字：

```python
from jevner import (
    Entity,          # 已识别实体: text, types, start, end
    EntityNode,      # 树节点: entity + children
    EntityPack,      # 实体类型的具名捆绑 (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # 开放类型系统: 注册表 + from_json + resolve
    EntityType,      # 标签 + 描述的值对象
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # 识别器本身
    Person,          # Entity + gender + gender_probability
)
```

## 设计边界

JevNER 只拥有识别流水线及其契约。它刻意不提供：

- 任何形式的本地模型推理；
- 模型托管、微调或权重分发；
- 训练数据、标注工具或评测框架；
- UI 组件；以及
- 网关 SDK 内部的再导出。

这条边界让 SDK 保持为一条薄薄、可验证的流水线，把托管与产品体验留给平台。

## 赞助商

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**所有 System One 模型，一个端点——完美兼容 [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python)。**

EasyJev 聚合各家 System One 判决模型——Jev、Decider、Drex、d1、Clef、kev 等——收敛在一个端点之后，官方 [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) 无需任何改动即可直连：把 `base_url` 和 `api_key` 指向网关，每次请求任选模型；已有模型名（`jev-latest`、`jev-1.14.0`……）原样可用。判决毫秒级返回，所有模型输出词元一律 $0 计费；邮箱验证码即可登录——首次登录自动建号并赠 $1 体验额度。充值支持信用卡（Visa / Mastercard）、Apple Pay、Google Pay、微信支付，一次性支付、无需订阅。

[立即注册 &rarr;](https://www.easyjev.online/) &middot; [模型广场](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [定价](https://www.easyjev.online/#/pricing) &middot; [服务条款](https://www.easyjev.online/#/tos) &middot; [隐私政策](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## 开发

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # 风格与静态检查
uv run mypy jevner                          # 类型检查
uv run python tests/test_local.py           # 离线算法测试 (12 项)
uv run python tests/test_e2e.py             # 端到端 (需要 TYPESAFE_API_KEY, 产生真实计费)
uv run python tests/test_languages.py       # 24 语言扫描 (产生真实计费)
uv run python tests/bench_quick.py          # 紧凑性能实验 (产生真实计费)
```

## 参与贡献

贡献应改进可验证的流水线行为：检测召回、分型精度、嵌套保真、词表校验、缓存行为或请求预算安全。保持公开 API 精简，并为新行为添加可独立运行的测试。

## 许可证

[MIT](../LICENSE)
