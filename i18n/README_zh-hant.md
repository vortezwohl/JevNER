<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**零樣本多語言命名實體識別。由 System One 判決模型驅動。**

*一個型別安全的 Python SDK，把判決模型變成命名實體識別器——巢狀實體樹、自訂實體包、內建 Qwen3 分詞器，無需訓練或部署任何本地模型。*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="72" />
  </a>
  <br/>
  <strong>由 <a href="https://www.easyjev.online/">EasyJev</a> 贊助 —— 所有判決模型，一個端點。</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**免訓練。任意語言。樹就在回傳值裡。**

[English](../README.md) | [简体中文](README_zh-hans.md) | 繁體中文 | [日本語](README_ja-jp.md) | [한국어](README_ko-kr.md) | [Français](README_fr-fr.md) | [Español](README_es-es.md) | [Deutsch](README_de-de.md) | [Português](README_pt-br.md)

</div>

---

## 為什麼選擇 JevNER

大多數 NER 技術棧都逼你二選一：本地模型把標籤集凍結在微調那一刻，或者託管 API 把真實置信度藏起來。

JevNER 走第三條路。一個 **System One 判決模型**——而不是文字產生器——閱讀每一個問題並回傳**校準過的機率**而非散文，SDK 再把這些判決組裝成一棵可以直接印成樹的**巢狀實體森林**。

| 傳統 NER 的短板 | JevNER 的做法 |
| --- | --- |
| 標籤集在微調時凍結；每換一個領域就要重訓一次。 | **天生零樣本。** 實體類型是開放資料：任何使用者（包括不會寫程式的人）都能在執行期透過一份普通 JSON 文件增改類型與實體包。 |
| 多語言覆蓋意味著每種語言一個微調檢查點。 | **一條流水線，已驗證 24 種語言**——中文、泰文、阿拉伯文、希伯來文、印地文、土耳其文、印尼文、菲律賓文、越南文、日文、韓文、英文、法文、德文、西班牙文、葡萄牙文、義大利文、俄文、荷蘭文、波蘭文等，含低資源與無空格文字。 |
| 託管 NER API 回傳扁平 span；巢狀被壓平或丟棄。 | **回傳值就是樹。** `node.children` 持有 `node` 內包含的每個實體，由原文偏移推導（`text == source[start:end]`），巢狀是結構性的，不是猜的。 |
| softmax 分數不是可以拿來做路由的機率。 | **校準機率**來自判決模型：`0.8` 的含義是「十次裡對八次」，這正是置信度門控人工覆核流程真正需要的東西。 |
| 一個 span 只有一個類型，哪怕一個名字既是地點又是機構。 | **多類型實體，按機率排序。** `Entity.types` 攜帶所有競爭類型及其機率，按相關度降序。 |
| 每次部署都要下載模型權重。 | **完全無需本地模型。** 套件內建 Qwen3 分詞器（覆蓋 100+ 語言）；所有推理經 EasyJev 閘道完成。 |

## 安裝

- 從 [PyPI](https://pypi.org/project/jevner/) 安裝

    ```shell
    pip install -U jevner
    ```

- 推薦：使用 [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- 從 [GitHub](https://github.com/vortezwohl/JevNER) 安裝——搶先體驗未發布功能。

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## 快速上手

```python
from jevner import JevNER

ner = JevNER()  # 憑證從 TYPESAFE_API_KEY / TYPESAFE_BASE_URL 環境變數讀取

# 回傳值就是巢狀森林：直接列印即可渲染成樹。
forest = ner("你好, 我的名字是陳嘉禾, 來自江蘇南京.")
for node in forest:
    print(node)
# 陳嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江蘇南京 (GPE 1.00) [16:20]
# ├── 江蘇 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

這就是完整的第一閉環：不下載模型、不微調、不寫後處理。PERSON 實體會升級為帶性別判決的 `Person`，偏移永遠滿足 `text == source[start:end]`，巢狀關係就裝在回傳值裡。

> [!TIP]
> 閘道路由了多個 System One 模型。預設是 `baseten/mammoth-1`；隨時用 `JevNER(model_name="jev-latest")` 切換。

> 想用 TypeScript、Go 或裸 `curl` 呼叫？線協定就是純 JSON REST——見 [`examples/`](../examples/README.md)。

## 核心能力

| 能力 | 你能得到什麼 |
| --- | --- |
| 零樣本識別 | 不訓練、不標註、不微調。定義好要抽什麼，直接開呼叫。 |
| 判決模型地基 | Noul / Choice 問題攜帶校準機率——是/否置信度與完整選項分布，不是自由文字生成。 |
| 巢狀實體森林 | `ner(text)` 回傳 `list[EntityNode]`。交叉 span 以兄弟身分掛到最近公共祖先下；巢狀由偏移推導，不做子字串猜測。 |
| 校準的多類型實體 | `Entity.types` 是按相關度排序的 `(EntityType, probability)` 元組序列——一個實體可以既是 GPE *又是* LOC。 |
| 24 種語言覆蓋 | 在亞洲、歐洲、美洲端到端驗證，含低資源與無空格文字。 |
| JSON 開放詞表 | `EntityTaxonomy.from_json` 接受三段式文件（`entity_types`、`entity_packs`、`default_packs`），可擴充或覆寫 31 個內建類型與 4 個實體包——為不寫程式的使用者而生。 |
| 垂直實體包 | 預設攜帶 `CORE`（18 個 OntoNotes 5.0 類型 + MISC）；`BIOMED`、`FINANCE`、`TECH` 按需疊加。 |
| 性別識別 | PERSON 實體升級為 `Person`，攜帶 `MALE` / `FEMALE` / `UNKNOWN` 與機率，程序內快取。 |
| 內建 Qwen3 分詞器 | 覆蓋 100+ 語言的位元組級 BPE 隨套件發布——不存取 HuggingFace、不需要 `transformers`、不需要 `torch`。 |
| 請求預算防護 | 視窗按閘道 65,535 位元組請求體上限裁切；分型批次按 512 選項上限收窄——每個請求在建構上就不會越界。 |
| 三層快取 | 檢測、分型、性別結果按視窗雜湊存入 LRU 快取；重複或交疊文字毫秒級回傳。 |
| 全程型別安全 | 完整標註並攜帶 `py.typed`，嚴格模式 mypy 零錯誤；提示詞經結構化 prompt4py 範本渲染。 |

## 常用模式

### 列印森林，而不是扁平列表

回傳值已經是樹。`str(node)` 用製表符線條渲染節點及其後代——類型、機率、性別、偏移一應俱全。

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### 啟用垂直實體包

實體包按名疊加；發給模型的類型表就是你選擇的集合的聯集。

```python
forest = ner(
    "患者確診為二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### 用 JSON 定義詞表

任何使用者——包括從沒打開過程式碼編輯器的人——都能用一份 JSON 文件擴充識別詞表。自訂類型與內建類型平起平坐。

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

`from_json` 接受 JSON 字串或已解析的 dict。未知類型引用、空描述、未註冊的預設包都會攜帶出錯欄位路徑拋出 `ValueError`。

### 按校準置信度路由

每個機率都來自以校準為訓練目標的判決模型，所以閾值是名副其實的：

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 真的是「十次對八次」
```

同一文字的重複呼叫毫秒級命中快取——對 RAG 重查詢和文件重跑很友好。

## 判決模型

JevNER 可以運行在閘道上註冊的任何 System One 判決模型上——這類模型用校準機率回答型別化問題（Noul 是/否、Choice 選擇），而不是產生文字。預設 `baseten/mammoth-1`；`jev-latest` 及另外 20+ 個模型只需一個參數：

| | |
| --- | --- |
| NER 已驗證 | `baseten/mammoth-1`（預設）、`jev-latest`、`jev-1.13.0`、`jev-1.14.0` |
| 最便宜 | `baseten/joey-1`、`respan/span-01-lite`、`perplexity/pplx-decider-flash-v1.2` —— 輸入低至 $0.01/百萬詞元 |
| 同樣可用 | Cloudflare Clef 家族、Nace.AI Drex、LiquidAI d1、Microsoft Decision-1、Upstage Solar Decide、Inception Mercury Decide、OpenAI GPT-6 Luna Decisions、`jaredpalmer/kev-4b` 等 |

含價格與逐模型注意事項的完整目錄：[支援的判決模型](../docs/models.md)。

## 生態與對比

JevNER 是判決模型（System One）浪潮的參考應用——這個生態裡的模型和引擎回傳型別化的校準判決而非散文：TypeSafe **Jev** 家族、[EasyJev](https://www.easyjev.online/) 聚合閘道、[laya](https://github.com/NandhaKishorM/laya)（非自回歸決策引擎，100+ 語言）與 [kev](https://github.com/jaredpalmer/kev)（基於 Qwen 的開源決策模型）這樣的開源引擎、以及 [AnyJev](https://github.com/nokia-applied-research/AnyJev) 這樣的轉接器。全景圖：[判決模型生態](../docs/ecosystem.md)。

正在選型 NER 技術棧？看 [JevNER 對比 GLiNER、spaCy 與託管 NER API](../docs/alternatives.md)——一份誠實的對比，包括什麼時候本地 [GLiNER](https://github.com/urchade/GLiNER) 檢查點或 [spaCy](https://github.com/explosion/spaCy) 流水線才是更優解。

## 公開 API

根套件刻意只暴露八個名字：

```python
from jevner import (
    Entity,          # 已識別實體: text, types, start, end
    EntityNode,      # 樹節點: entity + children
    EntityPack,      # 實體類型的具名捆綁 (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # 開放類型系統: 註冊表 + from_json + resolve
    EntityType,      # 標籤 + 描述的值物件
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # 識別器本身
    Person,          # Entity + gender + gender_probability
)
```

## 設計邊界

JevNER 只擁有識別流水線及其契約。它刻意不提供：

- 任何形式的本地模型推理；
- 模型託管、微調或權重分發；
- 訓練資料、標註工具或評測框架；
- UI 元件；以及
- 閘道 SDK 內部的再匯出。

這條邊界讓 SDK 保持為一條薄薄、可驗證的流水線，把託管與產品體驗留給平台。

## 贊助商

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**所有判決模型。一個 API。毫秒級判決。**

EasyJev 是 System One 判決模型的聚合之家——**Jev、Decider、Drex、d1、Clef 等，來自 14 家廠商、28 個在售模型**——統一收斂在一個 `typesafe-sdk` 相容端點之後。單次前向傳播毫秒級完成每個判決；所有模型輸出詞元一律 $0 計費，每次呼叫按上游用量逐詞元結算在整數配額帳本上（1,000,000,000 配額 = $1，無浮點漂移）。信箱驗證碼登入——首次登入自動建號，新使用者贈 $1 體驗額度，儲值為一次性 USDT/USDC 支付、無需訂閱。已有模型名（`jev-latest`、`jev-1.14.0`……）原樣可用：把 `base_url` 和 `api_key` 指向閘道，每次請求任選模型。

[立即註冊 &rarr;](https://www.easyjev.online/) &middot; [模型廣場](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [定價](https://www.easyjev.online/#/pricing) &middot; [服務條款](https://www.easyjev.online/#/tos) &middot; [隱私政策](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## 開發

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # 風格與靜態檢查
uv run mypy jevner                          # 型別檢查
uv run python tests/test_local.py           # 離線演算法測試 (12 項)
uv run python tests/test_e2e.py             # 端到端 (需要 TYPESAFE_API_KEY, 產生真實計費)
uv run python tests/test_languages.py       # 24 語言掃描 (產生真實計費)
uv run python tests/bench_quick.py          # 緊湊效能實驗 (產生真實計費)
```

## 參與貢獻

貢獻應改進可驗證的流水線行為：檢測召回、分型精確率、巢狀保真、詞表驗證、快取行為或請求預算安全。保持公開 API 精簡，並為新行為添加可獨立執行的測試。

## 授權條款

[MIT](../LICENSE)
