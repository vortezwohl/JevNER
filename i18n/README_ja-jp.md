<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**ゼロショット多言語固有表現抽出。System One 判断モデル搭載。**

*判断モデルを固有表現抽出器に変える型安全な Python SDK —— ネストされたエンティティツリー、カスタムエンティティパック、Qwen3 トークナイザ同梱。ローカルモデルの学習もデプロイも不要。*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="72" />
  </a>
  <br/>
  <strong><a href="https://www.easyjev.online/">EasyJev</a> スポンサード —— すべての判断モデルを、ひとつのエンドポイントへ。</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**学習不要。任意の言語。ツリーは戻り値そのもの。**

[English](/vortezwohl/JevNER#readme) | [简体中文](/vortezwohl/JevNER/blob/main/i18n/README_zh-hans.md) | [繁體中文](/vortezwohl/JevNER/blob/main/i18n/README_zh-hant.md) | 日本語 | [한국어](/vortezwohl/JevNER/blob/main/i18n/README_ko-kr.md) | [Français](/vortezwohl/JevNER/blob/main/i18n/README_fr-fr.md) | [Español](/vortezwohl/JevNER/blob/main/i18n/README_es-es.md) | [Deutsch](/vortezwohl/JevNER/blob/main/i18n/README_de-de.md) | [Português](/vortezwohl/JevNER/blob/main/i18n/README_pt-br.md)

</div>

---

## JevNER を選ぶ理由

多くの NER スタックは二者択一を強います。ファインチューニング時にラベルセットが凍結されるローカルモデルか、本当の信頼度を隠すホステッド API か。

JevNER は第三の道を取ります。テキスト生成器ではなく **System One 判断モデル** がすべての質問を読み、散文の代わりに**較正された確率**を返す。SDK はその判断を組み立てて、ツリーとしてそのまま印字できる**ネストされたエンティティフォレスト**にします。

| 従来の NER の弱点 | JevNER のやり方 |
| --- | --- |
| ラベルセットはファインチューニング時に凍結。新しいドメインごとに再学習が必要。 | **設計からゼロショット。** エンティティ型はオープンデータ。プログラマでないユーザーも、実行時にプレーンな JSON 文書で型とパックを追加できる。 |
| 多言語対応は言語ごとにファインチューニングされたチェックポイントを意味する。 | **単一パイプラインで 24 言語を検証済み** —— 中国語・タイ語・アラビア語・ヘブライ語・ヒンディー語・トルコ語・インドネシア語・フィリピン語・ベトナム語・日本語・韓国語・英語・フランス語・ドイツ語・スペイン語・ポルトガル語・イタリア語・ロシア語・オランダ語・ポーランド語など。低リソース言語と分かち書きなし文字を含む。 |
| ホステッド NER API はフラットな span を返し、ネストは潰されるか捨てられる。 | **戻り値がツリー。** `node.children` が `node` に含まれるすべてのエンティティを保持し、原典オフセットから導出（`text == source[start:end]`）。ネストは構造的なもので、推測ではない。 |
| softmax スコアはルーティングに使える確率ではない。 | **較正された確率** が判断モデルから返る。`0.8` は「10 回中 8 回正しい」を意味し、信頼度ゲートの人間レビュー流程が本当に必要としているもの。 |
| 名前が地名でも組織でもある場合でも、1 span に 1 型。 | **マルチ型エンティティ、ランキング付き。** `Entity.types` は関連度順の `(EntityType, probability)` タプル列。 |
| デプロイのたびにモデル重みをダウンロード。 | **ローカルモデル完全不要。** 100+ 言語をカバーする Qwen3 トークナイザをパッケージに同梱。推理はすべて EasyJev ゲートウェイ経由。 |

## インストール

- [PyPI](https://pypi.org/project/jevner/) から

    ```shell
    pip install -U jevner
    ```

- 推奨: [uv](https://docs.astral.sh/uv/) を使う

    ```shell
    uv add -U jevner
    ```

- [GitHub](https://github.com/vortezwohl/JevNER) から —— 未リリース機能をいち早く。

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## クイックスタート

```python
from jevner import JevNER

ner = JevNER()  # 認証情報は TYPESAFE_API_KEY / TYPESAFE_BASE_URL から読み込まれる

# 戻り値そのものがネストされたフォレスト。print するだけでツリーになる。
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

これが最初の完全ループ。モデルのダウンロードもファインチューニングも後処理も不要。PERSON エンティティは性別判断付きの `Person` にアップグレードされ、オフセットは常に `text == source[start:end]` を満たし、ネストは戻り値そのものに乗っている。

> [!TIP]
> ゲートウェイは複数の System One モデルをルーティングする。デフォルトは `baseten/mammoth-1`。`JevNER(model_name="jev-latest")` でいつでも切替可能。

> TypeScript・Go・素の `curl` から呼びたい？ ワイヤープロトコルはプレーンな JSON REST —— [`examples/`](../examples/README.md) を参照。

## コア機能

| 機能 | 得られるもの |
| --- | --- |
| ゼロショット認識 | 学習・アノテーション・ファインチューニング不要。抽出対象を定義して即呼び出し。 |
| 判断モデル基盤 | 較正確率付きの Noul / Choice 質問 —— 自由生成ではなく yes/no 信頼度と完全な選択肢分布。 |
| ネストされたエンティティフォレスト | `ner(text)` は `list[EntityNode]` を返す。交差 span は最近共通祖先の兄弟として接続。ネストはオフセットから導出。 |
| 較正されたマルチ型エンティティ | `Entity.types` は関連度順の `(EntityType, probability)` タプル列 —— 1 エンティティが GPE *かつ* LOC になれる。 |
| 24 言語カバー | アジア・ヨーロッパ・南北アメリカでエンドツーエンド検証済み。低リソース・分かち書きなし文字を含む。 |
| JSON からのオープンタクソノミー | `EntityTaxonomy.from_json` は 3 セクション文書（`entity_types`・`entity_packs`・`default_packs`）を受け付け、31 の組込型と 4 パックを拡張・上書きする —— 非プログラマ向け。 |
| 垂直エンティティパック | `CORE`（OntoNotes 5.0 の 18 型 + MISC）をデフォルトで同梱。`BIOMED`・`FINANCE`・`TECH` をオンデマンドで重ねられる。 |
| 性別認識 | PERSON エンティティは `MALE` / `FEMALE` / `UNKNOWN` と確率付きの `Person` にアップグレード。プロセス内キャッシュ。 |
| Qwen3 トークナイザ同梱 | 100+ 言語のバイトレベル BPE をパッケージ内に —— HuggingFace アクセスも `transformers` も `torch` も不要。 |
| リクエスト予算安全 | ウィンドウはゲートウェイの 65,535 バイト本体上限から切り出し、タイピングバッチは 512 選択肢上限で絞り込み —— すべてのリクエストが構造的に上限内。 |
| 3 層キャッシュ | 検出・タイピング・性別の結果をウィンドウハッシュで LRU キャッシュ。重複・重なりテキストはミリ秒で返る。 |
| 全体で型安全 | 完全アノテーション + `py.typed`、厳格モードの mypy クリーン。プロンプトは構造化 prompt4py テンプレートで描画。 |

## よくあるパターン

### フラットなリストではなくフォレストを印字する

戻り値はすでにツリー。`str(node)` はノードと子孫を罫線付きで描画する —— 型・確率・性別・オフセット付き。

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### 垂直パックを有効化する

パックは名前で重なる。モデルに送られる型テーブルは、選択した集合の和集合そのもの。

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### JSON でタクソノミーを定義する

コードエディタを開けたことのないユーザーでも、JSON 文書で認識語彙を拡張できる。カスタム型は組込型と対等な市民権を持つ。

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

`from_json` は JSON 文字列でも解析済み dict でも受け付ける。未知の型参照・空の説明・未登録のデフォルトパックは、問題のフィールドパスを伴う `ValueError` を投げる。

### 較正された信頼度でルーティングする

すべての確率は較正を目的に学習された判断モデル由来。なので閾値は看板に偽りない：

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 は本当に「10 回中 8 回正しい」
```

同じテキストへの再呼び出しはミリ秒でキャッシュにヒット —— RAG の再クエリや文書の再実行に便利。

## 判断モデル

JevNER はゲートウェイに登録された任意の System One 判断モデルで動く —— テキストを生成する代わりに、型付き質問（Noul yes/no、Choice 選択）に較正確率で答えるモデル群。デフォルトは `baseten/mammoth-1`。`jev-latest` ほか 20+ モデルが引数 1 つで切替可能：

| | |
| --- | --- |
| NER 検証済み | `baseten/mammoth-1`（デフォルト）、`jev-latest`、`jev-1.13.0`、`jev-1.14.0` |
| 最安 | `baseten/joey-1`、`respan/span-01-lite`、`perplexity/pplx-decider-flash-v1.2` —— 入力 $0.01/100 万トークンから |
| その他利用可 | Cloudflare Clef ファミリー、Nace.AI Drex、LiquidAI d1、Microsoft Decision-1、Upstage Solar Decide、Inception Mercury Decide、OpenAI GPT-6 Luna Decisions、`jaredpalmer/kev-4b` など |

価格とモデルごとの注意事項を含む完全カタログ：[サポートされる判断モデル](../docs/models.md)。

## エコシステムと比較

JevNER は判断モデル（System One）ムーブメントのリファレンスアプリケーション —— 散文の代わりに型付きの較正判断を返すモデルとエンジンのエコシステム：TypeSafe **Jev** ファミリー、[EasyJev](https://www.easyjev.online/) の集約ゲートウェイ、[laya](https://github.com/NandhaKishorM/laya)（非自己回帰決定エンジン、100+ 言語）や [kev](https://github.com/jaredpalmer/kev)（Qwen ベースのオープン決定モデル）といったオープンソースエンジン、そして [AnyJev](https://github.com/nokia-applied-research/AnyJev) のようなアダプター。地図：[判断モデルエコシステム](../docs/ecosystem.md)。

NER スタックの選定中？ [JevNER vs GLiNER vs spaCy vs ホステッド NER API](../docs/alternatives.md) —— ローカル [GLiNER](https://github.com/urchade/GLiNER) チェックポイントや [spaCy](https://github.com/explosion/spaCy) パイプラインが正解になる場合も含めた、正直な比較。

## パブリック API

ルートパッケージは意図的に 8 つの名前のみを公開する：

```python
from jevner import (
    Entity,          # 認識済みエンティティ: text, types, start, end
    EntityNode,      # ツリーノード: entity + children
    EntityPack,      # エンティティ型の名前付きバンドル (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # オープン型システム: レジストリ + from_json + resolve
    EntityType,      # ラベル + 説明の値オブジェクト
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # 認識器
    Person,          # Entity + gender + gender_probability
)
```

## 設計境界

JevNER が所有するのは認識パイプラインとその契約だけ。意図的に提供しないもの：

- いかなる形のローカルモデル推理；
- モデルホスティング・ファインチューニング・重み配布；
- 学習データ・アノテーションツール・評価ハーネス；
- UI コンポーネント；そして
- ゲートウェイ SDK 内部の再エクスポート。

この境界により、SDK は薄く検証可能なパイプラインであり続け、ホスティングと製品体験はプラットフォームに委ねられる。

## スポンサー

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**すべての判断モデル。ひとつの API。ミリ秒の判断。**

EasyJev は System One 判断モデルの集約ホーム —— **Jev、Decider、Drex、d1、Clef など、14 ベンダー・28 モデル** が、ひとつの `typesafe-sdk` 互換エンドポイントの背後に。単一フォワードパスで各判断をミリ秒で完了。全モデルの出力トークンは $0 課金で、呼び出しごとに上流使用量に対してトークン単位で整数クォータ台帳に決済される（1,000,000,000 クォータ = $1、浮動小数点ドリフトなし）。メール認証コードでログイン —— 初回ログインで自動アカウント作成、新規ユーザーに $1 の体験クレジット、チャージは一回払いの USDT/USDC でサブスク不要。既存のモデル名（`jev-latest`、`jev-1.14.0`…）はそのまま動く。`base_url` と `api_key` をゲートウェイに向ければ、リクエストごとに任意のモデルを選べる。

[登録する &rarr;](https://www.easyjev.online/) &middot; [モデル広場](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [料金](https://www.easyjev.online/#/pricing) &middot; [利用規約](https://www.easyjev.online/#/tos) &middot; [プライバシー](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## 開発

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # スタイルと静的チェック
uv run mypy jevner                          # 型チェック
uv run python tests/test_local.py           # オフラインアルゴリズムテスト (12 項目)
uv run python tests/test_e2e.py             # エンドツーエンド (TYPESAFE_API_KEY 必須、API 課金あり)
uv run python tests/test_languages.py       # 24 言語スイープ (API 課金あり)
uv run python tests/bench_quick.py          # 簡易性能実験 (API 課金あり)
```

## コントリビューション

コントリビューションは検証可能なパイプライン挙動の改善を対象とする：検出リコール、タイピング精度、ネスト忠実度、タクソノミ検証、キャッシュ挙動、リクエスト予算安全性。公開 API は狭く保ち、新しい挙動には独立に実行できるテストを追加すること。

## ライセンス

[MIT](../LICENSE)
