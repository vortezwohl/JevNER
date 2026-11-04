<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**제로샷 다국어 개체명 인식. System One 판단 모델 기반.**

*판단 모델을 개체명 인식기로 바꾸는 타입 세이프 Python SDK — 중첩 개체 트리, 커스텀 개체 팩, 내장 Qwen3 토크나이저. 학습하거나 배포할 로컬 모델이 전혀 없습니다.*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="72" />
  </a>
  <br/>
  <strong><a href="https://www.easyjev.online/">EasyJev</a> 스폰서 — 모든 판단 모델을 하나의 엔드포인트로.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**학습 없음. 모든 언어. 트리가 반환값에 담겨 옵니다.**

[English](../README.md) | [简体中文](README_zh-hans.md) | [繁體中文](README_zh-hant.md) | [日本語](README_ja-jp.md) | 한국어 | [Français](README_fr-fr.md) | [Español](README_es-es.md) | [Deutsch](README_de-de.md) | [Português](README_pt-br.md)

</div>

---

## JevNER를 선택해야 하는 이유

대부분의 NER 스택은 양자택일을 강요합니다. 파인튜닝 시점에 라벨 집합이 얼어붙는 로컬 모델이거나, 진짜 신뢰도를 감추는 호스티드 API이거나.

JevNER는 세 번째 길을 갑니다. 텍스트 생성기가 아닌 **System One 판단 모델**이 모든 질문을 읽고 산문 대신 **보정된 확률**을 반환하며, SDK가 그 판단들을 조립해 트리로 바로 인쇄할 수 있는 **중첩 개체 포리스트**로 만듭니다.

| 기존 NER의 약점 | JevNER의 접근 |
| --- | --- |
| 라벨 집합은 파인튜닝 시점에 고정. 새 도메인마다 재학습 필요. | **설계 자체가 제로샷.** 개체 타입은 오픈 데이터입니다. 프로그래머가 아닌 사용자도 런타임에 평범한 JSON 문서로 타입과 팩을 추가할 수 있습니다. |
| 다국어 지원은 언어별 파인튜닝 체크포인트를 의미. | **단일 파이프라인으로 24개 언어 검증 완료** — 중국어·태국어·아랍어·히브리어·힌디어·터키어·인도네시아어·필리핀어·베트남어·일본어·한국어·영어·프랑스어·독일어·스페인어·포르투갈어·이탈리아어·러시아어·네덜란드어·폴란드어 등. 저자원 언어와 띄어쓰기 없는 문자 포함. |
| 호스티드 NER API는 평평한 span만 반환. 중첩은 눌리거나 버려짐. | **반환값이 곧 트리.** `node.children`이 `node`에 포함된 모든 개체를 갖고, 원문 오프셋에서 도출됩니다(`text == source[start:end]`). 중첩은 구조적이지 추측이 아닙니다. |
| softmax 점수는 라우팅에 쓸 수 있는 확률이 아님. | **보정된 확률**이 판단 모델에서 옵니다. `0.8`은 "열 번에 여덟 번 맞는다"는 뜻이고, 이것이 신뢰도 게이트 인간 검수 흐름이 정말로 필요로 하는 것입니다. |
| 이름이 장소이면서 조직이어도 span당 타입 하나. | **다중 타입 개체, 순위 포함.** `Entity.types`가 관련도 내림차순의 `(EntityType, probability)` 튜플 시퀀스를 담습니다. |
| 배포할 때마다 모델 가중치 다운로드. | **로컬 모델 전혀 없음.** 100+ 언어를 커버하는 Qwen3 토크나이저가 패키지에 내장. 모든 추론은 EasyJev 게이트웨이를 통해. |

## 설치

- [PyPI](https://pypi.org/project/jevner/)에서

    ```shell
    pip install -U jevner
    ```

- 권장: [uv](https://docs.astral.sh/uv/) 사용

    ```shell
    uv add -U jevner
    ```

- [GitHub](https://github.com/vortezwohl/JevNER)에서 — 미출시 기능을 먼저 만나보려면.

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## 빠른 시작

```python
from jevner import JevNER

ner = JevNER()  # 자격 증명은 TYPESAFE_API_KEY / TYPESAFE_BASE_URL에서 읽힘

# 반환값 자체가 중첩 포리스트입니다. print하면 트리로 렌더링됩니다.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

이것이 완전한 첫 루프입니다. 모델 다운로드도, 파인튜닝도, 후처리도 없습니다. PERSON 개체는 성별 판단이 붙은 `Person`으로 업그레이드되고, 오프셋은 항상 `text == source[start:end]`를 만족하며, 중첩은 반환값 그 자체에 실려 옵니다.

> [!TIP]
> 게이트웨이는 여러 System One 모델을 라우팅합니다. 기본값은 `baseten/mammoth-1`이며 `JevNER(model_name="jev-latest")`로 언제든 전환할 수 있습니다.

> TypeScript·Go·순수 `curl`로 호출하고 싶나요? 와이어 프로토콜은 평범한 JSON REST입니다 — [`examples/`](../examples/README.md)를 보세요.

## 핵심 기능

| 기능 | 무엇을 얻는가 |
| --- | --- |
| 제로샷 인식 | 학습 없음, 주석 없음, 파인튜닝 없음. 추출할 대상을 정의하고 바로 호출. |
| 판단 모델 기반 | 보정 확률을 동반한 Noul / Choice 질문 — 자유 생성이 아니라 yes/no 신뢰도와 완전한 선택지 분포. |
| 중첩 개체 포리스트 | `ner(text)`가 `list[EntityNode]`를 반환. 교차 span은 최근 공통 조상의 형제로 연결. 중첩은 오프셋에서 도출. |
| 보정된 다중 타입 개체 | `Entity.types`는 관련도순 `(EntityType, probability)` 튜플 시퀀스 — 한 개체가 GPE *이면서* LOC일 수 있습니다. |
| 24개 언어 커버리지 | 아시아·유럽·아메리카에서 엔드투엔드 검증. 저자원 및 띄어쓰기 없는 문자 포함. |
| JSON 오픈 택소노미 | `EntityTaxonomy.from_json`이 3섹션 문서(`entity_types`·`entity_packs`·`default_packs`)를 받아 31개 내장 타입과 4개 팩을 확장하거나 덮어씁니다 — 비프로그래머를 위해 설계. |
| 수직 개체 팩 | `CORE`(OntoNotes 5.0의 18개 타입 + MISC)가 기본 동봉. `BIOMED`·`FINANCE`·`TECH`는 필요 시 겹쳐 켤 수 있습니다. |
| 성별 인식 | PERSON 개체가 `MALE` / `FEMALE` / `UNKNOWN`과 확률을 동반한 `Person`으로 업그레이드. 프로세스 내 캐시. |
| Qwen3 토크나이저 내장 | 100+ 언어의 바이트 수준 BPE가 패키지 안에 — HuggingFace 접근도 `transformers`도 `torch`도 불필요. |
| 요청 예산 안전장치 | 윈도우는 게이트웨이의 65,535바이트 본문 상한에서 산출되고 타이핑 배치는 512선택지 상한으로 좁혀집니다 — 모든 요청이 구조적으로 한계 이내. |
| 3계층 캐시 | 검출·타이핑·성별 결과가 윈도우 해시로 LRU 캐시에. 반복·겹친 텍스트는 밀리초에 반환. |
| 전체 타입 세이프 | 완전한 어노테이션과 `py.typed`, 엄격 모드 mypy 클린. 프롬프트는 구조화된 prompt4py 템플릿으로 렌더링. |

## 자주 쓰는 패턴

### 평평한 리스트가 아니라 포리스트를 인쇄하기

반환값은 이미 트리입니다. `str(node)`는 노드와 자손을 상자 긋기 선으로 렌더링합니다 — 타입·확률·성별·오프셋 포함.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### 수직 팩 활성화

팩은 이름으로 쌓입니다. 모델에 보내지는 타입 테이블은 당신이 고른 집합의 합집합 그 자체입니다.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### JSON으로 택소노미 정의하기

코드 에디터를 열어본 적 없는 사용자도 JSON 문서 하나로 인식 어휘를 확장할 수 있습니다. 커스텀 타입은 내장 타입과 대등한 시민입니다.

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

`from_json`은 JSON 문자열이나 파싱된 dict를 받습니다. 알 수 없는 타입 참조, 빈 설명, 등록되지 않은 기본 팩은 문제의 필드 경로와 함께 `ValueError`를 던집니다.

### 보정된 신뢰도로 라우팅하기

모든 확률은 보정을 목표로 학습된 판단 모델에서 오므로, 임계값은 겉과 속이 같습니다:

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8은 정말로 "열 번에 여덟 번 맞음"
```

같은 텍스트의 반복 호출은 밀리초에 캐시에 hit — RAG 재쿼리와 문서 재실행에 유용합니다.

## 판단 모델

JevNER는 게이트웨이에 등록된 모든 System One 판단 모델에서 동작합니다 — 텍스트를 생성하는 대신 타입화된 질문(Noul yes/no, Choice 선택)에 보정 확률로 답하는 모델들. 기본값은 `baseten/mammoth-1`이고 `jev-latest` 외 20여 개 모델이 인자 하나로 전환됩니다:

| | |
| --- | --- |
| NER 검증 완료 | `baseten/mammoth-1`(기본값), `jev-latest`, `jev-1.13.0`, `jev-1.14.0` |
| 최저가 | `baseten/joey-1`, `respan/span-01-lite`, `perplexity/pplx-decider-flash-v1.2` — 입력 $0.01/100만 토큰부터 |
| 그 외 사용 가능 | Cloudflare Clef 패밀리, Nace.AI Drex, LiquidAI d1, Microsoft Decision-1, Upstage Solar Decide, Inception Mercury Decide, OpenAI GPT-6 Luna Decisions, `jaredpalmer/kev-4b` 등 |

가격과 모델별 주의사항을 담은 전체 카탈로그: [지원되는 판단 모델](../docs/models.md).

## 생태계와 비교

JevNER는 판단 모델(System One) 무브먼트의 레퍼런스 애플리케이션입니다 — 산문 대신 타입화된 보정 판단을 반환하는 모델과 엔진의 생태계: TypeSafe **Jev** 패밀리, [EasyJev](https://www.easyjev.online/) 통합 게이트웨이, [laya](https://github.com/NandhaKishorM/laya)(비자기회귀 결정 엔진, 100+ 언어)와 [kev](https://github.com/jaredpalmer/kev)(Qwen 기반 오픈소스 결정 모델) 같은 오픈소스 엔진, 그리고 [AnyJev](https://github.com/nokia-applied-research/AnyJev) 같은 어댑터. 전체 지도: [판단 모델 생태계](../docs/ecosystem.md).

NER 스택을 고르는 중이라면? [JevNER vs GLiNER vs spaCy vs 호스티드 NER API](../docs/alternatives.md) — 로컬 [GLiNER](https://github.com/urchade/GLiNER) 체크포인트나 [spaCy](https://github.com/explosion/spaCy) 파이프라인이 더 나은 선택인 순간까지 포함한 정직한 비교.

## 퍼블릭 API

루트 패키지는 의도적으로 여덟 이름만 노출합니다:

```python
from jevner import (
    Entity,          # 인식된 개체: text, types, start, end
    EntityNode,      # 트리 노드: entity + children
    EntityPack,      # 개체 타입의 이름 붙은 묶음 (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # 오픈 타입 시스템: 레지스트리 + from_json + resolve
    EntityType,      # 라벨 + 설명 값 객체
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # 인식기
    Person,          # Entity + gender + gender_probability
)
```

## 설계 경계

JevNER가 소유하는 것은 인식 파이프라인과 그 계약뿐입니다. 의도적으로 제공하지 않는 것:

- 어떤 형태의 로컬 모델 추론;
- 모델 호스팅·파인튜닝·가중치 배포;
- 학습 데이터·주석 도구·평가 하니스;
- UI 컴포넌트; 그리고
- 게이트웨이 SDK 내부의 재익스포트.

이 경계 덕분에 SDK는 얇고 검증 가능한 파이프라인으로 남고, 호스팅과 제품 경험은 플랫폼에 맡겨집니다.

## 스폰서

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="EasyJev logo" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**모든 System One 모델, 하나의 엔드포인트 — [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) 와 완벽 호환.**

EasyJev는 각 벤더의 System One 판단 모델 — Jev, Decider, Drex, d1, Clef, kev 등 — 을 하나의 엔드포인트 뒤에 모읍니다. 공식 [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) 는 수정 없이 그대로 연결됩니다: `base_url`과 `api_key`를 게이트웨이로 향하게 하면 요청마다 원하는 모델을 고를 수 있고, 기존 모델 이름(`jev-latest`, `jev-1.14.0`…)도 그대로 동작합니다. 판단은 밀리초에 돌아오고 모든 모델의 출력 토큰은 $0로 과금됩니다. 이메일 인증 코드만으로 로그인할 수 있으며 첫 로그인 시 자동 계정 생성과 함께 $1 체험 크레딧이 지급됩니다. 충전은 카드(Visa / Mastercard)·Apple Pay·Google Pay·WeChat Pay를 지원하며, 일회성 결제에 구독은 없습니다.

[가입하기 &rarr;](https://www.easyjev.online/) &middot; [모델 광장](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [요금제](https://www.easyjev.online/#/pricing) &middot; [약관](https://www.easyjev.online/#/tos) &middot; [개인정보](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## 개발

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # 스타일 및 정적 검사
uv run mypy jevner                          # 타입 검사
uv run python tests/test_local.py           # 오프라인 알고리즘 테스트 (12항목)
uv run python tests/test_e2e.py             # 엔드투엔드 (TYPESAFE_API_KEY 필요, API 과금 발생)
uv run python tests/test_languages.py       # 24개 언어 스윕 (API 과금 발생)
uv run python tests/bench_quick.py          # 간소화 성능 실험 (API 과금 발생)
```

## 기여

기여는 검증 가능한 파이프라인 동작의 개선을 대상으로 해야 합니다: 검출 재현율, 타이핑 정밀도, 중첩 충실도, 택소노미 검증, 캐시 동작, 요청 예산 안전성. 퍼블릭 API는 좁게 유지하고 새 동작에는 독립적으로 실행 가능한 테스트를 추가하세요.

## 라이선스

[MIT](../LICENSE)
