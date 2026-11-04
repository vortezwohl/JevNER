<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**Reconhecimento de entidades nomeadas multilíngue zero-shot. Alimentado por modelos de julgamento System One.**

*Um SDK Python type-safe que transforma um modelo de julgamento em um reconhecedor de entidades nomeadas — árvores de entidades aninhadas, pacotes de entidades personalizáveis e um tokenizador Qwen3 embutido, sem nenhum modelo local para treinar ou implantar.*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="Logo da EasyJev" width="72" />
  </a>
  <br/>
  <strong>Patrocinado pela <a href="https://www.easyjev.online/">EasyJev</a> — todos os modelos de decisão, um único endpoint.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**Sem treinamento. Qualquer idioma. A árvore vem no valor de retorno.**

[English](../README.md) | [简体中文](README_zh-hans.md) | [繁體中文](README_zh-hant.md) | [日本語](README_ja-jp.md) | [한국어](README_ko-kr.md) | [Français](README_fr-fr.md) | [Español](README_es-es.md) | [Deutsch](README_de-de.md) | Português

</div>

---

## Por que JevNER

A maioria das stacks de NER obriga você a escolher: um modelo local congelado num conjunto de rótulos, ou uma API hospedada que esconde quão confiante realmente é.

JevNER segue um terceiro caminho. Um **modelo de julgamento System One** — não um gerador de texto — lê cada pergunta e devolve **probabilidades calibradas** em vez de prosa, e o SDK monta esses julgamentos numa **floresta de entidades aninhadas** que você imprime como árvore.

| Onde o NER clássico falha | O que o JevNER faz |
| --- | --- |
| Conjuntos de rótulos congelam no fine-tuning; cada novo domínio exige um novo treino. | **Zero-shot por design.** Tipos de entidade são dados abertos: qualquer usuário (inclusive não programadores) adiciona tipos e pacotes em tempo de execução por um simples documento JSON. |
| Cobertura multilíngue significa um checkpoint afinado por idioma. | **Um pipeline só, 24 idiomas verificados** — chinês, tailandês, árabe, hebraico, híndi, turco, indonésio, filipino, vietnamita, japonês, coreano, inglês, francês, alemão, espanhol, português, italiano, russo, neerlandês, polonês e mais. Inclui idiomas de baixo recurso e escritas sem espaços. |
| APIs de NER hospedadas devolvem spans planos; o aninhamento é achatado ou descartado. | **O valor de retorno É a árvore.** `node.children` guarda cada entidade contida em `node`, derivada dos offsets do texto original (`text == source[start:end]`) — o aninhamento é estrutural, não adivinhado. |
| Um score softmax não é uma probabilidade na qual dar rota. | **Probabilidades calibradas** do modelo de julgamento: `0.8` se comporta como “certo 80% das vezes”, exatamente o que fluxos de revisão humana com limiar de confiança precisam. |
| Um span ganha um tipo só, mesmo quando um nome é ao mesmo tempo lugar e organização. | **Entidades multitypo, ranqueadas.** `Entity.types` carrega cada tipo concorrente com sua probabilidade, ordenado por relevância. |
| Todo deploy baixa pesos de modelo. | **Zero modelos locais.** Um tokenizador Qwen3 (100+ idiomas) vem dentro do pacote; toda inferência passa pelo gateway EasyJev. |

## Instalação

- Do [PyPI](https://pypi.org/project/jevner/)

    ```shell
    pip install -U jevner
    ```

- Recomendado, com [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- Do [GitHub](https://github.com/vortezwohl/JevNER) — acesso a funcionalidades ainda não lançadas.

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## Início rápido

```python
from jevner import JevNER

ner = JevNER()  # credenciais lidas de TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# O valor de retorno É a floresta aninhada: imprimir renderiza uma árvore.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

Esse é o primeiro ciclo completo: sem download de modelo, sem fine-tuning, sem pós-processamento. Entidades PERSON são promovidas a `Person` com um julgamento de gênero, offsets sempre satisfazem `text == source[start:end]`, e o aninhamento viaja no próprio valor de retorno.

> [!TIP]
> O gateway roteia vários modelos System One. O padrão é `baseten/mammoth-1`; troque quando quiser com `JevNER(model_name="jev-latest")`.

> Chamando de TypeScript, Go ou `curl` puro? O protocolo é JSON REST puro — veja [`examples/`](../examples/README.md).

## Capacidades centrais

| Capacidade | O que você ganha |
| --- | --- |
| Reconhecimento zero-shot | Sem treino, sem anotação, sem fine-tuning. Defina o que extrair e comece a chamar. |
| Fundação em modelo de julgamento | Perguntas Noul / Choice com probabilidades calibradas — confiança sim/não e distribuições completas de opções, não geração livre. |
| Floresta de entidades aninhadas | `ner(text)` devolve `list[EntityNode]`. Spans cruzados anexam-se como irmãos ao ancestral comum mais próximo; o aninhamento deriva dos offsets, sem palpite de substring. |
| Entidades multitypo calibradas | `Entity.types` é uma sequência de tuplas `(EntityType, probability)` ordenada por relevância — uma entidade pode ser GPE *e* LOC. |
| Cobertura de 24 idiomas | Verificada de ponta a ponta na Ásia, Europa e Américas, incluindo escritas de baixo recurso e sem espaços. |
| Taxonomia aberta via JSON | `EntityTaxonomy.from_json` aceita um documento em três seções (`entity_types`, `entity_packs`, `default_packs`) que estende ou sobrescreve os 31 tipos e 4 pacotes embutidos — feito para usuários não programadores. |
| Pacotes verticais de entidades | `CORE` (18 tipos do OntoNotes 5.0 + MISC) vem por padrão; `BIOMED`, `FINANCE` e `TECH` empilham sob demanda. |
| Reconhecimento de gênero | Entidades PERSON são promovidas a `Person` com `MALE` / `FEMALE` / `UNKNOWN` e uma probabilidade, em cache no processo. |
| Tokenizador Qwen3 embutido | BPE em nível de byte cobrindo 100+ idiomas dentro do pacote — sem acesso ao HuggingFace, sem `transformers`, sem `torch`. |
| Segurança de orçamento de requisição | Janelas dimensionadas pelo limite de corpo de 65.535 bytes do gateway; lotes de digitação estreitados pelo teto de 512 opções — cada requisição fica nos limites por construção. |
| Cache em três camadas | Resultados de detecção, digitação e gênero em caches LRU indexados por hash de janela; texto repetido ou sobreposto volta em milissegundos. |
| Type-safe de ponta a ponta | Anotações completas com `py.typed`, mypy limpo em modo estrito, prompts renderizados pelos templates estruturados do prompt4py. |

## Padrões comuns

### Imprima a floresta, não uma lista plana

O valor de retorno já é a árvore. `str(node)` renderiza um nó e seus descendentes com linhas de desenho, tipos, probabilidades, gênero e offsets.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### Ative um pacote vertical

Pacotes empilham por nome; a tabela de tipos enviada ao modelo contém exatamente a união da sua seleção.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### Defina uma taxonomia via JSON

Qualquer usuário — inclusive quem nunca abre um editor de código — expande o vocabulário de reconhecimento com um documento JSON. Tipos personalizados são cidadãos de primeira classe ao lado dos embutidos.

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

`from_json` aceita uma string JSON ou um dict já interpretado. Referências de tipos desconhecidas, descrições vazias e pacotes padrão não registrados levantam `ValueError` com o caminho do campo infrator.

### Roteie por confiança calibrada

Toda probabilidade vem de um modelo de julgamento treinado para calibração, então limiares significam o que dizem:

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 significa mesmo "certo 80% das vezes"
```

Chamadas repetidas sobre o mesmo texto acertam o cache em milissegundos — útil para re-consultas RAG e reprocessamentos de documentos.

## Modelos de decisão

O JevNER roda em qualquer modelo de decisão System One registrado no gateway — modelos que respondem perguntas tipadas (Noul sim/não, Choice seleção) com probabilidades calibradas em vez de gerar texto. Padrão `baseten/mammoth-1`; `jev-latest` e mais de 20 outros trocam com um argumento:

| | |
| --- | --- |
| Verificados para NER | `baseten/mammoth-1` (padrão), `jev-latest`, `jev-1.13.0`, `jev-1.14.0` |
| Mais baratos | `baseten/joey-1`, `respan/span-01-lite`, `perplexity/pplx-decider-flash-v1.2` — a partir de $0.01/M de tokens de entrada |
| Também disponíveis | família Cloudflare Clef, Nace.AI Drex, LiquidAI d1, Microsoft Decision-1, Upstage Solar Decide, Inception Mercury Decide, OpenAI GPT-6 Luna Decisions, `jaredpalmer/kev-4b`, e mais |

Catálogo completo com preços e ressalvas por modelo: [Modelos de decisão suportados](../docs/models.md).

## Ecossistema e comparações

O JevNER é uma aplicação de referência do movimento dos modelos de decisão (System One) — o ecossistema de modelos e engines que devolvem julgamentos tipados e calibrados em vez de prosa: a família **Jev** da TypeSafe, o gateway agregado [EasyJev](https://www.easyjev.online/), engines abertas como [laya](https://github.com/NandhaKishorM/laya) (engine de decisão não autorregressiva, 100+ idiomas) e [kev](https://github.com/jaredpalmer/kev) (modelos de decisão abertos sobre Qwen), e adaptadores como [AnyJev](https://github.com/nokia-applied-research/AnyJev). O mapa: [o ecossistema de modelos de decisão](../docs/ecosystem.md).

Escolhendo uma stack de NER? [JevNER vs GLiNER vs spaCy vs APIs de NER hospedadas](../docs/alternatives.md) — uma comparação honesta, incluindo quando um checkpoint [GLiNER](https://github.com/urchade/GLiNER) local ou um pipeline [spaCy](https://github.com/explosion/spaCy) é a melhor escolha.

## API pública

O pacote raiz expõe deliberadamente oito nomes:

```python
from jevner import (
    Entity,          # uma entidade reconhecida: text, types, start, end
    EntityNode,      # nó da árvore: entity + children
    EntityPack,      # pacote nomeado de tipos de entidade (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # sistema de tipos aberto: registro + from_json + resolve
    EntityType,      # objeto valor rótulo + descrição
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # o reconhecedor
    Person,          # Entity + gender + gender_probability
)
```

## Limites de design

O JevNER é dono do pipeline de reconhecimento e de seus contratos. Ele deliberadamente não fornece:

- inferência com modelo local de qualquer espécie;
- hospedagem de modelos, fine-tuning ou distribuição de pesos;
- dados de treino, ferramentas de anotação ou arneses de avaliação;
- componentes de UI; nem
- re-exports de internos do SDK do gateway.

Essa fronteira mantém o SDK um pipeline fino e verificável, deixando hospedagem e experiência de produto para a plataforma.

## Patrocinador

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="Logo da EasyJev" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**Todos os modelos de decisão. Uma única API. Julgamentos em milissegundos.**

A EasyJev é a casa agregada dos modelos de decisão System One — **Jev, Decider, Drex, d1, Clef e mais,
de 14 fornecedores, 28 modelos à venda hoje** — por trás de um único endpoint compatível com `typesafe-sdk`. Uma única passagem direta completa cada julgamento em milissegundos; tokens de saída são cobrados a $0 em todos os modelos, e cada chamada é liquidada token a token contra o uso upstream num razão de cotas inteiras (1.000.000.000 de cotas = $1, sem desvio de ponto flutuante). Entre com um código por e-mail — sua conta é criada no primeiro acesso, novos usuários ganham $1 de crédito de teste, e recargas são pagamentos únicos em USDT/USDC sem assinatura. Nomes de modelos existentes (`jev-latest`, `jev-1.14.0`, …) continuam funcionando inalterados: aponte `base_url` e `api_key` para o gateway e escolha qualquer modelo por requisição.

[Cadastrar-se &rarr;](https://www.easyjev.online/) &middot; [Praça de modelos](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [Preços](https://www.easyjev.online/#/pricing) &middot; [Termos](https://www.easyjev.online/#/tos) &middot; [Privacidade](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## Desenvolvimento

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # estilo e checagens estáticas
uv run mypy jevner                          # verificação de tipos
uv run python tests/test_local.py           # testes de algoritmo offline (12 checagens)
uv run python tests/test_e2e.py             # ponta a ponta (requer TYPESAFE_API_KEY, gera custos na API)
uv run python tests/test_languages.py       # varredura de 24 idiomas (gera custos na API)
uv run python tests/bench_quick.py          # experimento de desempenho compacto (gera custos na API)
```

## Contribuindo

Contribuições devem melhorar comportamento verificável do pipeline: recall de detecção, precisão de digitação, fidelidade de aninhamento, validação de taxonomia, comportamento de cache ou segurança de orçamento de requisição. Mantenha a API pública enxuta e adicione testes executáveis de forma independente para cada novo comportamento.

## Licença

[MIT](../LICENSE)
