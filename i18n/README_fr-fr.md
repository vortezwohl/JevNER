<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**Reconnaissance d'entités nommées multilingue zéro-shot. Propulsé par les modèles de jugement System One.**

*Un SDK Python type-safe qui transforme un modèle de jugement en extracteur d'entités nommées — arbres d'entités imbriqués, packs d'entités personnalisés et tokenizer Qwen3 intégré, sans aucun modèle local à entraîner ni déployer.*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="Logo EasyJev" width="72" />
  </a>
  <br/>
  <strong>Sponsorisé par <a href="https://www.easyjev.online/">EasyJev</a> — tous les modèles de décision, un seul point de terminaison.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**Zéro entraînement. Toutes les langues. L'arbre est dans la valeur de retour.**

[English](/vortezwohl/JevNER#readme) | [简体中文](/vortezwohl/JevNER/blob/main/i18n/README_zh-hans.md) | [繁體中文](/vortezwohl/JevNER/blob/main/i18n/README_zh-hant.md) | [日本語](/vortezwohl/JevNER/blob/main/i18n/README_ja-jp.md) | [한국어](/vortezwohl/JevNER/blob/main/i18n/README_ko-kr.md) | Français | [Español](/vortezwohl/JevNER/blob/main/i18n/README_es-es.md) | [Deutsch](/vortezwohl/JevNER/blob/main/i18n/README_de-de.md) | [Português](/vortezwohl/JevNER/blob/main/i18n/README_pt-br.md)

</div>

---

## Pourquoi JevNER

La plupart des stacks NER vous obligent à choisir : un modèle local figé sur un jeu de labels, ou une API hébergée qui masque sa confiance réelle.

JevNER prend un troisième chemin. Un **modèle de jugement System One** — pas un générateur de texte — lit chaque question et renvoie des **probabilités calibrées** au lieu de prose, et le SDK assemble ces jugements en une **forêt d'entités imbriquées** imprimable sous forme d'arbre.

| Là où la NER classique échoue | Ce que fait JevNER |
| --- | --- |
| Les jeux de labels sont figés au fine-tuning ; chaque nouveau domaine impose un nouvel entraînement. | **Zéro-shot par conception.** Les types d'entités sont des données ouvertes : tout utilisateur (y compris non-programmeur) peut ajouter types et packs à l'exécution via un simple document JSON. |
| La couverture multilingue signifie un checkpoint fine-tuné par langue. | **Un seul pipeline, 24 langues vérifiées** — chinois, thaï, arabe, hébreu, hindi, turc, indonésien, philippin, vietnamien, japonais, coréen, anglais, français, allemand, espagnol, portugais, italien, russe, néerlandais, polonais, et plus. Langues peu dotées et écritures sans espaces incluses. |
| Les API NER hébergées renvoient des spans plats ; l'imbrication est aplatie ou perdue. | **La valeur de retour EST l'arbre.** `node.children` contient chaque entité incluse dans `node`, dérivée des offsets du texte source (`text == source[start:end]`) — l'imbrication est structurelle, pas devinée. |
| Un score softmax n'est pas une probabilité sur laquelle router. | **Probabilités calibrées** issues du modèle de jugement : `0.8` se comporte comme « juste 80 % du temps », exactement ce que nécessitent les flux de revue humaine à seuil de confiance. |
| Un span n'a qu'un type, même quand un nom est à la fois un lieu et une organisation. | **Entités multi-types, classées.** `Entity.types` porte chaque type concurrent avec sa probabilité, trié par pertinence. |
| Chaque déploiement télécharge des poids de modèle. | **Aucun modèle local.** Un tokenizer Qwen3 (100+ langues) livré dans le package ; toute l'inférence passe par la passerelle EasyJev. |

## Installation

- Depuis [PyPI](https://pypi.org/project/jevner/)

    ```shell
    pip install -U jevner
    ```

- Recommandé, avec [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- Depuis [GitHub](https://github.com/vortezwohl/JevNER) — pour accéder aux fonctionnalités non publiées.

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## Démarrage rapide

```python
from jevner import JevNER

ner = JevNER()  # identifiants lus depuis TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# La valeur de retour EST la forêt imbriquée : l'imprimer rend un arbre.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

Voici la boucle complète : aucun téléchargement de modèle, aucun fine-tuning, aucun post-traitement. Les entités PERSON sont promues en `Person` avec un jugement de genre, les offsets satisfont toujours `text == source[start:end]`, et l'imbrication voyage dans la valeur de retour elle-même.

> [!TIP]
> La passerelle route plusieurs modèles System One. Le défaut est `baseten/mammoth-1` ; changez à tout moment avec `JevNER(model_name="jev-latest")`.

> Un appel depuis TypeScript, Go ou `curl` brut ? Le protocole est du JSON REST pur — voir [`examples/`](../examples/README.md).

## Capacités clés

| Capacité | Ce que vous obtenez |
| --- | --- |
| Reconnaissance zéro-shot | Pas d'entraînement, pas d'annotation, pas de fine-tuning. Définissez ce qu'il faut extraire et appelez. |
| Fondation modèle de jugement | Questions Noul / Choice avec probabilités calibrées — confiance oui/non et distributions complètes des options, pas de génération libre. |
| Forêt d'entités imbriquées | `ner(text)` renvoie `list[EntityNode]`. Les spans croisés s'attachent en frères au plus proche ancêtre commun ; l'imbrication est dérivée des offsets, pas de devinettes de sous-chaînes. |
| Entités multi-types calibrées | `Entity.types` est une séquence de tuples `(EntityType, probability)` triée par pertinence — une entité peut être GPE *et* LOC. |
| Couverture 24 langues | Vérifiée de bout en bout en Asie, Europe et Amériques, y compris écritures peu dotées et sans espaces. |
| Taxonomie ouverte depuis JSON | `EntityTaxonomy.from_json` accepte un document en trois sections (`entity_types`, `entity_packs`, `default_packs`) qui étend ou surcharge les 31 types et 4 packs intégrés — pensé pour les non-programmeurs. |
| Packs d'entités verticaux | `CORE` (18 types OntoNotes 5.0 + MISC) livré par défaut ; `BIOMED`, `FINANCE` et `TECH` se superposent à la demande. |
| Reconnaissance du genre | Les entités PERSON sont promues en `Person` avec `MALE` / `FEMALE` / `UNKNOWN` et une probabilité, en cache dans le processus. |
| Tokenizer Qwen3 intégré | Un BPE au niveau octet couvrant 100+ langues livré dans le package — pas d'accès HuggingFace, pas de `transformers`, pas de `torch`. |
| Sécurité de budget de requête | Fenêtres dimensionnées sur la limite de corps de 65 535 octets de la passerelle ; lots de typage resserrés par la limite de 512 options — chaque requête reste dans les limites par construction. |
| Cache à trois niveaux | Détection, typage et genre dans des caches LRU indexés par hachage de fenêtre ; le texte répété ou chevauchant revient en millisecondes. |
| Type-safe de bout en bout | Annotations complètes avec `py.typed`, mypy propre en mode strict, prompts rendus via les templates structurés prompt4py. |

## Motifs courants

### Imprimer la forêt, pas une liste plate

La valeur de retour est déjà l'arbre. `str(node)` rend un nœud et ses descendants avec des lignes de tracé, types, probabilités, genre et offsets.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### Activer un pack vertical

Les packs se superposent par nom ; la table de types envoyée au modèle contient exactement l'union de votre sélection.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### Définir une taxonomie depuis JSON

Tout utilisateur — même celui qui n'ouvre jamais d'éditeur de code — peut étendre le vocabulaire de reconnaissance avec un document JSON. Les types personnalisés sont des citoyens de première classe à côté des intégrés.

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

`from_json` accepte une chaîne JSON ou un dict déjà analysé. Les références de types inconnues, les descriptions vides et les packs par défaut non enregistrés lèvent une `ValueError` avec le chemin du champ en faute.

### Router sur la confiance calibrée

Chaque probabilité vient d'un modèle de jugement entraîné pour la calibration, donc les seuils veulent ce qu'ils disent :

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 veut vraiment dire « juste 80 % du temps »
```

Les appels répétés sur le même texte touchent le cache en millisecondes — utile pour les re-requêtes RAG et les reprises de documents.

## Modèles de décision

JevNER tourne sur tout modèle de décision System One enregistré sur la passerelle — des modèles qui répondent aux questions typées (Noul oui/non, Choice sélection) avec des probabilités calibrées au lieu de générer du texte. Défaut `baseten/mammoth-1` ; `jev-latest` et plus de 20 autres se changent en un argument :

| | |
| --- | --- |
| Vérifiés pour la NER | `baseten/mammoth-1` (défaut), `jev-latest`, `jev-1.13.0`, `jev-1.14.0` |
| Moins chers | `baseten/joey-1`, `respan/span-01-lite`, `perplexity/pplx-decider-flash-v1.2` — dès $0.01/M de tokens d'entrée |
| Également disponibles | famille Cloudflare Clef, Nace.AI Drex, LiquidAI d1, Microsoft Decision-1, Upstage Solar Decide, Inception Mercury Decide, OpenAI GPT-6 Luna Decisions, `jaredpalmer/kev-4b`, et plus |

Catalogue complet avec prix et réserves par modèle : [Modèles de décision supportés](../docs/models.md).

## Écosystème et comparaisons

JevNER est une application de référence du mouvement des modèles de décision (System One) — l'écosystème de modèles et de moteurs qui renvoient des jugements typés et calibrés au lieu de prose : la famille **Jev** de TypeSafe, la passerelle agrégée [EasyJev](https://www.easyjev.online/), des moteurs ouverts comme [laya](https://github.com/NandhaKishorM/laya) (moteur de décision non-autorégressif, 100+ langues) et [kev](https://github.com/jaredpalmer/kev) (modèles de décision ouverts sur Qwen), et des adaptateurs comme [AnyJev](https://github.com/nokia-applied-research/AnyJev). La carte : [l'écosystème des modèles de décision](../docs/ecosystem.md).

Vous choisissez une stack NER ? [JevNER vs GLiNER vs spaCy vs API NER hébergées](../docs/alternatives.md) — une comparaison honnête, y compris quand un checkpoint [GLiNER](https://github.com/urchade/GLiNER) local ou un pipeline [spaCy](https://github.com/explosion/spaCy) est le meilleur choix.

## API publique

Le package racine n'expose délibérément que huit noms :

```python
from jevner import (
    Entity,          # une entité reconnue : text, types, start, end
    EntityNode,      # nœud d'arbre : entity + children
    EntityPack,      # regroupement nommé de types d'entités (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # système de types ouvert : registre + from_json + resolve
    EntityType,      # objet valeur label + description
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # le extracteur
    Person,          # Entity + gender + gender_probability
)
```

## Limites de conception

JevNER possède le pipeline de reconnaissance et ses contrats. Il ne fournit volontairement pas :

- d'inférence de modèle local, sous quelque forme que ce soit ;
- d'hébergement de modèle, de fine-tuning ni de distribution de poids ;
- de données d'entraînement, d'outils d'annotation ni de harnais d'évaluation ;
- de composants UI ; ni
- de ré-exports des internes du SDK de la passerelle.

Cette frontière garde le SDK comme un pipeline fin et vérifiable, en laissant l'hébergement et l'expérience produit à la plateforme.

## Sponsor

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="Logo EasyJev" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**Tous les modèles de décision. Une seule API. Des jugements en millisecondes.**

EasyJev est le foyer agrégé des modèles de décision System One — **Jev, Decider, Drex, d1, Clef et plus,
de 14 fournisseurs, 28 modèles en vente aujourd'hui** — derrière un seul point de terminaison compatible `typesafe-sdk`. Une unique passe avant complète chaque jugement en millisecondes ; les tokens de sortie sont facturés $0 sur tous les modèles, et chaque appel est réglé token par token sur un registre à quota entier (1 000 000 000 quotas = $1, sans dérive flottante). Connexion par code e-mail — votre compte est créé à la première connexion, les nouveaux utilisateurs reçoivent 1 $ de crédit d'essai, et les rechargements sont des paiements USDT/USDC ponctuels sans abonnement. Les noms de modèles existants (`jev-latest`, `jev-1.14.0`, …) continuent de fonctionner tels quels : pointez `base_url` et `api_key` vers la passerelle et choisissez n'importe quel modèle par requête.

[S'inscrire &rarr;](https://www.easyjev.online/) &middot; [Place des modèles](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [Tarifs](https://www.easyjev.online/#/pricing) &middot; [Conditions](https://www.easyjev.online/#/tos) &middot; [Confidentialité](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## Développement

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # style et contrôles statiques
uv run mypy jevner                          # vérification des types
uv run python tests/test_local.py           # tests algorithmiques hors ligne (12 contrôles)
uv run python tests/test_e2e.py             # bout en bout (nécessite TYPESAFE_API_KEY, facture l'API)
uv run python tests/test_languages.py       # balayage 24 langues (facture l'API)
uv run python tests/bench_quick.py          # expérience de performance compacte (facture l'API)
```

## Contribuer

Les contributions doivent améliorer un comportement vérifiable du pipeline : rappel de détection, précision de typage, fidélité d'imbrication, validation de taxonomie, comportement du cache ou sécurité de budget de requête. Gardez l'API publique resserrée et ajoutez des tests exécutables indépendamment pour tout nouveau comportement.

## Licence

[MIT](../LICENSE)
