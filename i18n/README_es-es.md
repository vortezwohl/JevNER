<div align="center">

<h1>
  <img src="../assets/jevner-wordmark.svg" alt="JevNER" width="500" />
</h1>

**Reconocimiento de entidades con nombre multilingüe zero-shot. Impulsado por modelos de juicio System One.**

*Un SDK de Python type-safe que convierte un modelo de juicio en un reconocedor de entidades con nombre — árboles de entidades anidadas, paquetes de entidades personalizables y un tokenizador Qwen3 incluido, sin modelos locales que entrenar ni desplegar.*

<p>
  <a href="https://www.easyjev.online/">
    <img src="../assets/easyjev-logo.svg" alt="Logotipo de EasyJev" width="72" />
  </a>
  <br/>
  <strong>Patrocinado por <a href="https://www.easyjev.online/">EasyJev</a> — todos los modelos de decisión, un solo endpoint.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/jevner.svg?style=flat-square&color=8CE838&label=PyPI&cacheSeconds=1)](https://pypi.org/project/jevner/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/jevner/)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square&labelColor=10140A)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vortezwohl/JevNER?style=flat-square&label=Stars)](https://github.com/vortezwohl/JevNER/stargazers)

**Sin entrenamiento. Cualquier idioma. El árbol viene en el valor de retorno.**

[English](../README.md) | [简体中文](README_zh-hans.md) | [繁體中文](README_zh-hant.md) | [日本語](README_ja-jp.md) | [한국어](README_ko-kr.md) | [Français](README_fr-fr.md) | Español | [Deutsch](README_de-de.md) | [Português](README_pt-br.md)

</div>

---

## Por qué JevNER

La mayoría de las stacks de NER te obligan a elegir: un modelo local congelado en un conjunto de etiquetas, o una API alojada que oculta su confianza real.

JevNER toma un tercer camino. Un **modelo de juicio System One** — no un generador de texto — lee cada pregunta y devuelve **probabilidades calibradas** en lugar de prosa, y el SDK ensambla esos juicios en un **bosque de entidades anidadas** que se imprime como un árbol.

| Donde la NER clásica falla | Lo que hace JevNER |
| --- | --- |
| Los conjuntos de etiquetas se congelan al hacer fine-tuning; cada dominio nuevo implica un entrenamiento nuevo. | **Zero-shot por diseño.** Los tipos de entidades son datos abiertos: cualquier usuario (incluidos los no programadores) puede añadir tipos y paquetes en tiempo de ejecución mediante un simple documento JSON. |
| La cobertura multilingüe significa un checkpoint ajustado por idioma. | **Un solo pipeline, 24 idiomas verificados** — chino, tailandés, árabe, hebreo, hindi, turco, indonesio, filipino, vietnamita, japonés, coreano, inglés, francés, alemán, español, portugués, italiano, ruso, neerlandés, polaco y más. Incluye idiomas de bajos recursos y escrituras sin espacios. |
| Las API de NER alojadas devuelven spans planos; el anidamiento se aplana o se pierde. | **El valor de retorno ES el árbol.** `node.children` contiene cada entidad incluida en `node`, derivada de los offsets del texto original (`text == source[start:end]`), así que el anidamiento es estructural, no adivinado. |
| Un puntaje softmax no es una probabilidad sobre la que enrutar. | **Probabilidades calibradas** del modelo de juicio: `0.8` se comporta como «acierta el 80 % de las veces», que es lo que los flujos de revisión humana por confianza necesitan de verdad. |
| Un span recibe un solo tipo, aunque un nombre sea a la vez lugar y organización. | **Entidades multitypo, ordenadas.** `Entity.types` lleva cada tipo competidor con su probabilidad, ordenado por relevancia. |
| Cada despliegue descarga pesos de modelos. | **Cero modelos locales.** Un tokenizador Qwen3 (100+ idiomas) viene dentro del paquete; toda la inferencia pasa por la puerta de enlace EasyJev. |

## Instalación

- Desde [PyPI](https://pypi.org/project/jevner/)

    ```shell
    pip install -U jevner
    ```

- Recomendado, con [uv](https://docs.astral.sh/uv/)

    ```shell
    uv add -U jevner
    ```

- Desde [GitHub](https://github.com/vortezwohl/JevNER) — accede a funciones sin publicar.

    ```shell
    pip install git+https://github.com/vortezwohl/JevNER.git
    ```

## Inicio rápido

```python
from jevner import JevNER

ner = JevNER()  # las credenciales se leen de TYPESAFE_API_KEY / TYPESAFE_BASE_URL

# El valor de retorno ES el bosque anidado: imprimirlo renderiza un árbol.
forest = ner("你好, 我的名字是陈嘉禾, 来自江苏南京.")
for node in forest:
    print(node)
# 陈嘉禾 (PERSON 1.00 | MALE 1.00) [9:12]
# 江苏南京 (GPE 1.00) [16:20]
# ├── 江苏 (GPE 1.00) [16:18]
# └── 南京 (GPE 1.00) [18:20]
```

Ese es el primer ciclo completo: sin descarga de modelos, sin fine-tuning, sin posprocesamiento. Las entidades PERSON se promocionan a `Person` con un juicio de género, los offsets siempre cumplen `text == source[start:end]`, y el anidamiento viaja en el propio valor de retorno.

> [!TIP]
> La puerta de enlace enruta varios modelos System One. El predeterminado es `baseten/mammoth-1`; cámbialo cuando quieras con `JevNER(model_name="jev-latest")`.

> ¿Llamadas desde TypeScript, Go o `curl` puro? El protocolo es JSON REST puro — ver [`examples/`](../examples/README.md).

## Capacidades clave

| Capacidad | Qué obtienes |
| --- | --- |
| Reconocimiento zero-shot | Sin entrenamiento, sin anotación, sin fine-tuning. Define qué extraer y empieza a llamar. |
| Fundación de modelo de juicio | Preguntas Noul / Choice con probabilidades calibradas — confianza sí/no y distribuciones completas de opciones, no generación libre. |
| Bosque de entidades anidadas | `ner(text)` devuelve `list[EntityNode]`. Los spans cruzados se adjuntan al ancestro común más cercano como hermanos; el anidamiento se deriva de los offsets, sin adivinar subcadenas. |
| Entidades multitypo calibradas | `Entity.types` es una secuencia de tuplas `(EntityType, probability)` ordenada por relevancia — una entidad puede ser GPE *y* LOC. |
| Cobertura de 24 idiomas | Verificada de extremo a extremo en Asia, Europa y América, incluidas escrituras de bajos recursos y sin espacios. |
| Taxonomía abierta desde JSON | `EntityTaxonomy.from_json` acepta un documento de tres secciones (`entity_types`, `entity_packs`, `default_packs`) que extiende o sobrescribe los 31 tipos y 4 paquetes integrados — pensado para usuarios no programadores. |
| Paquetes verticales de entidades | `CORE` (18 tipos de OntoNotes 5.0 + MISC) viene por defecto; `BIOMED`, `FINANCE` y `TECH` se apilan a demanda. |
| Reconocimiento de género | Las entidades PERSON se promocionan a `Person` con `MALE` / `FEMALE` / `UNKNOWN` y una probabilidad, en caché dentro del proceso. |
| Tokenizador Qwen3 incluido | Un BPE a nivel de byte que cubre 100+ idiomas dentro del paquete — sin acceso a HuggingFace, sin `transformers`, sin `torch`. |
| Seguridad de presupuesto de peticiones | Ventanas dimensionadas según el límite de cuerpo de 65 535 bytes de la puerta de enlace; lotes de tipado estrechados por el tope de 512 opciones — cada petición queda dentro de los límites por construcción. |
| Caché de tres niveles | Los resultados de detección, tipado y género viven en cachés LRU indexadas por hash de ventana; el texto repetido o solapado vuelve en milisegundos. |
| Type-safe de punta a punta | Anotaciones completas con `py.typed`, mypy limpio en modo estricto, prompts renderizados con las plantillas estructuradas de prompt4py. |

## Patrones comunes

### Imprime el bosque, no una lista plana

El valor de retorno ya es el árbol. `str(node)` renderiza un nodo y sus descendientes con líneas de trazado, tipos, probabilidades, género y offsets.

```python
forest = ner("Bank of America opened a new branch in New York.")
print(forest[0])
# Bank of America (ORG 1.00) [0:15]
# └── America (GPE 0.52 · ORG 0.30) [8:15]
```

### Activa un paquete vertical

Los paquetes se apilan por nombre; la tabla de tipos enviada al modelo contiene exactamente la unión de lo que seleccionaste.

```python
forest = ner(
    "患者确诊为二型糖尿病, 服用阿司匹林控制血糖.",
    entity_packs=["BIOMED"],
)
# 二型糖尿病 (DISEASE 1.00) [5:10]
# ├── 糖尿病 (DISEASE 0.99) [7:10]
# 阿司匹林 (DRUG 1.00) [14:18]
```

### Define una taxonomía desde JSON

Cualquier usuario — incluso quien nunca abre un editor de código — puede ampliar el vocabulario de reconocimiento con un documento JSON. Los tipos personalizados son ciudadanos de primera clase junto a los integrados.

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

`from_json` acepta una cadena JSON o un dict ya parseado. Las referencias a tipos desconocidos, las descripciones vacías y los paquetes predeterminados sin registrar lanzan `ValueError` con la ruta del campo infractor.

### Enruta por confianza calibrada

Cada probabilidad proviene de un modelo de juicio entrenado para la calibración, así que los umbrales significan lo que dicen:

```python
for node in ner(doc, with_gender=False):
    if node.entity.base_probability < 0.7:
        queue_for_human_review(node)   # 0.8 significa de verdad «acierta el 80 % de las veces»
```

Las llamadas repetidas sobre el mismo texto golpean la caché en milisegundos — útil para reconsultas RAG y reejecuciones de documentos.

## Modelos de decisión

JevNER funciona con cualquier modelo de decisión System One registrado en la puerta de enlace — modelos que responden preguntas tipadas (Noul sí/no, Choice selección) con probabilidades calibradas en lugar de generar texto. Predeterminado `baseten/mammoth-1`; `jev-latest` y más de 20 más se cambian con un argumento:

| | |
| --- | --- |
| Verificados para NER | `baseten/mammoth-1` (predeterminado), `jev-latest`, `jev-1.13.0`, `jev-1.14.0` |
| Más baratos | `baseten/joey-1`, `respan/span-01-lite`, `perplexity/pplx-decider-flash-v1.2` — desde $0.01/M de tokens de entrada |
| También disponibles | familia Cloudflare Clef, Nace.AI Drex, LiquidAI d1, Microsoft Decision-1, Upstage Solar Decide, Inception Mercury Decide, OpenAI GPT-6 Luna Decisions, `jaredpalmer/kev-4b`, y más |

Catálogo completo con precios y salvedades por modelo: [Modelos de decisión soportados](../docs/models.md).

## Ecosistema y comparaciones

JevNER es una aplicación de referencia del movimiento de los modelos de decisión (System One) — el ecosistema de modelos y motores que devuelven juicios tipados y calibrados en lugar de prosa: la familia **Jev** de TypeSafe, la puerta de enlace agregada [EasyJev](https://www.easyjev.online/), motores abiertos como [laya](https://github.com/NandhaKishorM/laya) (motor de decisión no autorregresivo, 100+ idiomas) y [kev](https://github.com/jaredpalmer/kev) (modelos de decisión abiertos sobre Qwen), y adaptadores como [AnyJev](https://github.com/nokia-applied-research/AnyJev). El mapa: [el ecosistema de modelos de decisión](../docs/ecosystem.md).

¿Eligiendo una stack de NER? [JevNER vs GLiNER vs spaCy vs API de NER alojadas](../docs/alternatives.md) — una comparación honesta, incluido cuándo un checkpoint [GLiNER](https://github.com/urchade/GLiNER) local o un pipeline [spaCy](https://github.com/explosion/spaCy) es la mejor opción.

## API pública

El paquete raíz expone deliberadamente ocho nombres:

```python
from jevner import (
    Entity,          # una entidad reconocida: text, types, start, end
    EntityNode,      # nodo del árbol: entity + children
    EntityPack,      # paquete nombrado de tipos de entidades (CORE/BIOMED/FINANCE/TECH)
    EntityTaxonomy,  # sistema de tipos abierto: registro + from_json + resolve
    EntityType,      # objeto valor etiqueta + descripción
    Gender,          # MALE / FEMALE / UNKNOWN
    JevNER,          # el reconocedor
    Person,          # Entity + gender + gender_probability
)
```

## Límites de diseño

JevNER es dueño del pipeline de reconocimiento y sus contratos. Intencionadamente no proporciona:

- inferencia de modelos locales de ningún tipo;
- alojamiento de modelos, fine-tuning ni distribución de pesos;
- datos de entrenamiento, herramientas de anotación ni arneses de evaluación;
- componentes de UI; ni
- re-exportaciones de las interioridades del SDK de la puerta de enlace.

Esa frontera mantiene el SDK como un pipeline delgado y verificable, dejando el alojamiento y la experiencia de producto a la plataforma.

## Patrocinador

<div align="center">

<a href="https://www.easyjev.online/">
  <img src="../assets/easyjev-logo.svg" alt="Logotipo de EasyJev" width="128" />
</a>

### [EasyJev](https://www.easyjev.online/)

**Todos los modelos System One. Un solo endpoint. Perfectamente compatible con [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python).**

EasyJev agrega los modelos de decisión System One — Jev, Decider, Drex, d1, Clef, kev y más de todos los proveedores — tras un único endpoint al que el [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) oficial se conecta sin ningún cambio: apunta `base_url` y `api_key` al gateway y elige cualquier modelo por petición; los nombres existentes (`jev-latest`, `jev-1.14.0`, …) siguen funcionando tal cual. Los juicios vuelven en milisegundos, los tokens de salida se facturan a $0 en todos los modelos, y el acceso es con un simple código por correo — tu cuenta se crea en el primer acceso con 1 $ de crédito de prueba. Recarga con tarjeta (Visa / Mastercard), Apple Pay, Google Pay o WeChat Pay — pagos únicos, sin suscripción.

[Registrarse &rarr;](https://www.easyjev.online/) &middot; [Plaza de modelos](https://www.easyjev.online/#/models) &middot; [Playground](https://www.easyjev.online/#/playground) &middot; [Precios](https://www.easyjev.online/#/pricing) &middot; [Términos](https://www.easyjev.online/#/tos) &middot; [Privacidad](https://www.easyjev.online/#/privacy) &middot; support@easyjev.online

</div>

## Desarrollo

```bash
uv run ruff check jevner tests && uv run ruff format --check jevner tests  # estilo y comprobaciones estáticas
uv run mypy jevner                          # comprobación de tipos
uv run python tests/test_local.py           # pruebas de algoritmo sin conexión (12 comprobaciones)
uv run python tests/test_e2e.py             # extremo a extremo (requiere TYPESAFE_API_KEY, factura la API)
uv run python tests/test_languages.py       # barrido de 24 idiomas (factura la API)
uv run python tests/bench_quick.py          # experimento de rendimiento compacto (factura la API)
```

## Contribuir

Las contribuciones deben mejorar un comportamiento verificable del pipeline: recall de detección, precisión de tipado, fidelidad del anidamiento, validación de taxonomías, comportamiento de caché o seguridad del presupuesto de peticiones. Mantén estrecha la API pública y añade pruebas ejecutables de forma independiente para cada comportamiento nuevo.

## Licencia

[MIT](../LICENSE)
