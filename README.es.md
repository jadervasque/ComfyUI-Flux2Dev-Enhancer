<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI-Flux2Dev-Enhancer

Nodos con detección de arquitectura para condicionamiento, referencias, transferencia de identidad y guidance en espacio latente para la familia open-weight FLUX.2 en ComfyUI.

> **Estado:** `1.0.0b1`, beta standalone. La API pública contiene únicamente los nodos canónicos documentados en [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md). Los IDs históricos y aliases de compatibilidad no forman parte de este proyecto.

## Características principales

- Detección estructural de FLUX.2 dev, Klein 4B y Klein 9B.
- Identity Feature Transfer con schedules adaptados a la arquitectura, máscaras, ventanas de denoising, sigmas y matching por chunks.
- Condicionamiento con una o varias referencias.
- Control de atención de referencias y equilibrio texto/referencia.
- Mejora del conditioning y secciones de prompt basadas en el tokenizer.
- Color Anchor e Identity Guidance después de CFG.
- Architecture Inspector para diagnosticar loaders y cuantizaciones.
- Workflows visualmente organizados con notas Markdown en inglés.
- Pruebas automatizadas y CI para Python 3.10–3.12.

## Perfiles compatibles

| Perfil | Detección | Nodos canónicos | Validación |
|---|---:|---:|---|
| FLUX.2 dev | Implementada | Implementados | Validación amplia en GPU en curso |
| FLUX.2 Klein 4B | Implementada | Implementados | Validación amplia en GPU en curso |
| FLUX.2 Klein 9B | Implementada | Implementados | Arquitectura y rutas de código cubiertas |
| Variantes KV cache | Compatible por arquitectura | Depende del loader | Requiere un loader KV compatible |
| Reempaquetados BF16 / FP8 | Compatible por arquitectura | Depende del loader | Probar el loader concreto |
| Reempaquetados GGUF | Compatible por arquitectura | Depende del loader | Probar el loader concreto |

“Depende del loader” significa que debe conservar las APIs de patch y los metadatos de runtime descritos en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Instalación

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Reinicie ComfyUI después de instalar o actualizar.

Las dependencias de desarrollo son opcionales:

```bash
python -m pip install -e ".[dev]"
```

## Orden recomendado del workflow

### Ruta del modelo

```text
Load Diffusion Model
        ↓
Aplicar LoRA(s), cuando se utilicen
        ↓
Controles de referencia, cuando se utilicen
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Ruta del conditioning

```text
Text encoder FLUX.2
        ↓
Conditioning del prompt
        ↓
Herramientas opcionales de conditioning
        ↓
FLUX.2 Multi Reference Latent
        ↓
Controles opcionales de referencia
        ↓
Positive conditioning
```

### Ruta de referencias

```text
Imagen de referencia
        ↓
Redimensionar o recortar
        ↓
FLUX.2 VAE Encode
        ↓
latent_1 ... latent_8
```

El orden de referencias debe corresponder al orden de máscaras:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Nodos canónicos

| Nodo | Función |
|---|---|
| **FLUX.2 Architecture Inspector** | Informa arquitectura, bloques, ancho del conditioning y hooks del loader. |
| **FLUX.2 Conditioning Enhancer** | Escala y normaliza conditioning activo y segmentos de capas del encoder. |
| **FLUX.2 Text Conditioning Enhancer** | Controles simples de magnitud, contraste y normas. |
| **FLUX.2 Sectioned Encoder** | Codifica FRONT/MID/END y registra rangos derivados del tokenizer. |
| **FLUX.2 Detail Controller** | Escala secciones del prompt o rangos explícitos de tokens. |
| **FLUX.2 Multi Reference Latent** | Añade hasta ocho referencias codificadas por VAE al conditioning. |
| **FLUX.2 Reference Attention Control** | Escala keys y values de una referencia, con fade espacial opcional. |
| **FLUX.2 Reference Weight** | Multiplicador ligero aplicado solo al modelo. |
| **FLUX.2 Text/Reference Balance** | Atenúa texto o referencias alrededor de un punto neutro. |
| **FLUX.2 Reference Latent Mask** | Atenúa regiones enmascaradas dentro de un latent de referencia. |
| **FLUX.2 Identity Feature Transfer** | Realiza matching y transferencia de features durante el denoising. |
| **FLUX.2 Color Anchor** | Corrige medias de canales latentes hacia una referencia. |
| **FLUX.2 Identity Guidance** | Corrección latente adaptativa, directa o por estadísticas de canales. |

El contrato completo se encuentra en [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md).

## Identity Feature Transfer

El nodo clona el modelo e instala un patch en la salida de atención. En los bloques y steps seleccionados:

1. separa tokens de texto, imagen generada y referencias;
2. selecciona referencias y aplica máscaras opcionales;
3. centra y normaliza los vectores;
4. calcula similitud coseno por chunks;
5. rechaza coincidencias por debajo de `similarity_floor`;
6. combina features de referencia con softmax controlado por temperatura;
7. aplica transferencia regulada por la confianza.

Transfiere features internas, no píxeles. Identidad, pose, iluminación, ropa, cabello y fondo permanecen parcialmente entrelazados. Se recomiendan presets conservadores, máscaras, ventanas tardías y comparaciones con seed fijo.

### Presets

- `AUTO_SOFT`: mayor libertad para prompt y composición.
- `AUTO_BALANCED`: punto de partida general.
- `AUTO_STRONG`: bloqueo más fuerte y mayor riesgo de copia no deseada.
- `CUSTOM`: schedules explícitos para double y single blocks.

### Modos de fuerza

- `normalized_total`: distribuye una fuerza agregada aproximada entre bloques activos.
- `per_block`: aplica directamente los valores del schedule en cada bloque.

Reduzca `query_chunk_size` si el matching produce un pico de VRAM.

## Workflows de ejemplo

Los workflows mantenidos están en [`example_workflow/`](example_workflow/):

- FLUX.2 dev con una referencia;
- Klein 9B con una referencia;
- dos referencias enmascaradas;
- controles de atención combinados con Identity Feature Transfer.

Cada workflow posee cuatro zonas visuales y una `MarkdownNote` en inglés junto a cada nodo del proyecto demostrado.

## Política de API standalone

Este es un proyecto standalone independiente. Solo se admiten los IDs `Flux2...` incluidos en la referencia de nodos. Los IDs históricos, sampler experimental directo, presets heredados y adaptadores de compatibilidad no se registran.

Los cambios incompatibles en IDs o sockets requieren una nueva versión principal. La compatibilidad con loaders cuantizados depende de los hooks y metadatos conservados, no del nombre del checkpoint.

## Documentación

- [Arquitectura](docs/ARCHITECTURE.md)
- [Referencia de nodos](docs/NODE_REFERENCE.md)
- [Desarrollo](docs/DEVELOPMENT.md)
- [Proceso de release](docs/RELEASES.md)
- [Workflows de ejemplo](example_workflow/README.md)
- [Plan de implementación 1](plans/PLAN1.md)

## Desarrollo

```bash
python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
pytest -q
```

Las pruebas automatizadas validan código, metadatos, registro, estructura de workflows, layout y documentación. No sustituyen las pruebas de calidad visual con checkpoints reales y GPU.

Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`SUPPORT.md`](SUPPORT.md) y [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Créditos

ComfyUI-Flux2Dev-Enhancer es mantenido como proyecto independiente por **Jader Vasque**.

Los conceptos iniciales de transferencia de identidad y partes de la implementación inicial se derivaron de [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer), creado por **capitan01R**. Los avisos de copyright y licencia MIT se conservan en [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md) y [`AUTHORS.md`](AUTHORS.md).

Este repositorio no está afiliado ni respaldado por Black Forest Labs, ComfyUI o el autor del proyecto original. Las licencias de los modelos FLUX.2 son independientes de la licencia MIT de esta extensión.
