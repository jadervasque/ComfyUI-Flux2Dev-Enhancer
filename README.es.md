<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI FLUX.2 Enhancer

Nodos de condicionamiento, referencias latentes, transferencia de identidad,
control de color y guidance con detección de arquitectura para la familia
open-weight FLUX.2 en ComfyUI.

Este proyecto es un fork independiente de
[`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer).
El proyecto y los algoritmos originales fueron creados por **capitan01R**. Este
fork conserva el aviso MIT original y generaliza la extensión más allá de una
arquitectura Klein 9B fija.

> [!IMPORTANT]
> Las licencias de los modelos FLUX.2 son independientes de la licencia MIT de
> esta extensión. “Open weight” no implica que todos los checkpoints permitan el
> mismo uso comercial. Revisa la licencia del modelo exacto que vas a cargar.

## Estado

Versión: **4.0.0 beta**

La extensión detecta la arquitectura y las capacidades del loader en tiempo de
ejecución. La compatibilidad no se decide por el nombre del checkpoint.

| Familia del modelo | Detección | Nodos genéricos | Presets | Validación visual |
|---|---:|---:|---|---|
| FLUX.2 [dev] | Implementada | Implementados | Automáticos conservadores | Validación amplia en GPU pendiente |
| FLUX.2 [klein] 4B distilled | Implementada | Implementados | Automáticos | Validación amplia en GPU pendiente |
| FLUX.2 [klein] 4B base | Implementada | Implementados | Automáticos | Validación amplia en GPU pendiente |
| FLUX.2 [klein] 9B distilled | Implementada | Implementados | Automáticos y legacy | Comportamiento upstream más rutas nuevas |
| FLUX.2 [klein] 9B base | Implementada | Implementados | Automáticos y legacy | Validación amplia en GPU pendiente |
| FLUX.2 [klein] 9B KV | Compatible por arquitectura | Depende del loader | Conservadores | Requiere pruebas del loader KV |
| Repackages BF16 / FP8 | Compatibles por arquitectura | Dependen del loader | Mismo perfil | Probar el loader específico |
| Repackages GGUF | Compatibles por arquitectura | Dependen del loader | Mismo perfil | Probar el loader específico |

“Depende del loader” significa que debe conservar las APIs de patch de ComfyUI y
los metadatos del transformer. Usa **FLUX.2 Architecture Inspector** para revisar
el modelo cargado.

## Por qué importa la detección de arquitectura

Las variantes oficiales no tienen la misma profundidad:

| Perfil | Hidden width | Attention heads | Double blocks | Single blocks | Ancho de texto |
|---|---:|---:|---:|---:|---:|
| FLUX.2 [dev] | 6144 | 48 | 8 | 48 | 15360 |
| FLUX.2 [klein] 9B | 4096 | 32 | 8 | 24 | 12288 |
| FLUX.2 [klein] 4B | 3072 | 24 | 5 | 20 | 7680 |

Un schedule calibrado para Klein 9B no debe copiarse sin cambios a dev o Klein 4B.
La extensión lee las listas reales de bloques, valida schedules personalizados y
proyecta schedules legacy según la profundidad relativa.

## Instalación

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Reinicia ComfyUI después de instalar o actualizar.

No se requieren paquetes adicionales en ejecución aparte de los usados por
ComfyUI. Las pruebas de desarrollo requieren `pytest`.

## Workflows de ejemplo

Los workflows recomendados están en [`example_workflow`](example_workflow):

- `FLUX2_dev_single_reference_identity.json`
- `FLUX2_klein_single_reference_identity.json`
- `FLUX2_multi_reference_masked_identity.json`
- `FLUX2_reference_attention_controls.json`

Los grafos recomendados están organizados visualmente en modelo/prompt,
preparación de referencias, nodos del Enhancer y sampling/salida. Cada nodo del
Enhancer utilizado tiene una **Markdown Note** nativa adyacente, escrita en inglés,
que explica su función, ubicación, comportamiento y controles.

Los workflows Klein heredados permanecen como ejemplos legacy y pruebas de
compatibilidad. Consulta
[`example_workflow/README.md`](example_workflow/README.md).

## Orden recomendado

### Ruta del modelo

```text
Load Diffusion Model
        ↓
Aplicar LoRA(s), cuando se utilicen
        ↓
Controles FLUX.2 de referencia/modelo
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Condicionamiento y referencias

```text
Text encoder FLUX.2
        ↓
Conditioning del prompt
        ↓
FLUX.2 Multi Reference Latent
        ↓
Controles opcionales de referencia
        ↓
Conditioning positivo del sampler

Imagen 1 → VAE Encode FLUX.2 → latent_1
Imagen 2 → VAE Encode FLUX.2 → latent_2
...
```

### Salida

```text
Empty FLUX.2 Latent o imagen fuente codificada
        ↓
Sampler
        ↓
VAE Decode FLUX.2
        ↓
Imagen
```

El canvas de salida está controlado por el latent del sampler. Las dimensiones de
las referencias no definen automáticamente la resolución final.

## Nodos genéricos

| Nodo | Objetivo |
|---|---|
| **FLUX.2 Architecture Inspector** | Informa variante, bloques, ancho del conditioning, guidance, método de referencia y hooks del loader. |
| **FLUX.2 Identity Feature Transfer** | Acerca features generadas a features coincidentes de las referencias mediante schedules y máscaras dependientes de la arquitectura. |
| **FLUX.2 Multi Reference Latent** | Añade hasta ocho referencias codificadas por el VAE con método y modo append/replace. |
| **FLUX.2 Reference Attention Control** | Escala keys y values de una referencia, opcionalmente con fade espacial. |
| **FLUX.2 Reference Weight** | Multiplicador ligero, solo del modelo, para keys y values de una referencia. |
| **FLUX.2 Text/Reference Balance** | Atenúa texto o referencias alrededor de un punto neutro. |
| **FLUX.2 Reference Latent Mask** | Atenúa regiones negras directamente en el latent de referencia. |
| **FLUX.2 Conditioning Enhancer** | Escala, blanquea, iguala y ajusta las tres slices apiladas del encoder. |
| **FLUX.2 Text Conditioning Enhancer** | Controles simples de magnitud, contraste y norma de tokens. |
| **FLUX.2 Sectioned Encoder** | Codifica FRONT/MID/END y registra rangos del tokenizer cuando están disponibles. |
| **FLUX.2 Detail Controller** | Escala secciones o rangos explícitos de tokens. |
| **FLUX.2 Color Anchor** | Corrige estadísticas de color por canal hacia una referencia. |
| **FLUX.2 Identity Guidance** | Aplica una corrección post-CFG del latent hacia un latent de identidad. |

## FLUX.2 Identity Feature Transfer

Durante cada bloque activo del denoising, el nodo:

1. Lee `reference_image_num_tokens`, `img_slice`, `block_type` y `block_index`.
2. Separa tokens de texto, generación y referencias.
3. Construye un banco de referencias y máscaras seleccionadas.
4. Centra y normaliza las features.
5. Calcula similitud coseno.
6. Descarta coincidencias bajo `similarity_floor`.
7. Combina features mediante softmax con temperatura.
8. Aplica transferencia limitada por confianza.
9. Devuelve la atención modificada a las capas restantes.

No copia píxeles y no es un face swap. Identidad, pose, iluminación, cabello,
ropa y fondo continúan parcialmente entrelazados en las features.

### Presets

| Preset | Uso |
|---|---|
| `AUTO_SOFT` | Conserva semejanza con amplia libertad de prompt y composición. |
| `AUTO_BALANCED` | Punto inicial general para edición con identidad. |
| `AUTO_STRONG` | Lock fuerte; usar máscaras y revisar copia de pose/fondo. |
| `KLEIN_LEGACY_HARD` | Schedule hard original proyectado a la arquitectura cargada. |
| `KLEIN_LEGACY_MID` | Ajuste medio original con profundidad proyectada. |
| `KLEIN_LEGACY_SOFT` | Ajuste selectivo original con profundidad proyectada. |
| `CUSTOM` | Usa directamente los schedules proporcionados. |

Los presets automáticos son puntos de partida y no garantizan intensidad visual
idéntica entre modelos, cuantizaciones, resoluciones, samplers o LoRAs.

### Fuerza, denoising y VRAM

`normalized_total` distribuye un blend agregado aproximado:

```python
per_application = 1 - (1 - total_strength) ** (1 / active_applications)
```

`legacy_per_block` aplica cada valor del schedule en todos los bloques activos y
puede hacerse excesivo con más bloques o steps.

- `start_percent` y `end_percent` definen la ventana.
- Conecta `SIGMAS` para progreso y equal-energy confiables.
- Las ventanas tardías suelen conservar mejor la composición.
- Reduce `query_chunk_size` para disminuir el pico de VRAM.
- Reduce resolución o número de referencias cuando sea necesario.

### Máscaras

- `focus_only` limita el banco explícito de transferencia.
- `zero_unmasked_tokens` también bloquea regiones excluidas en la atención nativa
  y requiere soporte de patch de entrada.

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Multi Reference Latent

Las entradas deben ser `LATENT` codificados por el VAE FLUX.2. Los elementos de
batch se separan en referencias individuales manteniendo el orden.

- `replace`: sustituye la lista existente.
- `append`: conserva y añade referencias.
- `model_default`: no fuerza el método; recomendado al principio.
- `index`, `offset`, `uxo/uno` e `index_timestep_zero`: métodos explícitos para
  modelos y loaders compatibles.

Que un método aparezca en la interfaz no garantiza que un loader externo lo
implemente correctamente.

## Controles de atención

**Reference Attention Control** multiplica keys y values de la referencia
seleccionada. `1.0` es neutral. Los fades espaciales crean pesos en espacio de
tokens a partir del latent.

**Text/Reference Balance** es neutral en `0.5`. Por debajo se atenúa el texto; por
encima se atenúan las referencias. Es un control de competencia, no una ganancia
independiente de ambas ramas.

## Herramientas de condicionamiento

Los text encoders oficiales exponen tres hidden states apilados. Conditioning
Enhancer puede ajustarlos cuando el ancho es divisible por tres.

Sectioned Encoder admite:

```text
[FRONT] sujeto y acción principal
[MID] ropa, objetos y escena
[END] iluminación, cámara y estilo
```

Con wrappers oficiales Qwen y Mistral se usan el template y tokenizer reales. Si
el loader oculta el tokenizer, la codificación sigue funcionando, pero no se
inventan rangos exactos.

## Color Anchor e Identity Guidance

Actúan sobre predicciones latentes después de CFG y no son extractores semánticos.

- **Color Anchor** ajusta medias espaciales por canal.
- **Identity Guidance — adaptive** pondera por similitud local.
- **Identity Guidance — direct** acerca todas las posiciones.
- **Identity Guidance — channel_match** ajusta estadísticas de canal.

Conecta `SIGMAS` cuando las ventanas exactas sean importantes.

## Architecture Inspector

Ejecuta **FLUX.2 Architecture Inspector** después del loader al probar checkpoints
o loaders cuantizados. Si falta un hook obligatorio, el nodo muestra un error
accionable en vez de declarar compatibilidad silenciosa.

## Compatibilidad legacy

Los IDs anteriores continúan registrados:

| ID legacy | ID nuevo recomendado |
|---|---|
| `IdentityFeatureTransferFinal` | `Flux2IdentityFeatureTransfer` |
| `Flux2KleinMultiReferenceLatent` | `Flux2MultiReferenceLatent` |
| `Flux2KleinRefLatentController` | `Flux2ReferenceAttentionControl` |
| `Flux2KleinRefLatentWeight` | `Flux2ReferenceWeight` |
| `Flux2KleinTextRefBalance` | `Flux2TextReferenceBalance` |
| `Flux2KleinMaskRefController` | `Flux2ReferenceLatentMask` |
| `Flux2KleinColorAnchor` | `Flux2ColorAnchor` |
| `IdentityGuidance` | `Flux2IdentityGuidance` |
| `Flux2KleinEnhancer` | `Flux2ConditioningEnhancer` |
| `Flux2KleinTextEnhancer` | `Flux2TextConditioningEnhancer` |
| `Flux2KleinSectionedEncoder` | `Flux2SectionedEncoder` |
| `Flux2KleinDetailController` | `Flux2DetailController` |

Los nodos básicos, Advanced y V3 siguen orientados a Klein. El sampler experimental
directo se mantiene solo por compatibilidad.

## Requisitos del loader y cuantización

El loader debe conservar, según el nodo:

- acceso al diffusion model y listas reales de bloques;
- `set_model_attn1_patch`;
- `set_model_attn1_output_patch`;
- `model_options` y callbacks post-CFG;
- `reference_image_num_tokens`;
- `img_slice`, `block_type` y `block_index`;
- forwarding de reference latents.

La cuantización por sí sola no impide la compatibilidad. El problema aparece
cuando un wrapper sustituye o ignora las interfaces de patch de ComfyUI.

## Resolución de problemas

### Identity Transfer no produce efecto

- Confirma que las referencias llegan al conditioning positivo.
- Activa `debug` y revisa los tokens.
- Revisa schedules, ventana de denoising y máscaras.

### Falta de VRAM

- Reduce `query_chunk_size`.
- Reduce resolución y cantidad de referencias.
- Usa un crop facial cuando solo necesites identidad.
- Acorta el schedule activo.

### Copia pose, encuadre o fondo

- Usa una máscara más ajustada.
- Comienza con `AUTO_SOFT`.
- Inicia la transferencia más tarde.
- Aumenta `similarity_floor`.
- Prueba `focus_only` antes de `zero_unmasked_tokens`.

## Desarrollo y pruebas

```bash
python -m py_compile architecture.py scheduling.py flux2_*.py
pytest -q
```

Las pruebas cubren arquitecturas, schedules, proyección legacy, normalización,
metadatos, slicing, secciones, registros, ejecución neutral, integridad de grafos,
documentación Markdown Note y navegación entre READMEs localizados.

Las pruebas de código no demuestran calidad visual. La validación requiere
checkpoints reales, seeds fijos, referencias y ejecución en GPU.

## Plan de implementación

Consulta [`plans/PLAN0.md`](plans/PLAN0.md).

## Créditos

Proyecto y algoritmos originales:

- **capitan01R**
- [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)

Fork multivariante y generalización:

- **Jader Vasque**
- [`ComfyUI-Flux2Dev-Enhancer`](https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer)

Consulta [`NOTICE.md`](NOTICE.md) y [`LICENSE`](LICENSE).

Este repositorio no está afiliado ni respaldado por Black Forest Labs, ComfyUI o
el autor upstream.
