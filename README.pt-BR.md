<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI FLUX.2 Enhancer

Nós de condicionamento, referências latentes, transferência de identidade,
controle de cor e guidance com detecção de arquitetura para a família open-weight
FLUX.2 no ComfyUI.

Este projeto é um fork independente de
[`capitan01R/ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer).
O projeto e os algoritmos originais foram criados por **capitan01R**. Este fork
preserva o aviso MIT original e generaliza a extensão para além de uma arquitetura
Klein 9B fixa.

> [!IMPORTANT]
> As licenças dos modelos FLUX.2 são separadas da licença MIT desta extensão.
> “Open weight” não significa que todos os checkpoints permitem o mesmo uso
> comercial. Consulte a licença do modelo exato que será carregado.

## Estado

Versão: **4.0.0 beta**

A extensão detecta a arquitetura e as capacidades do loader em tempo de execução.
A compatibilidade não é decidida pelo nome do checkpoint.

| Família do modelo | Detecção da arquitetura | Nós genéricos | Estado dos presets | Validação visual |
|---|---:|---:|---|---|
| FLUX.2 [dev] | Implementada | Implementados | Presets automáticos conservadores | Validação ampla em GPU pendente |
| FLUX.2 [klein] 4B distilled | Implementada | Implementados | Presets automáticos | Validação ampla em GPU pendente |
| FLUX.2 [klein] 4B base | Implementada | Implementados | Presets automáticos | Validação ampla em GPU pendente |
| FLUX.2 [klein] 9B distilled | Implementada | Implementados | Presets automáticos e legados | Comportamento upstream mais novos caminhos |
| FLUX.2 [klein] 9B base | Implementada | Implementados | Presets automáticos e legados | Validação ampla em GPU pendente |
| FLUX.2 [klein] 9B KV | Arquiteturalmente compatível | Depende do loader | Conservadores | Exige testes com loader KV |
| Repackages BF16 / FP8 | Arquiteturalmente compatíveis | Dependem do loader | Mesmo perfil arquitetural | Teste o loader específico |
| Repackages GGUF | Arquiteturalmente compatíveis | Dependem do loader | Mesmo perfil arquitetural | Teste o loader específico |

“Depende do loader” significa que o loader deve preservar as APIs de patch do
ComfyUI e os metadados do transformer. Use **FLUX.2 Architecture Inspector** para
verificar o modelo carregado.

## Por que a detecção de arquitetura é necessária

As variantes oficiais não possuem a mesma profundidade:

| Perfil | Hidden width | Attention heads | Double blocks | Single blocks | Largura do texto |
|---|---:|---:|---:|---:|---:|
| FLUX.2 [dev] | 6144 | 48 | 8 | 48 | 15360 |
| FLUX.2 [klein] 9B | 4096 | 32 | 8 | 24 | 12288 |
| FLUX.2 [klein] 4B | 3072 | 24 | 5 | 20 | 7680 |

Um schedule calibrado para Klein 9B não pode ser copiado sem alterações para dev
ou Klein 4B. A extensão lê as listas reais de blocos, valida schedules
personalizados e projeta schedules legados por profundidade relativa.

## Instalação

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Reinicie o ComfyUI após instalar ou atualizar.

Não há dependências adicionais de execução além das já usadas pelo ComfyUI.
Os testes de desenvolvimento exigem `pytest`.

## Workflows de exemplo

Os workflows recomendados estão em [`example_workflow`](example_workflow):

- `FLUX2_dev_single_reference_identity.json`
- `FLUX2_klein_single_reference_identity.json`
- `FLUX2_multi_reference_masked_identity.json`
- `FLUX2_reference_attention_controls.json`

Os grafos recomendados são organizados visualmente em modelo/prompt, preparação
das referências, nós do Enhancer e sampling/saída. Cada nó do Enhancer utilizado
possui uma **Markdown Note** nativa ao lado, em inglês, explicando função, posição,
comportamento e controles relevantes.

Os workflows Klein herdados permanecem como exemplos legados e testes de
compatibilidade. Consulte [`example_workflow/README.md`](example_workflow/README.md).

## Ordem recomendada

### Caminho do modelo

```text
Load Diffusion Model
        ↓
Aplicar LoRA(s), quando utilizadas
        ↓
Controles FLUX.2 de referência/modelo, quando utilizados
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Caminho de condicionamento e referências

```text
Text encoder FLUX.2
        ↓
Conditioning do prompt
        ↓
FLUX.2 Multi Reference Latent
        ↓
Controles opcionais de referência
        ↓
Conditioning positivo do sampler

Imagem 1 → VAE Encode FLUX.2 → latent_1
Imagem 2 → VAE Encode FLUX.2 → latent_2
...
```

### Saída

```text
Empty FLUX.2 Latent ou imagem fonte codificada
        ↓
Sampler
        ↓
VAE Decode FLUX.2
        ↓
Imagem
```

O canvas de saída é controlado pelo latent enviado ao sampler. As dimensões das
referências não definem automaticamente a resolução final.

## Nós genéricos

| Nó | Finalidade |
|---|---|
| **FLUX.2 Architecture Inspector** | Informa variante, blocos, largura do condicionamento, guidance, método de referência e hooks do loader. |
| **FLUX.2 Identity Feature Transfer** | Aproxima features da geração de features correspondentes das referências usando schedules e máscaras adaptados à arquitetura. |
| **FLUX.2 Multi Reference Latent** | Adiciona até oito referências codificadas pelo VAE, com método e modo append/replace. |
| **FLUX.2 Reference Attention Control** | Escala keys e values de uma referência, opcionalmente com fade espacial. |
| **FLUX.2 Reference Weight** | Multiplicador leve, somente no modelo, para keys e values de uma referência. |
| **FLUX.2 Text/Reference Balance** | Atenua texto ou referências ao redor de um ponto neutro. |
| **FLUX.2 Reference Latent Mask** | Atenua regiões pretas diretamente no latent de uma referência. |
| **FLUX.2 Conditioning Enhancer** | Escala, faz whitening, equaliza e ajusta as três fatias empilhadas do encoder. |
| **FLUX.2 Text Conditioning Enhancer** | Controles simplificados de magnitude, contraste e norma dos tokens. |
| **FLUX.2 Sectioned Encoder** | Codifica FRONT/MID/END e registra intervalos obtidos do tokenizer quando disponíveis. |
| **FLUX.2 Detail Controller** | Escala seções ou intervalos explícitos de tokens. |
| **FLUX.2 Color Anchor** | Corrige estatísticas de cor por canal em direção a uma referência. |
| **FLUX.2 Identity Guidance** | Aplica uma correção pós-CFG no latent em direção a um latent de identidade. |

## FLUX.2 Identity Feature Transfer

Durante cada bloco ativo do denoising, o nó:

1. Lê `reference_image_num_tokens`, `img_slice`, `block_type` e `block_index`.
2. Separa tokens de texto, geração e referências.
3. Constrói um banco com referências e máscaras selecionadas.
4. Centraliza e normaliza as features.
5. Calcula similaridade de cosseno.
6. Rejeita correspondências abaixo de `similarity_floor`.
7. Combina referências com softmax controlado por temperatura.
8. Aplica transferência limitada pela confiança.
9. Devolve a saída modificada para os blocos restantes.

Ele não copia pixels e não é um face swap. Identidade, pose, iluminação, cabelo,
roupa e fundo continuam parcialmente entrelaçados nas features.

### Presets

| Preset | Uso |
|---|---|
| `AUTO_SOFT` | Mantém alguma semelhança com ampla liberdade de prompt e composição. |
| `AUTO_BALANCED` | Ponto inicial geral para edições com preservação de identidade. |
| `AUTO_STRONG` | Lock forte; use máscaras e observe cópia de pose/fundo. |
| `KLEIN_LEGACY_HARD` | Schedule hard original projetado para a arquitetura carregada. |
| `KLEIN_LEGACY_MID` | Configuração intermediária original com profundidade projetada. |
| `KLEIN_LEGACY_SOFT` | Configuração seletiva original com profundidade projetada. |
| `CUSTOM` | Usa diretamente os schedules fornecidos. |

Os presets automáticos são pontos de partida, não garantias de intensidade visual
igual entre modelos, quantizações, resoluções, samplers ou LoRAs.

### Modos de força

`normalized_total` trata `total_strength` como blend agregado aproximado:

```python
per_application = 1 - (1 - total_strength) ** (1 / active_applications)
```

`legacy_per_block` aplica os valores do schedule em cada bloco ativo. Reproduz
melhor o comportamento legado, mas pode ficar excessivo com mais blocos ou steps.

### Denoising, VRAM e máscaras

- `start_percent` e `end_percent` definem a janela de atuação.
- Conecte `SIGMAS` para progresso e equal-energy mais confiáveis.
- Janelas tardias geralmente preservam melhor a composição.
- Reduza `query_chunk_size` para diminuir o pico de VRAM.
- Reduza resolução ou quantidade de referências quando necessário.
- `focus_only` limita o banco explícito de transferência.
- `zero_unmasked_tokens` também bloqueia regiões mascaradas na atenção nativa e
  exige suporte a patch de entrada da atenção.

Correspondência:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Multi Reference Latent

As entradas devem ser valores `LATENT` codificados pelo VAE FLUX.2. Itens de batch
são separados em referências individuais mantendo a ordem.

- `replace`: substitui uma lista existente.
- `append`: mantém referências existentes e acrescenta as novas.
- `model_default`: não força método; recomendado inicialmente.
- `index`, `offset`, `uxo/uno` e `index_timestep_zero`: métodos explícitos para
  modelos e loaders compatíveis.

A presença de um método na interface não garante suporte correto por loaders de
terceiros.

## Controles de atenção das referências

**Reference Attention Control** multiplica keys e values da referência selecionada.
`1.0` é neutro. Fades espaciais criam pesos em espaço de tokens a partir do latent.

**Text/Reference Balance** é neutro em `0.5`. Abaixo disso o texto é atenuado;
acima disso as referências são atenuadas. É um controle de disputa, não ganho
independente para os dois lados.

## Ferramentas de condicionamento

Os encoders oficiais expõem três hidden states empilhados. O Conditioning Enhancer
pode ajustar essas fatias quando a largura é divisível por três.

O Sectioned Encoder aceita:

```text
[FRONT] sujeito e ação principal
[MID] roupa, objetos e detalhes do cenário
[END] iluminação, câmera e estilo de renderização
```

Com wrappers oficiais Qwen e Mistral, o template e tokenizer reais são usados. Se
um loader esconder o tokenizer, a codificação continua, mas os intervalos exatos
não são inventados.

## Color Anchor e Identity Guidance

Esses nós atuam sobre previsões do latent após CFG e não são extratores semânticos
de identidade.

- **Color Anchor** ajusta médias espaciais por canal em direção à referência.
- **Identity Guidance — adaptive** pondera a correção por similaridade local.
- **Identity Guidance — direct** aproxima todas as posições do latent.
- **Identity Guidance — channel_match** ajusta estatísticas dos canais.

Conecte `SIGMAS` quando janelas exatas forem importantes.

## Architecture Inspector

Execute **FLUX.2 Architecture Inspector** após o loader ao testar um checkpoint ou
loader quantizado. Caso um hook obrigatório esteja ausente, os nós retornam um
erro explicativo em vez de alegar compatibilidade silenciosamente.

## Compatibilidade com workflows antigos

Os IDs antigos permanecem registrados:

| ID legado | Novo ID recomendado |
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

Os nós básicos, Advanced e V3 continuam específicos do Klein. O sampler
experimental direto também permanece apenas para compatibilidade.

## Requisitos de loader e quantização

O loader deve preservar, conforme o nó:

- acesso ao diffusion model e às listas reais de blocos;
- `set_model_attn1_patch`;
- `set_model_attn1_output_patch`;
- `model_options` e callbacks pós-CFG;
- `reference_image_num_tokens`;
- `img_slice`, `block_type` e `block_index`;
- encaminhamento de reference latents.

A quantização dos pesos, por si só, não impede a compatibilidade. O problema ocorre
quando o wrapper substitui ou ignora as interfaces de patch do ComfyUI.

## Solução de problemas

### Identity Transfer sem efeito

- Confirme que as referências chegam ao conditioning positivo.
- Ative `debug` e verifique a contagem de tokens.
- Confira schedules, janela de denoising e máscaras.

### Falta de VRAM

- Reduza `query_chunk_size`.
- Reduza resolução e quantidade de referências.
- Use crop do rosto quando somente a identidade for necessária.
- Encurte o schedule de blocos.

### Cópia de pose, enquadramento ou fundo

- Use máscara mais restrita.
- Comece com `AUTO_SOFT`.
- Inicie a transferência mais tarde.
- Aumente `similarity_floor`.
- Teste `focus_only` antes de `zero_unmasked_tokens`.

## Desenvolvimento e testes

```bash
python -m py_compile architecture.py scheduling.py flux2_*.py
pytest -q
```

Os testes cobrem arquiteturas, schedules, projeção legada, normalização de força,
metadados, slicing de tokens, seções, registros, pass-through neutro, integridade
dos workflows, documentação Markdown Note e navegação dos READMEs localizados.

Testes de código não provam qualidade visual. A validação visual exige checkpoints
reais, seeds fixos, referências e execução em GPU.

## Plano de implementação

Consulte [`plans/PLAN0.md`](plans/PLAN0.md).

## Créditos

Projeto e algoritmos originais:

- **capitan01R**
- [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)

Fork multivariante e generalização arquitetural:

- **Jader Vasque**
- [`ComfyUI-Flux2Dev-Enhancer`](https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer)

Consulte [`NOTICE.md`](NOTICE.md) e [`LICENSE`](LICENSE).

Este repositório não é afiliado nem endossado pela Black Forest Labs, ComfyUI ou
pelo autor upstream.
