<div align="center">

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [Español](README.es.md)

</div>

# ComfyUI-Flux2Dev-Enhancer

Nós com detecção de arquitetura para condicionamento, referências, transferência de identidade e guidance em espaço latente para a família open-weight FLUX.2 no ComfyUI.

> **Estado:** `1.0.0b1`, beta standalone. A API pública contém somente os nós canônicos documentados em [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md). IDs históricos e aliases de compatibilidade não fazem parte deste projeto.

## Destaques

- Detecção estrutural de FLUX.2 dev, Klein 4B e Klein 9B.
- Identity Feature Transfer com schedules dependentes da arquitetura, máscaras, janelas de denoising, sigmas e matching em chunks.
- Condicionamento com uma ou várias referências.
- Controle da atenção das referências e do equilíbrio texto/referência.
- Aprimoramento de conditioning e seções de prompt baseadas no tokenizer.
- Color Anchor e Identity Guidance após CFG.
- Architecture Inspector para diagnosticar loaders e quantizações.
- Workflows organizados visualmente com notas Markdown em inglês.
- Testes automatizados e CI para Python 3.10–3.12.

## Perfis suportados

| Perfil | Detecção | Nós canônicos | Validação |
|---|---:|---:|---|
| FLUX.2 dev | Implementada | Implementados | Validação ampla em GPU em andamento |
| FLUX.2 Klein 4B | Implementada | Implementados | Validação ampla em GPU em andamento |
| FLUX.2 Klein 9B | Implementada | Implementados | Arquitetura e caminhos de código cobertos |
| Variantes KV cache | Compatível pela arquitetura | Depende do loader | Requer loader KV compatível |
| Reempacotamentos BF16 / FP8 | Compatível pela arquitetura | Depende do loader | Testar o loader específico |
| Reempacotamentos GGUF | Compatível pela arquitetura | Depende do loader | Testar o loader específico |

“Depende do loader” significa que o loader deve preservar as APIs de patch e os metadados de runtime descritos em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Instalação

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/ComfyUI-Flux2Dev-Enhancer.git
```

Reinicie o ComfyUI depois de instalar ou atualizar.

As dependências de desenvolvimento são opcionais:

```bash
python -m pip install -e ".[dev]"
```

## Ordem recomendada do workflow

### Caminho do modelo

```text
Load Diffusion Model
        ↓
Aplicar LoRA(s), quando utilizados
        ↓
Controles de referência, quando utilizados
        ↓
FLUX.2 Identity Feature Transfer
        ↓
Guider / KSampler / SamplerCustom
```

### Caminho do conditioning

```text
Text encoder FLUX.2
        ↓
Conditioning do prompt
        ↓
Ferramentas opcionais de conditioning
        ↓
FLUX.2 Multi Reference Latent
        ↓
Controles opcionais de referência
        ↓
Positive conditioning
```

### Caminho das referências

```text
Imagem de referência
        ↓
Redimensionar ou recortar
        ↓
FLUX.2 VAE Encode
        ↓
latent_1 ... latent_8
```

A ordem das referências deve corresponder à ordem das máscaras:

```text
latent_1 ↔ subject_mask_1
latent_2 ↔ subject_mask_2
...
latent_8 ↔ subject_mask_8
```

## Nós canônicos

| Nó | Finalidade |
|---|---|
| **FLUX.2 Architecture Inspector** | Relata arquitetura, blocos, largura do conditioning e hooks do loader. |
| **FLUX.2 Conditioning Enhancer** | Escala e normaliza conditioning ativo e fatias das camadas do encoder. |
| **FLUX.2 Text Conditioning Enhancer** | Controles simples de magnitude, contraste e normas. |
| **FLUX.2 Sectioned Encoder** | Codifica FRONT/MID/END e registra intervalos derivados do tokenizer. |
| **FLUX.2 Detail Controller** | Escala seções do prompt ou intervalos explícitos de tokens. |
| **FLUX.2 Multi Reference Latent** | Adiciona até oito referências codificadas pelo VAE ao conditioning. |
| **FLUX.2 Reference Attention Control** | Escala keys e values de uma referência, com fade espacial opcional. |
| **FLUX.2 Reference Weight** | Multiplicador leve aplicado somente ao modelo. |
| **FLUX.2 Text/Reference Balance** | Atenua texto ou referências ao redor de um ponto neutro. |
| **FLUX.2 Reference Latent Mask** | Atenua regiões mascaradas em um latent de referência. |
| **FLUX.2 Identity Feature Transfer** | Faz matching e transferência de features durante o denoising. |
| **FLUX.2 Color Anchor** | Corrige médias dos canais latentes em direção à referência. |
| **FLUX.2 Identity Guidance** | Correção latente adaptativa, direta ou por estatística de canais. |

O contrato completo está em [`docs/NODE_REFERENCE.md`](docs/NODE_REFERENCE.md).

## Identity Feature Transfer

O nó clona o modelo e instala um patch na saída da atenção. Nos blocos e steps selecionados, ele:

1. separa tokens de texto, imagem gerada e referências;
2. seleciona referências e aplica máscaras opcionais;
3. centraliza e normaliza os vetores;
4. calcula similaridade de cosseno em chunks;
5. rejeita correspondências abaixo de `similarity_floor`;
6. combina features de referência com softmax controlado por temperatura;
7. aplica transferência regulada pela confiança.

Ele transfere features internas, não pixels. Identidade, pose, iluminação, roupa, cabelo e fundo permanecem parcialmente misturados. Use presets conservadores, máscaras, janelas mais tardias e comparações com seed fixo.

### Presets

- `AUTO_SOFT`: mais liberdade para prompt e composição.
- `AUTO_BALANCED`: ponto inicial geral.
- `AUTO_STRONG`: lock mais forte e maior risco de cópia indesejada.
- `CUSTOM`: schedules explícitos para double e single blocks.

### Modos de força

- `normalized_total`: distribui uma força agregada aproximada pelos blocos ativos.
- `per_block`: aplica diretamente os valores do schedule em cada bloco.

Reduza `query_chunk_size` quando o matching causar pico de VRAM.

## Workflows de exemplo

Os workflows mantidos estão em [`example_workflow/`](example_workflow/):

- FLUX.2 dev com uma referência;
- Klein 9B com uma referência;
- duas referências mascaradas;
- controles de atenção combinados com Identity Feature Transfer.

Cada workflow possui quatro zonas visuais e uma `MarkdownNote` em inglês ao lado de cada nó do projeto demonstrado.

## Política da API standalone

Este é um projeto standalone independente. Somente os IDs `Flux2...` presentes na referência de nós são suportados. IDs históricos, sampler experimental direto, presets herdados e adaptadores de compatibilidade não são registrados.

Mudanças incompatíveis em IDs ou sockets exigem nova versão principal. A compatibilidade com loaders quantizados depende dos hooks e metadados preservados, não do nome do checkpoint.

## Documentação

- [Arquitetura](docs/ARCHITECTURE.md)
- [Referência dos nós](docs/NODE_REFERENCE.md)
- [Desenvolvimento](docs/DEVELOPMENT.md)
- [Processo de release](docs/RELEASES.md)
- [Workflows de exemplo](example_workflow/README.md)
- [Plano de implementação 1](plans/PLAN1.md)

## Desenvolvimento

```bash
python -m compileall -q __init__.py comfyui_flux2dev_enhancer tests
ruff check comfyui_flux2dev_enhancer tests --select E9,F63,F7,F82
pytest -q
```

Os testes automatizados validam código, metadados, registro, estrutura dos workflows, layout e documentação. Eles não substituem testes de qualidade visual com checkpoints reais e GPU.

Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`SUPPORT.md`](SUPPORT.md) e [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Créditos

ComfyUI-Flux2Dev-Enhancer é mantido como projeto independente por **Jader Vasque**.

Conceitos iniciais de transferência de identidade e partes da implementação inicial foram derivados do projeto [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer), criado por **capitan01R**. Os avisos de copyright e licença MIT foram preservados em [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md) e [`AUTHORS.md`](AUTHORS.md).

Este repositório não é afiliado nem endossado pela Black Forest Labs, pelo ComfyUI ou pelo autor do projeto original. As licenças dos modelos FLUX.2 são separadas da licença MIT desta extensão.
