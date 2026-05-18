# Brain Watermark

Projeto de investigação para inserção de marcas de água digitais em imagens de ressonância magnética cerebral, com foco em imagens do dataset **Brain Tumor MRI Dataset**. O pipeline prepara os dados, gera uma watermark binária, aplica watermarking DWT-SVD em duas versões e calcula métricas de qualidade visual.

## Objetivos

- Organizar imagens cerebrais em estruturas multiclasses e binárias.
- Gerar uma watermark simples em escala de cinzento.
- Inserir a watermark em imagens MRI usando DWT-SVD.
- Comparar uma abordagem uniforme com uma abordagem adaptativa baseada em ROI/RONI.
- Avaliar a qualidade das imagens com MSE, PSNR e SSIM.

## Estrutura do Projeto

```text
brain_watermark/
├── data/
│   ├── raw/Brain_Tumor_MRI_Dataset/
│   │   ├── Training/
│   │   └── Testing/
│   ├── processed/
│   │   ├── multiclass/
│   │   ├── binary/
│   │   ├── roi_masks/
│   │   └── resized/
│   └── watermark/
├── outputs/
│   ├── watermarked/
│   ├── metrics/
│   ├── figures/
│   └── reports/
└── src/
    ├── experiments/
    ├── preprocessing/
    ├── watermarking/
    ├── evaluation/
    └── utils/
```

## Dataset Esperado

Coloque o dataset bruto em:

```text
data/raw/Brain_Tumor_MRI_Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

As classes usadas pelo projeto são:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Para o dataset binário, as classes `glioma`, `meningioma` e `pituitary` são agrupadas como `tumor`, enquanto `notumor` permanece como `notumor`.

## Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências principais:

```bash
pip install numpy opencv-python PyWavelets scikit-image
```

Execute os comandos a partir da raiz do projeto.

## Workflow

### 1. Preparar o dataset

```bash
python -m src.experiments.run_prepare_dataset
```

Este comando:

- cria as pastas necessárias;
- copia as imagens brutas para `data/processed/multiclass`;
- cria o dataset binário em `data/processed/binary`;
- mostra um resumo com o número de imagens por classe.

### 2. Gerar a watermark

```bash
python -m src.experiments.create_watermark
```

O ficheiro gerado fica em:

```text
data/watermark/watermark.png
```

Por padrão, a watermark tem tamanho `64 x 64` e contém o texto `WM`.

### 3. Gerar máscaras ROI

```bash
python -m src.preprocessing.generate_roi_masks
```

As máscaras são guardadas em:

```text
data/processed/roi_masks/
├── train/
└── test/
```

Estas máscaras são usadas pela versão adaptativa do algoritmo.

### 4. Aplicar watermark V1

```bash
python -m src.experiments.run_watermark_v1
```

A versão V1 aplica DWT-SVD uniforme usando uma sub-banda fixa.

Saídas principais:

```text
outputs/watermarked/v1_uniform/
outputs/metrics/image_quality_results_v1.csv
```

### 5. Aplicar watermark V2

```bash
python -m src.experiments.run_watermark_v2
```

A versão V2 aplica DWT-SVD adaptativo por blocos, usando máscaras ROI para reduzir ou evitar alterações em regiões sensíveis.

Saídas principais:

```text
outputs/watermarked/v2_adaptive/
outputs/metrics/image_quality_results_v2.csv
```

## Métodos de Watermarking

### V1: DWT-SVD uniforme

Implementado em `src/watermarking/watermark_dwt_svd.py`.

Fluxo geral:

1. Converte a imagem para escala numérica.
2. Aplica DWT 2D.
3. Seleciona uma sub-banda, por padrão `HL`.
4. Aplica SVD na sub-banda selecionada.
5. Modifica os valores singulares usando a watermark.
6. Reconstrói a imagem com DWT inversa.

### V2: DWT-SVD adaptativo

Implementado em `src/watermarking/watermark_dwt_svd_adaptive.py`.

Fluxo geral:

1. Aplica DWT 2D.
2. Seleciona a sub-banda configurada.
3. Divide a sub-banda em blocos.
4. Calcula a energia local de cada bloco.
5. Ajusta o fator de inserção `alpha` conforme energia local e interseção com ROI.
6. Insere segmentos da watermark nos valores singulares dos blocos permitidos.
7. Reconstrói a imagem final.

## Configurações Principais

As configurações estão em `src/config.py`.

```python
IMAGE_SIZE = (256, 256)
WATERMARK_SIZE = (64, 64)

DEFAULT_WAVELET = "haar"
DEFAULT_SUBBAND = "HL"

V1_ALPHA = 0.08

V2_ALPHA_BASE = 0.12
V2_BLOCK_SIZE = 32
V2_ROI_ALPHA_FACTOR = 0.0
V2_ROI_THRESHOLD = 0.05
```

## Métricas

As métricas de qualidade visual são calculadas em `src/evaluation/metrics_image.py`:

- **MSE**: erro quadrático médio entre imagem original e imagem marcada.
- **PSNR**: relação sinal-ruído de pico.
- **SSIM**: similaridade estrutural.

Também existem métricas para comparar watermarks em `src/evaluation/metrics_watermark.py`:

- **NC**: normalized correlation.
- **BER**: bit error rate.

## Resultados Atuais

Com base nos ficheiros já existentes em `outputs/metrics/`:

| Versão | Imagens | MSE médio | PSNR médio | SSIM médio | Blocos usados médios |
|---|---:|---:|---:|---:|---:|
| V1 uniforme | 7200 | 0.423841 | 51.923 | 0.995922 | - |
| V2 adaptativa | 7200 | 0.014958 | 67.773 | 0.999810 | 2.751 |

Estes valores indicam que, nesta execução, a versão adaptativa preservou melhor a qualidade visual média das imagens.

## Comandos Úteis

Executar todo o fluxo principal:

```bash
python -m src.experiments.run_prepare_dataset
python -m src.experiments.create_watermark
python -m src.preprocessing.generate_roi_masks
python -m src.experiments.run_watermark_v1
python -m src.experiments.run_watermark_v2
```

Redimensionar e normalizar imagens para uma pasta auxiliar:

```bash
python -m src.preprocessing.resize_normalize_dataset
```

Mostrar resumo do dataset multiclasses:

```bash
python -m src.preprocessing.build_multiclass_dataset
```

## Notas

- As imagens são carregadas em escala de cinzento.
- O tamanho padrão de processamento é `256 x 256`.
- A V2 depende das máscaras ROI; se uma máscara não existir para uma imagem, o código usa uma máscara vazia.
- A pasta `outputs/` concentra as imagens marcadas e os CSVs de métricas.
