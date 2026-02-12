# 🏥 Classificador de Doenças Pulmonares com Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

Sistema de diagnóstico automatizado que utiliza **Redes Neurais Convolucionais (CNNs)** e **Transfer Learning** para classificar imagens de raio-X em 4 categorias de doenças pulmonares.

---

## 📑 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Fundamentação Matemática](#-fundamentação-matemática)
  - [Redes Neurais Convolucionais](#1-redes-neurais-convolucionais-cnns)
  - [Funções de Ativação](#2-funções-de-ativação)
  - [Pooling](#3-pooling)
  - [Função de Perda](#4-função-de-perda-categorical-cross-entropy)
  - [Otimizador RMSprop](#5-otimizador-rmsprop)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Dataset](#-dataset)
- [Pré-processamento](#-pré-processamento)
- [Métricas de Avaliação](#-métricas-de-avaliação)
- [Resultados](#-resultados)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Referências](#-referências)

---

## 🎯 Sobre o Projeto

### Contexto

Técnicas de **Inteligência Artificial** e **Deep Learning** estão revolucionando a medicina diagnóstica. Este projeto simula um sistema contratado por um hospital para auxiliar na **detecção automatizada de doenças pulmonares** através de imagens de raio-X.

### Objetivo

> **Automatizar a classificação de doenças pulmonares a partir de imagens de raio-X, reduzindo tempo e custo do diagnóstico.**

### Classes de Classificação

| Código | Classe | Descrição |
|:------:|--------|-----------|
| 0 | **Covid-19** | Pacientes diagnosticados com COVID-19 |
| 1 | **Normal** | Pacientes saudáveis (sem doença pulmonar) |
| 2 | **Pneumonia Viral** | Pneumonia causada por vírus |
| 3 | **Pneumonia Bacteriana** | Pneumonia causada por bactérias |

---

## 📐 Fundamentação Matemática

### 1. Redes Neurais Convolucionais (CNNs)

As CNNs são arquiteturas especializadas em processar dados com estrutura de grade (como imagens). A operação fundamental é a **convolução**.

#### Operação de Convolução

A convolução 2D entre uma imagem `I` e um kernel (filtro) `K` é definida como:

```
(I * K)(i,j) = Σₘ Σₙ I(i+m, j+n) · K(m,n)
```

**Onde:**
- `I` = Imagem de entrada (matriz de pixels)
- `K` = Kernel/Filtro (matriz de pesos aprendíveis)
- `(i,j)` = Posição no mapa de características de saída
- `(m,n)` = Índices do kernel

#### Exemplo Numérico

Considere uma imagem 4x4 e um kernel 3x3:

```
Imagem I (4x4):          Kernel K (3x3):
┌─────────────────┐      ┌───────────┐
│  1   2   3   0  │      │  1  0  -1 │
│  0   1   2   3  │      │  1  0  -1 │
│  3   0   1   2  │      │  1  0  -1 │
│  2   3   0   1  │      └───────────┘
└─────────────────┘

Cálculo para posição (0,0):
(I * K)(0,0) = 1×1 + 2×0 + 3×(-1) +
               0×1 + 1×0 + 2×(-1) +
               3×1 + 0×0 + 1×(-1)
             = 1 + 0 - 3 + 0 + 0 - 2 + 3 + 0 - 1
             = -2
```

O kernel desliza pela imagem calculando o produto escalar em cada posição, gerando um **mapa de características** (feature map).

---

### 2. Funções de Ativação

#### ReLU (Rectified Linear Unit)

Usada nas camadas intermediárias para introduzir não-linearidade:

```
f(x) = max(0, x)
```

```
         │
    y    │      ╱
         │     ╱
         │    ╱
    ─────┼───╱────── x
         │  ╱
         │ ╱
         │╱
```

**Propriedades:**
- Se `x > 0`: saída = `x`
- Se `x ≤ 0`: saída = `0`
- Resolve o problema do gradiente desvanecente
- Computacionalmente eficiente

#### Softmax (Camada de Saída)

Converte os logits em probabilidades para classificação multi-classe:

```
σ(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

**Onde:**
- `zᵢ` = logit da classe `i`
- `Σⱼ e^zⱼ` = soma das exponenciais de todos os logits
- A soma de todas as probabilidades = 1

#### Exemplo com 4 Classes

```
Logits (saída da última camada densa):
z = [2.0, 1.0, 0.5, 0.1]

Cálculo:
e^2.0 = 7.389    e^1.0 = 2.718    e^0.5 = 1.649    e^0.1 = 1.105
Soma = 7.389 + 2.718 + 1.649 + 1.105 = 12.861

Probabilidades (Softmax):
P(Covid-19)     = 7.389 / 12.861 = 0.574 (57.4%)
P(Normal)       = 2.718 / 12.861 = 0.211 (21.1%)
P(Viral)        = 1.649 / 12.861 = 0.128 (12.8%)
P(Bacteriana)   = 1.105 / 12.861 = 0.086 (8.6%)

Previsão Final: Classe 0 (Covid-19) com 57.4% de confiança
```

---

### 3. Pooling

O **Average Pooling** reduz a dimensionalidade calculando a média de regiões:

```
out(i,j) = (1/k²) × Σₘ Σₙ in(i·s+m, j·s+n)
```

**Onde:**
- `k` = tamanho do pool (ex: 7x7)
- `s` = stride (passo)

#### Exemplo (Pool 2x2)

```
Entrada (4x4):              Saída (2x2):
┌─────────────────┐         ┌───────────┐
│  1   3 │ 2   4  │         │   2.0   3.0   │
│  2   4 │ 3   5  │   →     │   1.5   2.5   │
├────────┼────────┤         └───────────────┘
│  0   2 │ 1   3  │
│  1   3 │ 2   4  │         Cálculo:
└─────────────────┘         (1+3+2+4)/4 = 2.5
                            (2+4+3+5)/4 = 3.5
                            etc.
```

---

### 4. Função de Perda (Categorical Cross-Entropy)

Mede a diferença entre a distribuição prevista e a real:

```
L = -Σᵢ yᵢ · log(ŷᵢ)
```

**Onde:**
- `yᵢ` = valor real (one-hot encoded: 0 ou 1)
- `ŷᵢ` = probabilidade prevista pelo modelo
- `log` = logaritmo natural

#### Exemplo Numérico

```
Classe Real: Covid-19 (índice 0)
y = [1, 0, 0, 0]  (one-hot encoding)

Previsão do Modelo:
ŷ = [0.85, 0.05, 0.07, 0.03]

Cálculo da Loss:
L = -(1×log(0.85) + 0×log(0.05) + 0×log(0.07) + 0×log(0.03))
L = -log(0.85)
L = -(-0.163)
L = 0.163

→ Quanto menor a loss, melhor a previsão
→ Se ŷ₀ = 1.0 (previsão perfeita), L = -log(1) = 0
```

---

### 5. Otimizador RMSprop

Adapta a taxa de aprendizado para cada parâmetro usando média móvel dos gradientes ao quadrado:

```
E[g²]ₜ = γ · E[g²]ₜ₋₁ + (1-γ) · gₜ²

θₜ₊₁ = θₜ - η · gₜ / √(E[g²]ₜ + ε)
```

**Onde:**
- `E[g²]ₜ` = média móvel dos gradientes ao quadrado
- `γ` = fator de decaimento (tipicamente 0.9)
- `gₜ` = gradiente no tempo `t`
- `η` = taxa de aprendizado (learning rate = 1e-4 no projeto)
- `ε` = constante para estabilidade numérica (≈ 1e-8)
- `θ` = parâmetros do modelo (pesos)

**Intuição:** Parâmetros com gradientes grandes recebem atualizações menores, e vice-versa.

---

## 🏗 Arquitetura do Modelo

### Transfer Learning com ResNet50

Utilizamos a **ResNet50** pré-treinada no ImageNet (1M+ imagens) como extrator de características.

#### Conexões Residuais (Skip Connections)

A inovação da ResNet é a conexão residual:

```
y = F(x, {Wᵢ}) + x
```

```
        ┌─────────────────────────────────┐
        │                                 │
        │         ┌───────────┐           │
   x ───┼────────►│  Conv 3x3 │           │
        │         └─────┬─────┘           │
        │               │                 │
        │         ┌─────▼─────┐           │
        │         │   ReLU    │           │
        │         └─────┬─────┘           │
        │               │                 │
        │         ┌─────▼─────┐           │
        │         │  Conv 3x3 │           │
        │         └─────┬─────┘           │
        │               │                 │
        │         ┌─────▼─────┐           │
        └────────►│     +     │◄──────────┘
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │   ReLU    │
                  └─────┬─────┘
                        │
                        ▼
                   y = F(x) + x
```

**Benefício:** Permite treinar redes muito profundas sem degradação do gradiente.

### Arquitetura Completa

```
┌──────────────────────────────────────────────────────────────────┐
│                         ENTRADA                                   │
│                    Imagem 256×256×3                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    ResNet50 (Pré-treinada)                        │
│                    ~23 milhões de parâmetros                      │
│                    175 camadas convolucionais                     │
│                    Saída: 8×8×2048                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                  AveragePooling2D (7×7)                           │
│                    Saída: 1×1×2048                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                       Flatten                                     │
│                    Saída: 2048                                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                 Dense (256 neurônios, ReLU)                       │
│                 Dropout (20%)                                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                 Dense (256 neurônios, ReLU)                       │
│                 Dropout (20%)                                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                 Dense (4 neurônios, Softmax)                      │
│                    SAÍDA: Probabilidades                          │
│              [P(Covid), P(Normal), P(Viral), P(Bacterial)]        │
└──────────────────────────────────────────────────────────────────┘
```

### Tabela de Camadas Customizadas

| Camada | Tipo | Output Shape | Parâmetros |
|--------|------|--------------|------------|
| ResNet50 | Base | (8, 8, 2048) | 23,587,712 |
| AveragePooling2D | Pooling | (1, 1, 2048) | 0 |
| Flatten | Reshape | (2048,) | 0 |
| Dense | FC + ReLU | (256,) | 524,544 |
| Dropout | Regularização | (256,) | 0 |
| Dense | FC + ReLU | (256,) | 65,792 |
| Dropout | Regularização | (256,) | 0 |
| Dense | FC + Softmax | (4,) | 1,028 |

**Total de Parâmetros:** ~24 milhões

---

## 📊 Dataset

### Distribuição

| Classe | Treinamento | Teste | Total |
|--------|-------------|-------|-------|
| Covid-19 | 133 | 10 | 143 |
| Normal | 133 | 10 | 143 |
| Pneumonia Viral | 133 | 10 | 143 |
| Pneumonia Bacteriana | 133 | 10 | 143 |
| **Total** | **532** | **40** | **572** |

### Fonte dos Dados

- [COVID-19 Chest X-Ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## ⚙️ Pré-processamento

### 1. Redimensionamento

Todas as imagens são redimensionadas para **256×256 pixels**:

```python
img = cv2.resize(img, (256, 256))
```

### 2. Normalização

Os valores dos pixels são normalizados de [0, 255] para [0, 1]:

```python
img_normalizada = img / 255.0
```

**Fórmula:**
```
x_norm = x / 255
```

**Exemplo:**
```
Pixel original: 128
Pixel normalizado: 128 / 255 = 0.502
```

### 3. One-Hot Encoding

Os rótulos são convertidos para vetores binários:

```
Covid-19:           [1, 0, 0, 0]
Normal:             [0, 1, 0, 0]
Pneumonia Viral:    [0, 0, 1, 0]
Pneumonia Bacterial:[0, 0, 0, 1]
```

---

## 📏 Métricas de Avaliação

### Fórmulas

#### Accuracy (Acurácia)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Precision (Precisão)
```
Precision = TP / (TP + FP)
```
*"Das previsões positivas, quantas estavam corretas?"*

#### Recall (Sensibilidade)
```
Recall = TP / (TP + FN)
```
*"Dos casos positivos reais, quantos foram detectados?"*

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
*Média harmônica entre Precision e Recall*

**Onde:**
- `TP` = True Positives (Verdadeiros Positivos)
- `TN` = True Negatives (Verdadeiros Negativos)
- `FP` = False Positives (Falsos Positivos)
- `FN` = False Negatives (Falsos Negativos)

---

## 📈 Resultados

### Métricas por Classe

| Classe | Precision | Recall | F1-Score | Support |
|--------|:---------:|:------:|:--------:|:-------:|
| Covid-19 | 83% | **100%** | 91% | 10 |
| Normal | 62% | 100% | 77% | 10 |
| Pneumonia Viral | 83% | 50% | 62% | 10 |
| Pneumonia Bacterial | 83% | 50% | 62% | 10 |
| **Média** | **78%** | **75%** | **73%** | **40** |

### Matriz de Confusão

```
                    PREVISTO
              Covid  Normal  Viral  Bact.
         ┌────────────────────────────────┐
  Covid  │   10  │   0   │   0   │   0   │  → 100% Recall
         ├────────────────────────────────┤
  Normal │    0  │  10   │   0   │   0   │  → 100% Recall
REAL     ├────────────────────────────────┤
  Viral  │    2  │   3   │   5   │   0   │  →  50% Recall
         ├────────────────────────────────┤
  Bact.  │    0  │   3   │   1   │   6   │  →  60% Recall
         └────────────────────────────────┘
```

### Destaques

✅ **100% de Recall para Covid-19** - Todos os casos positivos foram detectados
✅ **100% de Recall para Normal** - Nenhum saudável foi diagnosticado incorretamente
⚠️ **50% de Recall para Pneumonias** - Dificuldade em distinguir tipos de pneumonia

### Evolução do Treinamento

```
Época 1:  Accuracy: 75.4%  |  Loss: 0.639
Época 5:  Accuracy: 97.2%  |  Loss: 0.104
Época 8:  Accuracy: 99.4%  |  Loss: 0.019  ← Melhor modelo salvo
Época 10: Accuracy: 98.9%  |  Loss: 0.022
```

---

## 🚀 Como Usar

### Requisitos

```
tensorflow>=2.0
numpy
opencv-python
matplotlib
seaborn
scikit-learn
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/departamento-medico.git
cd departamento-medico

# Instale as dependências
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

### Exemplo de Uso

```python
from keras.models import load_model
import cv2
import numpy as np

# Carregar o modelo treinado
model = load_model('melhor_modelo.keras')

# Definir mapeamento de classes
classes = {
    0: 'Covid-19',
    1: 'Normal',
    2: 'Pneumonia Viral',
    3: 'Pneumonia Bacteriana'
}

# Função para classificar uma imagem
def classificar_raio_x(caminho_imagem):
    # Carregar e pré-processar
    img = cv2.imread(caminho_imagem)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 3)

    # Fazer previsão
    predicao = model.predict(img)
    classe_idx = np.argmax(predicao)
    confianca = predicao[0][classe_idx] * 100

    return classes[classe_idx], confianca

# Usar
resultado, confianca = classificar_raio_x('raio_x_paciente.jpg')
print(f"Diagnóstico: {resultado}")
print(f"Confiança: {confianca:.2f}%")
```

**Saída esperada:**
```
Diagnóstico: Covid-19
Confiança: 97.85%
```

---

## 📁 Estrutura do Projeto

```
Departamento_Médico/
│
├── Dataset/                      # Dados de treinamento
│   ├── 0/                        # Covid-19 (133 imagens)
│   ├── 1/                        # Normal (133 imagens)
│   ├── 2/                        # Pneumonia Viral (133 imagens)
│   └── 3/                        # Pneumonia Bacteriana (133 imagens)
│
├── Test/                         # Dados de teste
│   ├── 0/                        # Covid-19 (10 imagens)
│   ├── 1/                        # Normal (10 imagens)
│   ├── 2/                        # Pneumonia Viral (10 imagens)
│   └── 3/                        # Pneumonia Bacteriana (10 imagens)
│
├── Departamento_Médico.ipynb     # Notebook principal
├── melhor_modelo.keras           # Modelo treinado
└── README.md                     # Este arquivo
```

---

## 📚 Referências

1. **ResNet Original:**
   - He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
   - [Paper](https://arxiv.org/abs/1512.03385)

2. **Transfer Learning:**
   - Yosinski, J., et al. (2014). *How transferable are features in deep neural networks?*
   - [Paper](https://arxiv.org/abs/1411.1792)

3. **COVID-19 Detection:**
   - Wang, L., et al. (2020). *COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images*
   - [Paper](https://arxiv.org/abs/2003.09871)

4. **Datasets:**
   - [IEEE COVID-19 Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
   - [Kaggle Chest X-Ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👥 Créditos

**Desenvolvido para o Tech Challenge - Fase 4**
**FIAP - Pós Tech em Inteligência Artificial para Devs**

---

<p align="center">
  <i>⚠️ Este projeto é apenas para fins educacionais. Não substitui diagnóstico médico profissional.</i>
</p>
