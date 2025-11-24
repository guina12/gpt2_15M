# GPT-2 - Treinamento do Zero

<img width="1000" height="600" alt="train_an_val_loss" src="https://github.com/user-attachments/assets/eec009b4-d329-4e2d-a5db-83fab7772bda" />


Implementação e treinamento de um modelo GPT-2 com 15 milhões de parâmetros, incluindo análise detalhada de complexidade computacional.

##  Índice

- [Visão Geral](#visão-geral)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Configuração de Treinamento](#configuração-de-treinamento)
- [Análise de Complexidade](#análise-de-complexidade)
- [Métricas de Performance](#métricas-de-performance)
- [Estrutura do Projeto](#estrutura-do-projeto)

##  Visão Geral

Este projeto implementa uma arquitetura GPT-2 do zero usando PyTorch, focando no entendimento dos requisitos computacionais e desafios de otimização de modelos de linguagem baseados em transformers. A implementação segue as especificações do paper original do GPT-2 com contagem reduzida de parâmetros adequada para treinamento em GPU única.

### Características Principais

- **15.103.616 parâmetros** totais
- **4 camadas** de transformer
- **256 dimensões** de embedding
- **4 cabeças** de atenção
- **256 tokens** de comprimento de contexto
- **Vocabulário** de 50.304 tokens

## 💻 Requisitos do Sistema

### Hardware Mínimo

- **GPU**: NVIDIA com pelo menos 6GB VRAM (testado em RTX 4050)
- **RAM**: 16GB recomendado
- **Armazenamento**: 5GB para dataset e checkpoints

### Software

- **Sistema Operacional**: Linux, Windows 10/11, ou macOS
- **Python**: 3.8 ou superior
- **CUDA**: 11.8 ou superior (para GPU NVIDIA)

### Dependências Python

```txt
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

##  Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/gpt2-from-scratch.git
cd gpt2-from-scratch
```

### 2. Crie o Ambiente Virtual

**No Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**No Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verifique a Instalação do PyTorch com CUDA

```bash
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

##  Como Executar

### Ativação do Ambiente Virtual

Antes de executar o script, sempre ative o ambiente virtual:

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### Execução do Treinamento

```bash
python gpt2.py
```

### Opções de Linha de Comando

```bash
# Treinamento básico
python gpt2.py

# Com configurações customizadas
python gpt2.py --batch-size 64 --learning-rate 1e-4 --epochs 10

# Continuar de um checkpoint
python gpt2.py --resume checkpoint.pt

# Modo de avaliação apenas
python gpt2.py --eval-only --checkpoint model_final.pt
```

### Desativação do Ambiente Virtual

Quando terminar:
```bash
deactivate
```

##  Arquitetura do Modelo

### Especificações

| Componente | Especificação |
|-----------|--------------|
| Parâmetros Totais | 15.103.616 |
| Camadas Transformer | 4 |
| Dimensão de Embedding | 256 |
| Cabeças de Atenção | 4 |
| Comprimento de Contexto | 256 tokens |
| Tamanho do Vocabulário | 50.304 |
| Taxa de Dropout | 0.7 |
| Função de Ativação | GELU |

### Componentes da Arquitetura

**Token Embedding**: 50.304 × 256 = 12.877.824 parâmetros

**Positional Embedding**: 256 × 256 = 65.536 parâmetros

**Blocos Transformer** (4 camadas):
- Auto-atenção multi-cabeça com mascaramento causal
- Rede feed-forward posicional (expansão 4×)
- Normalização de camada
- Conexões residuais

## ⚙️ Configuração de Treinamento

### Hiperparâmetros

```python
batch_size = 32
sequence_length = 256
learning_rate = 3e-4  # Otimizador AdamW
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
gradient_clip = 1.0
```

### Dataset

- **Conjunto de Treinamento**: 4M tokens
- **Batches por Época**: 2.848
- **Tokens por Época**: 23.330.816

## 📊 Análise de Complexidade

### Complexidade de Tempo

Para um modelo Transformer com L camadas, dimensão d e comprimento de sequência n:

**Camada de Auto-Atenção**: O(n² × d)
- Projeções Query, Key, Value: O(n × d²)
- Computação de atenção: O(n² × d)
- Projeção de saída: O(n × d²)

**Rede Feed-Forward**: O(n × d²)
- Duas transformações lineares com dimensão intermediária 4d

**Total por Camada**: O(n² × d + n × d²)

**Modelo Completo**: O(L × (n² × d + n × d²))

### Cálculo de FLOPs

**Forward Pass**:
```
FLOPs = 6 × parâmetros × sequence_length
FLOPs = 6 × 15.103.616 × 256
FLOPs = 23,2 GFLOPs por sequência
```

**Treinamento (Forward + Backward)**:
```
FLOPs = 23,2 × 3 = 69,6 GFLOPs por sequência
```

**Por Época**:
```
Sequências = 23.330.816 / 256 = 91.136
Total FLOPs = 69,6 × 10⁹ × 91.136 = 6,34 × 10¹⁵ FLOPs
```

##  Métricas de Performance

### Performance de Treinamento

| Métrica | Valor |
|--------|-------|
| Tokens por Época | 23.330.816 |
| Tempo de Treinamento | 37,5 min/época |
| Throughput | 10.369 tokens/seg |
| Sequências/seg | 40,5 |
| Tempo por Batch | 0,79 segundos |
| FLOPs Efetivos | 2,82 TFLOPS |
| MFU (Utilização de FLOPs) | 4,03% |

### Curvas de Loss

**Loss de Treinamento**:
- Inicial: 13,0
- Final: 5,3
- Redução: 59,2%

**Loss de Validação**:
- Inicial: 13,0
- Final: 9,4
- Estabilização: ~9,0 após 40M tokens

**Gap Treino-Validação**:
- Indica leve overfitting após 40M tokens
- Recomendado: early stopping ou aumento de regularização

##  Estrutura do Projeto

```
gpt2/
├── gpt2.py                 # Script principal
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
├── data/                  # Diretório de dados
│   ├── train.txt
│   └── val.txt
├── checkpoints/           # Checkpoints do modelo
│   ├── checkpoint_epoch_1.pt
│   └── model_final.pt
└── logs/                  # Logs de treinamento
    └── training.log
```

##  Solução de Problemas

### CUDA Out of Memory

Se encontrar erros de memória:
```python
# Reduza o batch size
batch_size = 16  # ou 8

# Ou reduza o sequence length
sequence_length = 128
```

### Performance Lenta

1. Verifique se CUDA está disponível
2. Use mixed precision training (FP16)
3. Aumente o batch size se houver VRAM disponível
4. Use gradient accumulation para batches maiores efetivos

##  Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

##  Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

##  Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.

- OpenAI pelo paper original do GPT-2
- Comunidade PyTorch
- Andrej Karpathy pelos tutoriais de implementação

---

**Nota**: Este é um projeto educacional. Para aplicações em produção, considere usar implementações otimizadas como Hugging Face Transformers.
