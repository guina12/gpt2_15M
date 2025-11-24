# GPT-2 - Treinamento do Zero

<table>
  <tr>
    <td align="center">
      <img width="450" alt="train_and_val_loss_4M_32batch" src="https://github.com/user-attachments/assets/eec009b4-d329-4e2d-a5db-83fab7772bda" />
      <br>
      <b>Treino com 4 milhões de tokens — Batch size 32</b>
    </td>
    <td align="center">
      <img width="450" alt="train_and_val_loss_10M_128batch" src="https://github.com/user-attachments/assets/67f69a86-fc1b-4c05-aba9-1fab6e6a9c46" />
      <br>
      <b>Treino com 10 milhões de tokens — Batch size 128</b>
    </td>
  </tr>
</table>


## Texto Gerado pelo modelo após o treinamento

  <img width="729" height="779" alt="image" src="https://github.com/user-attachments/assets/d162ed0c-9015-4667-8195-21ac455ec6d4" />




# Implementação e treinamento de um modelo GPT-2 com 15 milhões de parâmetros, incluindo análise detalhada de complexidade computacional.

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

##  Requisitos do Sistema

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
cd gpt2.py
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
# Treinamento
```bash
python gpt2.py
```

## Arquitetura do Modelo

### Especificações do Modelo

| Componente | Especificação |
|---------------------------|------------------|
| Parâmetros Totais | 15.103.616 |
| Camadas Transformer | 4 |
| Dimensão de Embedding | 256 |
| Cabeças de Atenção | 4 |
| Comprimento de Contexto | 32 tokens |
| Tamanho do Vocabulário | 50.304 |
| Taxa de Dropout | 0.7 |
| Função de Ativação | GELU |

### Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|---------------------------|------------------|
| Batch Size | 32 |
| Sequence Length | 32 |
| Learning Rate | 3e-4 |
| Weight Decay | 0.1 |
| Beta1 (AdamW) | 0.9 |
| Beta2 (AdamW) | 0.95 |
| Gradient Clip Norm | 1.0 |

### Componentes da Arquitetura

| Componente | Dimensões | Parâmetros |
|---------------------------|------------------|------------------|
| Token Embedding | 50.304 × 256 | 12.877.824 |
| Positional Embedding | 256 × 256 | 65.536 |
| Blocos Transformer | 4 camadas | - |
| Auto-atenção Multi-cabeça | 4 cabeças | Com mascaramento causal |
| Rede Feed-Forward | Expansão 4× | Dimensão intermediária 1024 |
| Normalização de Camada | Layer Norm | Após cada sub-camada |
| Conexões Residuais | Skip connections | Em cada bloco |

### Dataset

| Métrica | Valor |
|---------------------------|------------------|
| Conjunto de Treinamento | 4M tokens |
| Batches por Época | 2.848 |
| Tokens por Época | 23.330.816 |

## Análise de Complexidade

### Complexidade de Tempo por Camada

| Operação | Complexidade | Descrição |
|---------------------------|------------------|--------------------------------|
| Projeções QKV | O(n × d²) | Query, Key, Value projections |
| Computação de Atenção | O(n² × d) | Attention scores e aplicação |
| Projeção de Saída | O(n × d²) | Output projection |
| Feed-Forward Camada 1 | O(n × d × 4d) | Expansão para 4d |
| Feed-Forward Camada 2 | O(n × 4d × d) | Redução para d |
| **Total por Camada** | **O(n² × d + n × d²)** | Complexidade combinada |

### Complexidade do Modelo Completo

| Componente | Fórmula | Valor (L=4, d=256, n=32) |
|---------------------------|---------------------|--------------------------|
| Auto-Atenção | L × n² × d | 4 × 1.024 × 256 |
| Feed-Forward | L × n × d² | 4 × 32 × 65.536 |
| **Total** | **L × (n² × d + n × d²)** | **≈ 9.4M operações** |

### Cálculo de FLOPs por Camada

| Componente | Operação | FLOPs |
|---------------------------|----------------------|------------------|
| Atenção (QKV) | 3 projeções lineares | 3 × n × d² |
| Scores de Atenção | Multiplicação QK^T | n² × d |
| Aplicação de Atenção | Multiplicação com V | n² × d |
| Projeção de Saída | Projeção linear | n × d² |
| Feed-Forward (1ª camada) | Linear expansion | n × d × 4d |
| Feed-Forward (2ª camada) | Linear reduction | n × 4d × d |
| **Total por Camada** | **Soma de todas** | **10n × d² + 2n² × d** |

### FLOPs do Modelo Completo

| Escopo | Cálculo | FLOPs |
|---------------------------|--------------------------------|------------------|
| Por Camada | 10n × d² + 2n² × d | ≈ 2.6M |
| Modelo (4 camadas) | L × (10n × d² + 2n² × d) | ≈ 10.5M |
| Por Token (forward) | FLOPs do modelo / batch | ≈ 10.5M |
| Por Batch (32 tokens) | Batch size × FLOPs por token | ≈ 10.7B |
| Por Época (2.848 batches) | Batches × FLOPs por batch | ≈ 30.5T |

### Parâmetros por Tipo de Camada

| Tipo de Camada | Quantidade | Parâmetros por Camada | Total |
|---------------------------|------------|----------------------|--------------|
| Token Embedding | 1 | 12.877.824 | 12.877.824 |
| Positional Embedding | 1 | 65.536 | 65.536 |
| Transformer Block | 4 | ≈ 540.000 | ≈ 2.160.000 |
| **Total** | **-** | **-** | **15.103.616** |

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
- 

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
