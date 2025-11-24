GPT-2 Training from Scratch
Implementation and training of a GPT-2 model with 15 million parameters, including detailed computational complexity analysis.

Overview
This project implements a GPT-2 architecture from scratch using PyTorch, focusing on understanding the computational requirements and optimization challenges of transformer-based language models. The implementation follows the original GPT-2 paper specifications with a reduced parameter count suitable for single-GPU training.

Model Architecture
The model uses a decoder-only Transformer architecture with the following specifications:

Component	Specification
Total Parameters	15,103,616
Transformer Layers	4
Embedding Dimension	256
Attention Heads	4
Context Length	256 tokens
Vocabulary Size	50,304
Dropout Rate	0.7
Activation Function	GELU
Architecture Components
Token Embedding: 50,304 × 256 = 12,877,824 parameters
Positional Embedding: 256 × 256 = 65,536 parameters
Transformer Blocks (4 layers):
Multi-head self-attention with causal masking
Position-wise feed-forward network (4× expansion)
Layer normalization
Residual connections
Training Configuration
Hardware
GPU: NVIDIA GeForce RTX 4050
Theoretical Performance: 70 TFLOPS (FP16)
VRAM: 6GB GDDR6
Hyperparameters
python batch_size = 32 sequence_length = 256 learning_rate = 3e-4 # AdamW optimizer weight_decay = 0.1 beta1 = 0.9 beta2 = 0.95 gradient_clip = 1.0

Dataset
Training Set: 4M tokens
Batches per Epoch: 2,848
Tokens per Epoch: 23,330,816
Computational Complexity Analysis
Time Complexity
For a Transformer model with L layers, dimension d, and sequence length n:

Self-Attention Layer: O(n² × d) - Query, Key, Value projections: O(n × d²) - Attention computation: O(n² × d) - Output projection: O(n × d²)

Feed-Forward Network: O(n × d²) - Two linear transformations with intermediate dimension 4d

Total per Layer: O(n² × d + n × d²)

Complete Model: O(L × (n² × d + n × d²))

FLOPs Calculation
Forward Pass: FLOPs = 6 × parameters × sequence_length FLOPs = 6 × 15,103,616 × 256 FLOPs = 23.2 GFLOPs per sequence

Training (Forward + Backward): FLOPs = 23.2 × 3 = 69.6 GFLOPs per sequence

Per Epoch: Sequences = 23,330,816 / 256 = 91,136 Total FLOPs = 69.6 × 10⁹ × 91,136 = 6.34 × 10¹⁵ FLOPs

Performance Metrics
Training Performance
Metric	Value
Tokens per Epoch	23,330,816
Training Time	37.5 min/epoch
Throughput	10,369 tokens/sec
Sequences/sec	40.5
Time per Batch	0.79 seconds
Effective FLOPs	2.82 TFLOPS
MFU (Model FLOPs Utilization)	4.03%
Loss Curves
Training Loss: - Initial: 13.0 - Final: 5.3 - Reduction: 59.2%

Validation Loss: - Initial: 13.0 - Final: 9.4 - Stabilization: ~9.0 after 40M tokens

Train-Validation Gap:
