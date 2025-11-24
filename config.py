import math

max_steps = 10000
max_lr = 3e-4
min_lr = max_lr * 0.2
warmup_steps = 100

def cosine_decay(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    if it >= max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine * (max_lr - min_lr)
