from torch.optim import Adam, AdamW, SGD

OPTIMIZERS = {
    'adam': Adam,
    'adamw': AdamW,
    'sgd': SGD
}