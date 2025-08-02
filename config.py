import multiprocessing

DEBUG_MODE = True
NUM_ATTEMPTS_PER_MODEL = 10
PRODUCER_SEARCH_DEPTH = 10
PLANS_OUTPUT_DIR = "mutation_plans"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_CORES_TO_USE = 12
VALID_CHANNEL_SIZES = [16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 512, 640, 768, 1024]

# --- MUTATION TYPE CONFIGURATIONS ---
# Activation function mutation mappings
ACTIVATION_MUTATIONS = {
    'ReLU': ['GELU', 'ELU', 'LeakyReLU', 'SiLU'],  # Using SiLU instead of Swish
    'GELU': ['ReLU', 'ELU', 'SiLU'],
    'ELU': ['ReLU', 'GELU', 'LeakyReLU', 'SiLU'],
    'LeakyReLU': ['ReLU', 'GELU', 'ELU', 'SiLU'],
    'SiLU': ['ReLU', 'GELU', 'ELU'],  # SiLU is PyTorch's Swish
    'Tanh': ['ReLU', 'GELU', 'SiLU'],
    'Sigmoid': ['ReLU', 'GELU', 'Tanh']
}

# Layer type mutation mappings  
LAYER_TYPE_MUTATIONS = {
    'BatchNorm2d': ['GroupNorm', 'LayerNorm', 'InstanceNorm2d'],
    'GroupNorm': ['BatchNorm2d', 'LayerNorm', 'InstanceNorm2d'],
    'LayerNorm': ['BatchNorm2d', 'GroupNorm'],
    'InstanceNorm2d': ['BatchNorm2d', 'GroupNorm'],
    'MaxPool2d': ['AvgPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d'],
    'AvgPool2d': ['MaxPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d'],
}

# Mutation type weights (probability distribution)
MUTATION_TYPE_WEIGHTS = {
    'dimension': 0.4,      # 40% - existing dimension mutations
    'activation': 0.4,     # 40% - activation function mutations  
    'layer_type': 0.2      # 20% - layer type mutations
}