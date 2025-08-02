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

# --- HELPER FUNCTION MUTATION CONTROL ---
# Controls whether helper function calls (like conv3x3()) should be mutation targets
# True: Mutate helper function calls (current behavior) - allows indirect mutations
# False: Only mutate direct nn.Module instantiations - more semantically correct
ALLOW_HELPER_FUNCTION_MUTATIONS = False

# Helper function patterns to detect (used when ALLOW_HELPER_FUNCTION_MUTATIONS = False)
HELPER_FUNCTION_PATTERNS = [
    'conv1x1', 'conv3x3', 'conv5x5', 'conv7x7',      # Convolution helpers
    'make_layer', 'make_block', 'make_stage',          # Layer builders  
    'build_', 'create_', 'get_',                       # Factory functions
    'downsample', 'upsample',                          # Sampling helpers
]

# --- CONVNEXT COMPATIBILITY SETTINGS ---
# Modules that are problematic for torch.fx symbolic tracing
FX_INCOMPATIBLE_MODULES = [
    'StochasticDepth', 'LayerNorm2d', 'Permute'
]

# Alternative modules for FX-incompatible ones during mutation
FX_COMPATIBLE_REPLACEMENTS = {
    'StochasticDepth': 'Dropout',           # Replace with standard Dropout
    'LayerNorm2d': 'BatchNorm2d',          # Replace with BatchNorm2d
    'Permute': 'Identity',                 # Replace with Identity (no-op)
}

# ConvNeXT-specific mutation patterns
CONVNEXT_MUTATIONS = {
    # Depth-wise convolution mutations
    'depthwise_conv': {
        'kernel_sizes': [3, 5, 7],         # Alternative kernel sizes
        'group_ratios': [1, 0.5, 1.0],     # groups=1 (standard), groups=dim//2, groups=dim
    },
    # MLP expansion ratios
    'mlp_expansion': [2, 4, 6, 8],         # Alternative expansion ratios
}