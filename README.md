# Neural Network Mutation Framework

A sophisticated framework for automatically generating architectural variants of neural networks through intelligent mutations of PyTorch models.

## 🚀 Features

- **Automated Channel Mutations**: Intelligently modifies input/output channels while maintaining model validity
- **Source Code Transformation**: Directly mutates Python source code using AST manipulation
- **Helper Function Protection**: Preserves helper functions while targeting their call sites
- **Coordinated Mutations**: Ensures all dependent layers are updated consistently
- **High Success Rate**: Generates architecturally valid models with 70-90% success rates

## 🏗️ Architecture

The framework consists of several key components:

### Core Components

1. **ModelPlanner** (`model_planner.py`)
   - Analyzes model graphs using `torch.fx`
   - Plans coordinated mutations across multiple layers
   - Groups related layers for simultaneous updates

2. **ModuleSourceTracer** (`utils.py`)
   - Maps runtime modules to their source code locations
   - Uses stack inspection to identify instantiation sites
   - Creates source maps for precise code targeting

3. **CodeMutator** (`code_mutator.py`)
   - Performs AST-based source code transformations
   - Handles parameter aliasing (e.g., `in_planes` vs `in_channels`)
   - Protects helper functions from direct modification

4. **Main Orchestrator** (`main.py`)
   - Coordinates the entire mutation pipeline
   - Supports multiprocessing for parallel mutations
   - Validates generated models through execution

## 📁 Project Structure

```
mutator/
├── ab/gpt/
│   ├── main.py              # Main execution script
│   ├── model_planner.py     # Mutation planning logic
│   ├── code_mutator.py      # AST-based code transformation
│   ├── utils.py             # Source tracing utilities
│   └── config.py            # Configuration parameters
└── mutation_plans/          # Generated mutation outputs (gitignored)
    ├── CustomNetFromString/
    └── AlexNetStyleFromString/
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mutator
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv mutator_env
   mutator_env\Scripts\activate  # Windows
   # source mutator_env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision tqdm numpy
   ```

## 🚀 Usage

### Basic Usage

```bash
cd ab/gpt
python main.py
```

This will:
- Generate 30 mutation attempts per model
- Create unique architectural variants
- Save results in `mutation_plans/` directory

### Configuration

Modify `config.py` to customize:

```python
NUM_ATTEMPTS_PER_MODEL = 30  # Number of mutations per model
VALID_CHANNEL_SIZES = [16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 512, 640, 768, 1024]
DEBUG_MODE = False  # Enable for detailed logging
```

## 📊 Example Results

The framework typically achieves:
- **Success Rate**: 70-90% valid architectural mutations
- **Diversity**: 100% unique architectures (no duplicates)
- **Speed**: ~15-20 seconds for 60 total mutations

### Sample Mutations

**Original:**
```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = nn.BatchNorm2d(64)
```

**Mutated:**
```python
self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = nn.BatchNorm2d(128)
```

## 🔧 Technical Details

### Mutation Strategy

1. **Graph Analysis**: Uses `torch.fx` to create computational graphs
2. **Dependency Tracking**: Identifies producer-consumer relationships
3. **Coordinated Updates**: Ensures dimensional compatibility across layers
4. **Source Mapping**: Maps modules to exact source code locations

### Helper Function Protection

The framework intelligently distinguishes between:
- **Helper Function Definitions**: `def conv3x3(in_planes, out_planes):`
- **Call Sites**: `self.conv1 = conv3x3(inplanes, planes, stride)`

Only call sites are modified, preserving helper function integrity.

## 🐛 Known Issues

- Helper function protection occasionally needs restoration
- Limited to Conv2d, Linear, BatchNorm2d, and LayerNorm modules
- Requires models to be symbolically traceable by torch.fx

## 🛣️ Roadmap

- [ ] Support for more module types (GroupNorm, MultiheadAttention)
- [ ] Kernel size and activation function mutations
- [ ] Performance-guided mutation selection
- [ ] Architecture-aware mutation strategies

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Contact

[Add contact information]
