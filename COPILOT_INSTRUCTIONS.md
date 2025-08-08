# GitHub Copilot Instructions for Neural Network Mutation Framework

## Project Overview

This is a sophisticated neural network mutation framework that automatically generates architectural variants of PyTorch models through intelligent source code transformations. The system uses AST manipulation to modify Python source code directly while maintaining model validity.

## Core Architecture

### Key Components

1. **ModelPlanner** (`model_planner.py`)
   - Uses `torch.fx` for symbolic tracing and graph analysis
   - Identifies mutation targets and dependency chains
   - Groups related layers for coordinated mutations
   - Currently supports channel size mutations only

2. **ModuleSourceTracer** (`utils.py`)
   - Maps runtime PyTorch modules to source code locations
   - Uses stack inspection and AST parsing
   - Creates precise source maps for code targeting
   - CRITICAL: Must distinguish helper function definitions from call sites

3. **CodeMutator** (`code_mutator.py`)
   - Performs AST-based source code transformations
   - Handles parameter aliasing (in_planes, in_channels, etc.)
   - IMPORTANT: Should never modify helper function definitions
   - Targets only module instantiation call sites

4. **Main Orchestrator** (`main.py`)
   - Coordinates the complete mutation pipeline
   - Supports multiprocessing for parallel execution
   - Validates mutations through model execution

## Critical Design Principles

### 🚨 Helper Function Protection
The system MUST preserve helper functions like:
```python
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
```

**NEVER MODIFY**: The function definition itself
**ALWAYS TARGET**: Call sites like `self.conv1 = conv3x3(inplanes, planes, stride)`

### 🎯 Source Location Accuracy
- ModuleSourceTracer must map modules to exact call sites
- Use stack inspection with smart walking to avoid helper functions
- Target only the final instantiation points, not intermediate returns

### 🔗 Coordinated Mutations
When changing layer dimensions:
1. Update the producer layer's output channels
2. Update all consumer layers' input channels  
3. Update propagator layers (BatchNorm, LayerNorm) accordingly
4. Maintain architectural validity throughout

## Code Patterns and Conventions

### Parameter Aliasing Support
Handle common parameter name variations:
```python
VARIABLE_ALIASES = {
    'in_channels': ['in_channels', 'in_planes', 'input_channels', 'c_in'],
    'out_channels': ['out_channels', 'out_planes', 'output_channels', 'c_out'],
    'in_features': ['in_features', 'input_features', 'features_in', 'in_dim'],
    'out_features': ['out_features', 'output_features', 'features_out', 'out_dim'],
    'num_features': ['num_features', 'features', 'channels']
}
```

### Stack Inspection Pattern
When tracing source locations, use careful stack walking:
```python
# Look for actual instantiation sites, not helper function returns
frame = inspect.stack()[1]  # Start with immediate caller
# Walk up stack intelligently to find the true call site
```

### AST Transformation Safety
Always include extensive context in oldString replacements:
```python
# GOOD: Include 3-5 lines of context
old_string = """        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)"""

# BAD: Too narrow, might match multiple locations
old_string = "nn.BatchNorm2d(planes)"
```

## Common Issues and Solutions

### Issue: Helper Functions Being Modified
**Symptom**: `conv3x3` definition shows hardcoded values like `nn.Conv2d(640, 640, ...)`
**Solution**: Enhance ModuleSourceTracer stack walking logic and CodeMutator filtering

### Issue: Source Location Mismatches  
**Symptom**: Mutations target wrong lines or fail to apply
**Solution**: Improve AST node matching and line number accuracy

### Issue: Dimensional Incompatibility
**Symptom**: Generated models fail to execute
**Solution**: Strengthen dependency tracking in ModelPlanner

## Testing and Validation

### Expected Success Rates
- **Channel Mutations**: 70-90% success rate
- **Generated Models**: Must execute without errors
- **Uniqueness**: 100% unique architectures (no duplicates)

### Test Models
The system includes two test models:
1. **CustomNetFromString**: ResNet-style with helper functions
2. **AlexNetStyleFromString**: Sequential CNN architecture

### Validation Pipeline
Each mutation must pass:
1. AST parsing and code generation
2. Model instantiation
3. Forward pass execution
4. Dimension compatibility checks

## Development Guidelines

### When Adding New Mutation Types
1. Extend `config.py` with new parameters
2. Add planning logic to `ModelPlanner`
3. Update `CodeMutator` with transformation rules
4. Modify `main.py` orchestration logic
5. Test extensively with existing models

### When Debugging Source Tracing
1. Enable `DEBUG_MODE = True` in config.py
2. Examine stack traces and source maps
3. Verify AST node line numbers match actual code
4. Check helper function detection logic

### Code Quality Standards
- Use type hints for all public methods
- Include comprehensive docstrings
- Handle edge cases gracefully with fallbacks
- Maintain backwards compatibility when possible

## Future Enhancements (Roadmap)

### Planned Mutation Types
1. **Kernel Size Mutations**: Change Conv2d kernel_size parameters
2. **Activation Swaps**: Replace ReLU with GELU, SiLU, Mish
3. **Normalization Swaps**: BatchNorm2d ↔ LayerNorm ↔ GroupNorm
4. **Topology Changes**: Add/remove skip connections, layer duplication

### Performance Improvements
1. **Performance-Guided Evolution**: Evaluate mutation quality
2. **Architecture-Aware Planning**: Understand ResNet, VGG, Transformer patterns
3. **Multi-Objective Optimization**: Balance accuracy, speed, size

## Debugging Checklist

When mutations aren't working correctly:

1. ✅ Check helper function detection logic
2. ✅ Verify source map accuracy  
3. ✅ Examine AST transformation rules
4. ✅ Test with DEBUG_MODE enabled
5. ✅ Validate model execution pipeline
6. ✅ Review multiprocessing coordination

## Configuration Reference

Key settings in `config.py`:
- `NUM_ATTEMPTS_PER_MODEL`: Mutations per model (default: 30)
- `VALID_CHANNEL_SIZES`: Available channel dimensions
- `DEBUG_MODE`: Detailed logging (set True for development)
- `NUM_WORKERS`: Parallel processing workers
- `PRODUCER_SEARCH_DEPTH`: Graph traversal depth limit

---

This framework represents sophisticated AI-assisted code transformation. When working with it, prioritize correctness over speed, and always validate that generated models execute successfully.
