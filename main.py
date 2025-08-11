import torch
import torch.nn as nn
import torch.fx as fx
import multiprocessing
from collections import Counter
from tqdm import tqdm
import time
import os
import sys
import inspect
import warnings
import importlib
import traceback
import pandas as pd
import hashlib

# Import LEMUR API
import ab.nn.api as nn_dataset
from ab.nn.util.Util import uuid4  # use canonical hashing

import config
from utils import save_plan_to_file, ModuleSourceTracer
from model_planner import ModelPlanner
from code_mutator import CodeMutator

warnings.filterwarnings("ignore")

# LEMUR API utilities
def fetch_model_from_lemur(model_name):
    """Fetch model code from LEMUR database"""
    try:
        # Fetch model data from nn-dataset
        data = nn_dataset.data(only_best_accuracy=True)
        
        # Filter for specific model
        model_data = data[data['nn'] == model_name]
        
        if not model_data.empty:
            return model_data.iloc[0]['nn_code']
        return None
    except Exception as e:
        print(f"Error fetching model {model_name}: {str(e)}")
        return None

# Default parameters for LEMUR models
DEFAULT_IN_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)
DEFAULT_OUT_SHAPE = (1000,)  # Output shape for classification
DEFAULT_PRM = {
    'lr': 0.01,
    'momentum': 0.9,
    'dropout': 0.5
}

# Common safe defaults to fill supported hyperparameters when nn-dataset prm is unavailable
COMMON_PRM_DEFAULTS = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'dropout': 0.5,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'alpha': 0.99,
}

def load_lemur_model(model_source: str):
    """Load LEMUR model with proper parameters.
    - Prefer prm from LEMUR dataset for this exact code (best-accuracy row).
    - Filter prm to the model's supported_hyperparameters() if provided.
    - Fallback to safe defaults for any supported keys not present.
    """
    constructor = load_constructor_from_string_via_file(model_source)
    
    # LEMUR models require specific parameters
    sig = inspect.signature(constructor)
    params = {}

    # Discover the module that owns the constructor to query supported_hyperparameters()
    supported_params = None
    try:
        model_module = importlib.import_module(constructor.__module__)
        if hasattr(model_module, 'supported_hyperparameters'):
            maybe_supported = model_module.supported_hyperparameters()
            if isinstance(maybe_supported, (set, list, tuple)):
                supported_params = set(maybe_supported)
        # Also check class-level attribute for completeness
        if supported_params is None and hasattr(constructor, 'supported_hyperparameters'):
            maybe_supported = constructor.supported_hyperparameters()
            if isinstance(maybe_supported, (set, list, tuple)):
                supported_params = set(maybe_supported)
    except Exception:
        supported_params = None

    # Try to get prm from nn-dataset for this exact code
    dataset_prm = None
    try:
        df = nn_dataset.data(only_best_accuracy=False)
        rows = df[df['nn_code'] == model_source]
        if not rows.empty:
            best_row = rows.loc[rows['accuracy'].idxmax()]
            dataset_prm = best_row.get('prm', None)
    except Exception:
        dataset_prm = None
    
    if "in_shape" in sig.parameters:
        params["in_shape"] = DEFAULT_IN_SHAPE
    if "out_shape" in sig.parameters:
        params["out_shape"] = DEFAULT_OUT_SHAPE
    if "prm" in sig.parameters:
        # Start from dataset prm when available
        if isinstance(dataset_prm, dict) and dataset_prm:
            if supported_params:
                prm = {k: v for k, v in dataset_prm.items() if k in supported_params}
                # Fill any missing supported keys with safe defaults
                for k in supported_params:
                    if k not in prm and (k in COMMON_PRM_DEFAULTS or k in DEFAULT_PRM):
                        prm[k] = COMMON_PRM_DEFAULTS.get(k, DEFAULT_PRM.get(k))
            else:
                prm = dataset_prm
        else:
            # Build from supported set if present, otherwise fall back to legacy DEFAULT_PRM
            if supported_params:
                prm = {}
                for k in supported_params:
                    if k in COMMON_PRM_DEFAULTS:
                        prm[k] = COMMON_PRM_DEFAULTS[k]
                    elif k in DEFAULT_PRM:
                        prm[k] = DEFAULT_PRM[k]
                # If nothing matched, don't pass empty dict; use DEFAULT_PRM
                if not prm:
                    prm = DEFAULT_PRM
            else:
                prm = DEFAULT_PRM
        params["prm"] = prm
    if "device" in sig.parameters:
        params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return constructor(**params)

def load_constructor_from_string_via_file(code_string: str):
    pid = os.getpid(); timestamp = time.time_ns()
    temp_module_name = f"_temp_model_{pid}_{timestamp}"
    temp_module_path = f"{temp_module_name}.py"
    with open(temp_module_path, "w") as f: f.write(code_string)
    def cleanup():
        if os.path.exists(temp_module_path):
            try: os.remove(temp_module_path)
            except OSError: pass
        if temp_module_name in sys.modules:
            try: del sys.modules[temp_module_name]
            except KeyError: pass
    import atexit; atexit.register(cleanup)
    module = importlib.import_module(temp_module_name)
    return getattr(module, 'Net')

def run_single_mutation(worker_args):
    model_name, model_source = worker_args
    plan, original_model = {}, None
    try:
        if not isinstance(model_source, str):
            return 'skipped_not_string', None

        # Use LEMUR loader with specialized parameters
        with ModuleSourceTracer(model_source) as tracer:
            original_model = load_lemur_model(model_source)
        source_map = tracer.create_source_map(original_model)

        planner = ModelPlanner(original_model, source_map=source_map, search_depth=config.PRODUCER_SEARCH_DEPTH)
        plan = planner.plan_random_mutation()
        
        if not plan:
            save_plan_to_file(model_name, 'skipped_no_plan', {}, {"reason": "Planner could not find a valid mutation."})
            return 'skipped_no_plan', None
        
        mutated_model = planner.apply_plan()
        
        # Enhanced model verification
        try:
            # Test with intended input resolution only
            h, w = DEFAULT_IN_SHAPE[2:]
            test_input = torch.randn(2, 3, h, w)
            
            # Forward pass
            mutated_model.eval()
            with torch.no_grad():
                output = mutated_model(test_input)
            
            # Check output shape
            if output.shape[0] != 2 or output.shape[1] != DEFAULT_OUT_SHAPE[0]:
                raise RuntimeError(f"Output shape {output.shape} not as expected for input {h}x{w}")
            
            # Backward pass test
            mutated_model.train()
            test_input.requires_grad = True
            output = mutated_model(test_input)
            loss = output.sum()
            loss.backward()
            
            # Check gradients
            for name, param in mutated_model.named_parameters():
                if param.grad is None:
                    raise RuntimeError(f"No gradient for parameter {name}")
            
            # Simple learning capability check
            optimizer = torch.optim.SGD(mutated_model.parameters(), lr=0.01)
            mutated_model.train()
            test_input = torch.randn(2, 3, 224, 224)
            output = mutated_model(test_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {str(e)}")
        
        code_mutator = CodeMutator(model_source)
        
        for full_module_name, details in plan.items():
            location = details.get("source_location")
            if not location:
                continue

            mutation_type = details.get("mutation_type", "dimension")
            module = original_model.get_submodule(full_module_name)
            
            if mutation_type == "dimension":
                if details.get('new_out') is not None:
                    arg_to_change = None
                    if isinstance(module, nn.Linear): arg_to_change = 'out_features'
                    elif isinstance(module, nn.Conv2d): arg_to_change = 'out_channels'
                    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)): arg_to_change = 'num_features'
                    
                    if arg_to_change:
                        code_mutator.schedule_modification(location, arg_to_change, details['new_out'])
                
                if details.get('new_in') is not None:
                    arg_to_change = None
                    if isinstance(module, nn.Linear): arg_to_change = 'in_features'
                    elif isinstance(module, nn.Conv2d): arg_to_change = 'in_channels'
                    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)): arg_to_change = 'num_features'

                    if arg_to_change:
                        code_mutator.schedule_modification(location, arg_to_change, details['new_in'])
            
            elif mutation_type == "activation":
                new_activation = details.get('new_activation')
                if new_activation:
                    code_mutator.schedule_activation_modification(location, new_activation)
            
            elif mutation_type == "layer_type":
                new_layer_type = details.get('new_layer_type')
                mutation_params = details.get('mutation_params', {})
                if new_layer_type:
                    code_mutator.schedule_layer_type_modification(location, new_layer_type, mutation_params)
            
            elif mutation_type == "kernel_size":
                new_kernel_size = details.get('new_kernel_size')
                if new_kernel_size:
                    code_mutator.schedule_kernel_size_modification(location, new_kernel_size)

            elif mutation_type == "stride":
                new_stride = details.get('new_stride')
                if new_stride:
                    code_mutator.schedule_stride_modification(location, new_stride)

        modified_code = code_mutator.get_modified_code()
        
        # Save to nn-dataset repository
        # Use nn-dataset hashing (whitespace-insensitive) for consistency
        checksum = uuid4(modified_code)
        # Use configurable output root from config
        model_dir = os.path.join(config.MUTATED_MODELS_OUTPUT_ROOT, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"mutated_{checksum}.py")
        
        with open(model_path, 'w') as f:
            f.write(modified_code)
        
        # Optionally save mutated model to LEMUR DB (nn-dataset)
        if getattr(config, 'SAVE_MUTATED_TO_DB', False):
            try:
                # Build DB params: start from discovered/supported prm if possible
                db_prm = {}
                # Merge required training params from config
                db_prm.update(getattr(config, 'DB_TRAIN_PRM', {}))
                # Ensure minimal required keys exist
                required_keys = {'batch', 'epoch', 'transform'}
                missing = [k for k in required_keys if k not in db_prm]
                if missing:
                    raise ValueError(f"Missing required DB training params: {missing}. Set config.DB_TRAIN_PRM.")
                # Call nn-dataset to train-eval and save
                model_name, acc, a2t, code_score = nn_dataset.check_nn(
                    modified_code,
                    task=getattr(config, 'DB_TASK', None),
                    dataset=getattr(config, 'DB_DATASET', None),
                    metric=getattr(config, 'DB_METRIC', None),
                    prm=db_prm,
                    save_to_db=True,
                    prefix=getattr(config, 'DB_MODEL_PREFIX', 'mutated'),
                    save_path=None,
                    export_onnx=False,
                    epoch_limit_minutes=getattr(config, 'DB_EPOCH_LIMIT_MINUTES', 10)
                )
                if config.DEBUG_MODE:
                    print(f"  -> Saved to DB as: {model_name} (acc={acc:.4f})")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"[WARN] Failed to save mutated model to DB: {e}")
        
        return 'success', {"path": model_path, "checksum": checksum}

    except Exception as e:
        error_str = f"{repr(e)}\n{traceback.format_exc()}"
        status = 'fail_worker_uncaught_error'
        if 'original_model' not in locals() or not original_model: status = 'fail_instantiation'
        elif not plan: status = 'fail_planning'
        save_plan_to_file(model_name, status, plan, {"error": error_str})
        return status, None

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Starting LEMUR-based model mutation system")
    
    try:
        # Fetch ALL models from LEMUR database
        data = nn_dataset.data(only_best_accuracy=False)
        all_model_names = data['nn'].unique().tolist()
        print(f"Found {len(all_model_names)} total models in LEMUR database")
        
        # Filter to only BASE models (exclude generated variants with UUIDs)
        import re
        uuid_pattern = re.compile(r'-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        base_model_names = [name for name in all_model_names if not uuid_pattern.search(name)]
        print(f"Filtered to {len(base_model_names)} base models (excluding UUID variants)")
        
        # Further filter models based on config if specified
        if config.SPECIFIC_MODELS:
            model_names = [name for name in base_model_names if name in config.SPECIFIC_MODELS]
            print(f"Further filtered to {len(model_names)} specific models from config")
        else:
            model_names = base_model_names
        
        model_sources = {}
        empty_models = []
        
        with tqdm(model_names, desc="Fetching models") as pbar:
            for name in pbar:
                model_data = data[data['nn'] == name]
                if not model_data.empty:
                    # Get the best accuracy version
                    best_model = model_data.loc[model_data['accuracy'].idxmax()]
                    code = best_model['nn_code']
                    
                    if code and code.strip():
                        model_sources[name] = code
                        pbar.set_postfix_str(f"✓ {name}")
                    else:
                        empty_models.append(name)
                        pbar.set_postfix_str(f"⚠ Empty: {name}")
                else:
                    empty_models.append(name)
                    pbar.set_postfix_str(f"⚠ No data: {name}")
        
        print(f"\nSuccessfully fetched {len(model_sources)}/{len(model_names)} models from LEMUR")
        if empty_models:
            print(f"Models with issues: {', '.join(empty_models)}")
        
        if not model_sources:
            print("Error: No models were fetched from LEMUR. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error fetching models from LEMUR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    # Add summary header
    print("\n" + "=" * 80)
    print(f"STARTING MUTATION PROCESS: {len(model_sources)} MODELS")
    print("=" * 80 + "\n")
    
    # Initialize summary statistics
    total_success = 0
    total_failed = 0
    
    with tqdm(model_sources.items(), desc="Mutating models", total=len(model_sources)) as model_pbar:
        for model_name, source in model_pbar:
            model_pbar.set_postfix_str(f"Model: {model_name}")
            num_processes = min(config.NUM_WORKERS, config.MAX_CORES_TO_USE)
            
            if config.DEBUG_MODE:
                print("\n" + "#" * 80)
                print(f"# Mutating Model: {model_name} with {num_processes} workers")
                print("#" * 80)
            
            worker_args = [(model_name, source)] * config.NUM_ATTEMPTS_PER_MODEL
            stats = Counter()
            start_time = time.time()
            
            if config.DEBUG_MODE:
                results = [run_single_mutation(arg) for arg in tqdm(worker_args, desc=f"Mutating {model_name}")]
            else:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = list(tqdm(pool.imap_unordered(run_single_mutation, worker_args), 
                                    total=len(worker_args), 
                                    desc=f"Mutating {model_name}"))

            for status, data in results:
                stats[status] += 1
                if status == 'success':
                    # Only print path in debug mode to reduce clutter
                    if config.DEBUG_MODE:
                        print(f"  -> Mutated model saved to: {data['path']}")

            end_time = time.time()
            total = sum(stats.values())
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            
            # Print concise one-line summary
            failed = stats.get('fail_worker_uncaught_error',0) + stats.get('fail_instantiation',0) + stats.get('fail_planning',0)
            print(f"{model_name}: {stats['success']}✓ {failed}✗ in {end_time - start_time:.2f}s")
            
            # Update model progress bar
            model_pbar.set_postfix_str(f"{model_name}: ✓{stats['success']} ✗{failed}")
            
            # Update summary statistics
            total_success += stats['success']
            total_failed += failed

    # Print final summary
    print("\n" + "=" * 80)
    print(f"TOTAL MUTATION SUMMARY: {total_success} successful mutations, {total_failed} failed mutations")
    print("=" * 80)
