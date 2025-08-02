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

import config
from utils import save_plan_to_file, ModuleSourceTracer
from model_planner import ModelPlanner
from code_mutator import CodeMutator

warnings.filterwarnings("ignore")

# --- ORIGINAL MODEL DEFINITION ---
custom_net_code_string = """
import torch
import torch.nn as nn
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += identity; out = self.relu(out)
        return out
class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super(Net, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x
"""

# --- NEW MODEL DEFINITION (ADAPTED FOR OUR FRAMEWORK) ---
alexnet_style_code_string = """
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        in_channels = 3
        dropout_p = 0.5
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
"""

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

        constructor = load_constructor_from_string_via_file(model_source)
        
        with ModuleSourceTracer(model_source) as tracer:
            sig = inspect.signature(constructor)
            if "weights" in sig.parameters:
                original_model = constructor(weights=None)
            elif "num_classes" in sig.parameters:
                 original_model = constructor(num_classes=1000)
            else:
                original_model = constructor()
        source_map = tracer.create_source_map(original_model)

        planner = ModelPlanner(original_model, source_map=source_map, search_depth=config.PRODUCER_SEARCH_DEPTH)
        plan = planner.plan_random_mutation()
        
        if not plan:
            save_plan_to_file(model_name, 'skipped_no_plan', {}, {"reason": "Planner could not find a valid mutation."})
            return 'skipped_no_plan', None
        
        mutated_model = planner.apply_plan()
        mutated_graph_module = fx.symbolic_trace(mutated_model); mutated_graph_module.recompile()
        mutated_graph_module(torch.randn(2, 3, 224, 224))
        checksum = ModelPlanner.get_model_checksum(mutated_graph_module)

        code_mutator = CodeMutator(model_source)
        
        for full_module_name, details in plan.items():
            location = details.get("source_location")
            if not location:
                continue

            module = original_model.get_submodule(full_module_name)
            
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

        modified_code = code_mutator.get_modified_code()
        
        output_dir = os.path.join(config.PLANS_OUTPUT_DIR, model_name, "mutated_code")
        os.makedirs(output_dir, exist_ok=True)
        modified_code_path = os.path.join(output_dir, f"mutated_{checksum}.py")
        with open(modified_code_path, 'w') as f: f.write(modified_code)
        
        details = {"checksum": checksum, "static_code_path": modified_code_path}
        save_plan_to_file(model_name, 'success', plan, details)
        return 'success', details

    except Exception as e:
        error_str = f"{repr(e)}\n{traceback.format_exc()}"
        status = 'fail_worker_uncaught_error'
        if 'original_model' not in locals() or not original_model: status = 'fail_instantiation'
        elif not plan: status = 'fail_planning'
        save_plan_to_file(model_name, status, plan, {"error": error_str})
        return status, None

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(config.PLANS_OUTPUT_DIR, exist_ok=True)
    print(f"Mutation plans will be saved in '{config.PLANS_OUTPUT_DIR}/' directory.")
    
    # ADD THE NEW MODEL TO THE DICTIONARY OF SOURCES TO BE TESTED
    model_sources = { 
        "CustomNetFromString": custom_net_code_string,
        "AlexNetStyleFromString": alexnet_style_code_string
    }

    final_report = {}
    for model_name, source in model_sources.items():
        num_processes = min(config.NUM_WORKERS, config.MAX_CORES_TO_USE)
        print("\n" + "#" * 80); print(f"# Stress Testing Model: {model_name} with {num_processes} workers"); print("#" * 80)
        worker_args = [(model_name, source)] * config.NUM_ATTEMPTS_PER_MODEL
        stats = Counter(); generated_checksums = set()
        start_time = time.time()
        
        if config.DEBUG_MODE:
             print("!!! DEBUG MODE ON: Running sequentially. !!!")
             results = [run_single_mutation(arg) for arg in tqdm(worker_args, desc=f"Mutating {model_name}")]
        else:
            with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
                results = list(tqdm(pool.imap_unordered(run_single_mutation, worker_args), total=len(worker_args), desc=f"Mutating {model_name}"))

        for status, data in results:
            if status == 'success' and data:
                checksum = data['checksum']
                if checksum not in generated_checksums:
                    generated_checksums.add(checksum)
                    stats['success_unique'] += 1
                    print(f"\n  -> New unique architecture generated: {data['static_code_path']}")
                else:
                    stats['duplicate_outcome'] += 1
            else:
                stats[status] += 1
        
        end_time = time.time()
        total = sum(stats.values())
        success_rate = ((stats['success_unique'] + stats['duplicate_outcome']) / total * 100) if total > 0 else 0
        print(f"\n{model_name} Test Complete in {end_time - start_time:.2f} seconds.")
        print(f"  - SUCCESS (Total Valid):            {stats['success_unique'] + stats['duplicate_outcome']} ({success_rate:.2f}%)")
        print(f"  -   Unique Architectures:         {stats['success_unique']}")
        print(f"  -   Duplicate Architectures:      {stats['duplicate_outcome']}")
        print(f"  - FAILED (Init/Trace/Worker Error): {stats.get('fail_worker_uncaught_error', 0) + stats.get('fail_instantiation', 0) + stats.get('fail_planning', 0)}")
        print(f"  - SKIPPED (No valid plan found):    {stats.get('skipped_no_plan', 0)}")