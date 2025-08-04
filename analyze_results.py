import os
import json

results = {}
mutation_types = {'activation': 0, 'dimension': 0, 'layer_type': 0, 'architectural': 0}
total_success = 0

for model_dir in ['CustomNetFromString', 'AlexNetStyleFromString', 'ConvNeXTStyleFromString', 'SimpleConvNeXTFromString']:
    path = f'mutation_plans/{model_dir}'
    if os.path.exists(path):
        successes = [f for f in os.listdir(path) if f.startswith('success_')]
        failures = [f for f in os.listdir(path) if f.startswith('fail_')]
        
        model_mutations = {'activation': 0, 'dimension': 0, 'layer_type': 0, 'architectural': 0}
        
        for success_file in successes:
            with open(f'{path}/{success_file}', 'r') as f:
                data = json.load(f)
                for mutation in data.get('plan', {}).values():
                    mut_type = mutation.get('mutation_type', 'unknown')
                    model_mutations[mut_type] = model_mutations.get(mut_type, 0) + 1
                    mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
        
        results[model_dir] = {
            'successes': len(successes),
            'failures': len(failures), 
            'mutations': model_mutations
        }
        total_success += len(successes)

print('=== COMPLETE MUTATION SYSTEM RESULTS ===')
for model, data in results.items():
    print(f'{model}:')
    print(f'  Successes: {data["successes"]}/30 ({data["successes"]*100/30:.1f}%)')
    print(f'  Failures: {data["failures"]}/30')
    print(f'  Mutations by type: {data["mutations"]}')
    print()

print(f'TOTAL SUCCESS RATE: {total_success}/120 ({total_success*100/120:.1f}%)')
print(f'MUTATION TYPE DISTRIBUTION: {mutation_types}')
