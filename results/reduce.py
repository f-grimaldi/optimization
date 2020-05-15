"""
Get raw result with the list of weights  saved and return same object but with no list of weights
"""
import json

in_file = 'raw_result_v1.json'
out_file = 'result_v1.json'

with open(in_file, 'r') as f:
    data = json.load(f)

for model_results in data:
    model_results[1]['params_list'] = None

with open(out_file, 'w') as f:
    json.dump(data, f)
