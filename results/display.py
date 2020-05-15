import json
import numpy as np

file = 'result_v1.json'
with open(file, 'r') as f:
    data = json.load(f)

for result in data:
    print('Parameters:\n{}'.format(result[0]))
    print('Results are:')
    print('\tStep: {}'.format(len(result[1]['time_list'])))
    print('\tLoss: {:.5f}'.format(np.mean(result[1]['loss_list'][-5:])))
    print('\tScore: {:.5f}'.format(np.mean(result[1]['accuracy_list'][-5:])))
    print('\tTime: {:.5f}'.format(result[1]['time_list'][-1]))
    print('----------------------------------------------------')
    print('----------------------------------------------------')
