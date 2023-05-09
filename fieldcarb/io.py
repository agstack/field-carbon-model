'''
File interchange utilities, e.g., reading and writing parameter tables.
'''

import json
import numpy as np


def params_dict_from_json(file_path: str, **kwargs):
    '''
    Writes a parameter dictionary (e.g., as from
    `pyl4c.data.fixtures.restore_bplut_flat()`) to a JSON file, with proper
    handling of NumPy arrays.

    Parameters
    ----------
    file_path : str
        The input file path
    '''
    with open(file_path, 'r') as file:
        params = json.load(file)
    result = dict()
    for key, value in params.items():
        if hasattr(value, '__len__') and hasattr(value, 'sort'):
            result[key] = np.array(value)
        else:
            result[key] = value
    return result


def params_dict_to_json(params: dict, file_path: str, **kwargs):
    '''
    Writes a parameter dictionary (e.g., as from
    `pyl4c.data.fixtures.restore_bplut_flat()`) to a JSON file, with proper
    handling of NumPy arrays.

    Parameters
    ----------
    params : dict
        A dictionary of model parameters
    file_path : str
        The output file path
    **kwargs
        Additional keyword arguments to `json.dump()`
    '''
    result = dict()
    for key, value in params.items():
        if key == 'decay_rates':
            result[key] = [np.array(v).round(6).tolist() for v in value.tolist()]
        elif hasattr(value, 'tolist'):
            result[key] = [np.array(v).round(3).tolist() for v in value.tolist()]
        else:
            result[key] = value
    with open(file_path, 'w') as file:
        json.dump(result, file, **kwargs)
