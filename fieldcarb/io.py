'''
File interchange utilities, e.g., reading and writing parameter tables.
'''

import csv
import datetime
import json
import numpy as np
from typing import Sequence


def drivers_from_csv(file_path: str, fields_diff: Sequence = None):
    '''
    For a single-site time series, read driver data in from a CSV file. Each
    row should be a time step for a single site.

    Parameters
    ----------
    file_path : str
        The input file path
    fields_diff : Sequence
        Do not use

    Returns
    -------
    tuple
        A 2-element tuple of `(drivers, dates)`
    '''
    expected = ['fpar', 'swrad', 'tmean', 'qv2m', 'ps', 'tmin', 'smrz', 'tsoil', 'smsf']
    # To facilitate testing and some edge cases, allow (advanced) user to
    #   specify which fields might not be available
    fields = list(expected)
    if fields_diff is not None:
        fields = list(set(expected).difference(fields_diff))
    header = ('', 'date', *fields)
    drivers = []
    dates = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            if reader.line_num == 1:
                assert all([(col in header) for col in line])
                continue
            record = []
            if 'date' in line.keys():
                # Try to figure out the date-time format
                if len(dates) == 0:
                    for format in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
                        try:
                            datetime.datetime.strptime(line['date'], format)
                            break
                        except ValueError:
                            continue
                dates.append(datetime.datetime.strptime(line['date'], format))
            for key in expected:
                if key in line.keys():
                    record.append(float(line[key]))
                else:
                    record.append(np.nan)
            drivers.append(record)
    return (np.array(drivers).swapaxes(0, 1), np.array(dates))


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
