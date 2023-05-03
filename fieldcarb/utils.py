'''
'''

import numpy as np
from numbers import Number
from typing import Callable

def linear_constraint(
        xmin: Number, xmax: Number, form: str = None
    ) -> Callable:
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    xmin : int or float
        Lower bound of the linear ramp function
    xmax : int or float
        Upper bound of the linear ramp function
    form : str
        Type of ramp function: "reversed" decreases as x increases;
        "binary" returns xmax when x == 1; default (None) is increasing
        as x increases.

    Returns
    -------
    function
    '''
    assert form == 'binary' or np.any(xmax >= xmin),\
        'xmax must be greater than/ equal to xmin'
    if form == 'reversed':
        return lambda x: np.where(x >= xmax, 0,
            np.where(x < xmin, 1, 1 - np.divide(
                x - xmin, xmax - xmin)))
    if form == 'binary':
        return lambda x: np.where(x == 1, xmax, xmin)
    return lambda x: np.where(x >= xmax, 1,
        np.where(x < xmin, 0, np.divide(x - xmin, xmax - xmin)))
