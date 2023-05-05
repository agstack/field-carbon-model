'''
'''

import numpy as np
from numbers import Number
from typing import Callable

def arrhenius(
        tsoil: Number, beta0: float, beta1: float = 66.02,
        beta2: float = 227.13
    ) -> np.ndarray:
    r'''
    The Arrhenius equation for response of enzymes to (soil) temperature,
    constrained to lie on the closed interval [0, 1].

    $$
    f(T_{SOIL}) = \mathrm{exp}\left[\beta_0\left( \frac{1}{\beta_1} -
        \frac{1}{T_{SOIL} - \beta_2} \right) \right]
    $$

    Parameters
    ----------
    tsoil : numpy.ndarray
        Array of soil temperature in degrees K
    beta0 : float
        Coefficient for soil temperature (deg K)
    beta1 : float
        Coefficient for ... (deg K)
    beta2 : float
        Coefficient for ... (deg K)

    Returns
    -------
    numpy.ndarray
        Array of soil temperatures mapped through the Arrhenius function
    '''
    a = (1.0 / beta1)
    b = np.divide(1.0, np.subtract(tsoil, beta2))
    # This is the simple answer, but it takes on values >1
    y0 = np.exp(np.multiply(beta0, np.subtract(a, b)))
    # Constrain the output to the interval [0, 1]
    return np.where(y0 > 1, 1, np.where(y0 < 0, 0, y0))


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
