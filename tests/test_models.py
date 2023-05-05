'''
Test suite for core models.
'''

import numpy as np
import pytest
from fieldcarb.core import TCF

CEREAL_PARAMETERS = {
    'LUE': 1.61, 'tmin0': 257.3, 'tmin1': 285.9, 'vpd0': 150, 'vpd1': 4000,
    'smrz0': 0.1, 'smrz1': 0.3, 'smsf0': 0.0, 'smsf1': 0.25, 'ft0': 0.78
}

def random_tcf_data_cube(n_pixels, t_steps, seed = 406):
    '''
    Generates a random, synthetic data cube for running the TCF model.
    '''
    np.random.seed(seed)
    soc = np.random.randint(0, 5000, n_pixels)
    size = n_pixels * t_steps
    # Assumed Gaussian distributions based on mean, std. deviation of
    #   SMAP Level 4 Carbon, Version 7 inputs
    fpar = np.random.normal(0.46, 0.24, size)
    fpar = np.where(np.logical_or(fpar < 0, fpar > 1), 0, fpar)
    par = np.random.normal(6.8, 3.9, size)
    par = np.where(par < 0, 0, par)
    tmin = np.random.normal(278.5, 12.3, size)
    vpd = np.random.lognormal(5.4, 1.4, size)
    smrz = np.random.normal(0.77, 0.15, size)
    smrz = np.where(np.logical_or(smrz < 0, smrz > 1), 0, smrz)
    ft = np.random.choice((0, 1), size)
    tsoil = np.random.normal(284, 10, size)
    smsf = np.random.normal(0.50, 0.21, size)
    smsf = np.where(np.logical_or(smsf < 0, smsf > 1), 0, smsf)
    drivers = np.stack([
        fpar, par, tmin, vpd, smrz, ft, tsoil, smsf
    ], axis = 0).reshape((8, n_pixels, t_steps))
    return soc, drivers


def test_tcf_gpp_values():
    '''
    Test that the TCF model's GPP calculation is consistent.
    '''
    soc_state, drivers = random_tcf_data_cube(100, 100, seed = 406)
    tcf = TCF(CEREAL_PARAMETERS, 7, soc_state)
    gpp = tcf.gpp(drivers[0:6]).round(2)
    assert np.median(gpp) == 1.63
    assert np.var(gpp).round(2) == 7.99
    assert gpp.min() == 0.0
    assert gpp.max() == 20.75
