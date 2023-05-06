'''
Test suite for core models.
'''

import datetime
import numpy as np
import pytest
from fieldcarb.models import TCF

CEREAL_PARAMETERS = {
    'LUE': 1.61, 'tmin0': 257.3, 'tmin1': 285.9, 'vpd0': 150, 'vpd1': 4000,
    'smrz0': 0.1, 'smrz1': 0.3, 'smsf0': 0.0, 'smsf1': 0.25, 'ft0': 0.78,
    'beta0': 242.47, 'decay_rates': [0.018, 0.0072, 0.000167], 'CUE': 0.708,
    'f_structural': 0.5, 'f_metabolic': 0.78
}

def random_tcf_data_cube(
        n_pixels, t_steps, seed = 406, seasonal_cycle = False):
    '''
    Generates a random, synthetic data cube for running the TCF model.
    '''
    np.random.seed(seed)
    soc = np.stack([
        np.random.randint(17, 50, n_pixels),
        np.random.randint(15, 72, n_pixels),
        np.random.randint(128, 6569, n_pixels),
    ], axis = 0)
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
    if seasonal_cycle and t_steps >= 365:
        cycle = np.sin((np.pi * np.arange(t_steps)) / 365)[None,:]\
            .repeat(n_pixels, axis = 0)
        fpar = (fpar.reshape((n_pixels, t_steps)) * cycle).ravel()
        tmin = 273.15 + (
            (tmin.reshape((n_pixels, t_steps)) - 273.15) * cycle).ravel()
        tsoil = 273.15 + (
            (tsoil.reshape((n_pixels, t_steps)) - 273.15) * cycle).ravel()
    drivers = np.stack([
        fpar, par, tmin, vpd, smrz, ft, tsoil, smsf
    ], axis = 0).reshape((8, n_pixels, t_steps))
    return soc, drivers


def test_tcf_forward_run_values():
    '''
    Test that the TCF model's forward run calculations are consistent.
    '''
    soc_state, drivers = random_tcf_data_cube(
        10, 365, seed = 406, seasonal_cycle = True)
    dates = [
        datetime.date(2023, 1, 1) + datetime.timedelta(days = d)
        for d in range(0, 365)
    ]
    tcf = TCF(CEREAL_PARAMETERS, 7, soc_state)
    nee, gpp, rh = tcf.forward_run(drivers, dates = dates, verbose = False)
    assert np.equal(
        np.percentile(nee, (1, 10, 50, 90, 99)).round(2),
        np.array([-5.71, -1.67, 0.33, 1.64, 3.05])).all()
    assert np.equal(
        np.percentile(gpp, (1, 10, 50, 90, 99)).round(2),
        np.array([0, 0, 0.79, 4.08, 9.81])).all()
    assert np.equal(
        np.percentile(rh, (1, 10, 50, 90, 99)).round(2),
        np.array([0, 0.04, 0.19, 1.06, 2.2])).all()
    assert np.equal(
        np.percentile(
            tcf.state.soc, (0, 50, 100), axis = 1).astype(np.int32),
        np.array([[ 140, 78,  211],
                  [ 164, 97, 3485],
                  [ 182,101, 6039]])).all()


def test_tcf_gpp_values():
    '''
    Test that the TCF model's GPP calculation is consistent.
    '''
    soc_state, drivers = random_tcf_data_cube(100, 100, seed = 406)
    tcf = TCF(CEREAL_PARAMETERS, 7, soc_state)
    gpp = tcf.gpp(drivers[0:6]).round(2)
    assert np.median(gpp) == 1.61
    assert np.var(gpp).round(2) == 7.93
    assert gpp.min() == 0.0
    assert gpp.max() == 24.92


def test_tcf_nee_values():
    '''
    Test that the TCF model's NEE calculation is consistent.
    '''
    soc_state, drivers = random_tcf_data_cube(100, 100, seed = 406)
    tcf = TCF(CEREAL_PARAMETERS, 7, soc_state)
    # Using just one time slice
    nee = tcf.nee(drivers[...,0]).round(2)
    assert np.median(nee) == -0.95
    assert np.var(nee).round(2) == 4.42
    assert nee.min() == -8.32
    assert nee.max() == 1.77


def test_tcf_rh_values():
    '''
    Test that the TCF model's RH calculation is consistent.
    '''
    soc_state, drivers = random_tcf_data_cube(100, 100, seed = 406)
    tcf = TCF(CEREAL_PARAMETERS, 7, soc_state)
    # Using just one time slice
    rh = tcf.rh(drivers[-2:][...,0]).sum(axis = 0).round(2)
    assert np.median(rh) == 0.56
    assert np.var(rh).round(2) == 0.21
    assert rh.min() == 0.0
    assert rh.max() == 1.77