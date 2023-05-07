'''
Tests using or against the `pyl4c` library.
'''

import os
import numpy as np
import pytest
pytest.importorskip('pyl4c')
import fieldcarb
from fieldcarb.models import TCF
from test_models import random_tcf_data_cube
from pyl4c.data.fixtures import restore_bplut_flat

BPLUT = os.path.join(os.path.dirname(fieldcarb.__file__), 'data/SMAP_BPLUT_V7_rev_20220728.csv')

def test_tcf_gpp_using_v7_params_table_interface():
    '''
    Test that supplying the SMAP L4C BPLUT works for GPP calculation.
    '''
    params = restore_bplut_flat(BPLUT)
    # NOTE: L4C BPLUT has soil wetness in [%] units
    for key in ('smsf0', 'smsf1', 'smrz0', 'smrz1'):
        params[key] /= 100 # Convert from [%] to proportion
    _, drivers = random_tcf_data_cube(10, 365, seed = 406)
    tcf = TCF(params, np.random.choice(np.arange(1, 9), size = 10))
    gpp = tcf.gpp(drivers[0:6]).round(2)
    assert gpp.shape == (10, 365)


def test_tcf_rh_using_v7_params_table_interface():
    '''
    Test that supplying the SMAP L4C BPLUT works for RH calculation.
    '''
    params = restore_bplut_flat(BPLUT)
    # NOTE: L4C BPLUT has soil wetness in [%] units
    for key in ('smsf0', 'smsf1', 'smrz0', 'smrz1'):
        params[key] /= 100 # Convert from [%] to proportion
    soc_state, drivers = random_tcf_data_cube(10, 365, seed = 406)
    tcf = TCF(
        params, np.random.choice(np.arange(1, 9), size = 10),
        state = soc_state)
    rh = tcf.rh(drivers[-2:,:,0]).round(2)
    assert rh.shape == (3, 10)
