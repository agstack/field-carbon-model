'''
Unit tests for file interchange and driver data functions.
'''

import os
import agstack
from agstack.io import drivers_from_csv

CLIM_FILE = os.path.join(os.path.dirname(agstack.__file__), 'data/example_climatology_US-Ne3.csv')

def test_drivers_from_csv():
    drivers, dates = drivers_from_csv(
        CLIM_FILE, fields_diff = ('swrad', 'ps', 'qv2m', 'tmean'))
    assert drivers.shape == (9, 365)
