'''
The core carbon flux model(s). Currently, this is the Terrestrial Carbon
Flux (TCF) model.
'''

import numpy as np
from typing import Number, Sequence


class TCF(object):
    '''
    The Terrestrial Carbon Flux (TCF) model, a starting point for field-scale
    carbon (CO2) flux modeling. TCF incorporates these basic assumptions:

    1. Carbon assimilation (gross primary production) is linearly related to
        the amount of solar radiation and green photosynthetic land cover.
    2. There is a maximum rate of carbon assimilation. Low water availability,
        cold or freezing temperatures, and high atmospheric demand for water
        vapor each reduce the efficiency of carbon assimilation.
    3. Soil organic carbon decomposes at a different rates based on the type
        of material. The optimal rate of decomposition is fixed and depends
        only on the land-cover type.
    4. The efficiency of soil organic carbon decomposition is reduced under
        low soil temperatures or low soil moisture.

    Example use for a single pixel:

        tcf = TCF(land_cover_map = [7])
        tcf.nee_daily(state = [500, 500, 1000], drivers = [...])


    Parameters
    ----------
    land_cover_map : Sequence or numpy.ndarray
        1-dimensional sequence of one or more land-cover types, one for each
        model resolution cell (pixel)
    '''

    valid_pft = {
        7: 'Cereal Croplands',
        8: 'Broadleaf Croplands'
    }

    def __init__(self, land_cover_map: Sequence):
        self.pft = land_cover_map


    def gpp_daily(self, drivers: Sequence) -> numpy.ndarray:
        '''
        Calculates daily gross primary production (GPP) under prevailing
        climatic conditions.

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 3D data cube of
            shape (P x N x T), for N pixels, and T time steps

        Returns
        -------
        numpy.ndarray
        '''
        pass


    def nee_daily(self, state: Sequence, drivers: Sequence) -> numpy.ndarray:
        '''
        Calculates the net ecosystem CO2 exchange (NEE) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.

        Parameters
        ----------
        state : Sequence or numpy.ndarray
            A sequence of 3 values or an (3 x N) array representing the
            initial SOC state in each SOC pool
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 3D data cube of
            shape (P x N x T), for N pixels, and T time steps

        Returns
        -------
        numpy.ndarray
        '''
        pass


    def rh_daily(self, state: Sequence, drivers: Sequence) -> numpy.ndarray:
        '''
        Calculates daily heterotrophic respiration (RH) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.

        Parameters
        ----------
        state : Sequence or numpy.ndarray
            A sequence of 3 values or an (3 x N) array representing the
            initial SOC state in each SOC pool
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 3D data cube of
            shape (P x N x T), for N pixels, and T time steps

        Returns
        -------
        numpy.ndarray
        '''
        pass
