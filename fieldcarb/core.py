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

    A complete description is available in Kimball et al. (2009) and in Jones
    et al. (2017). Soil organic carbon and soil decomposition are discussed
    in Endsley et al. (2020) and Endsley et al. (2022).

    Example use for a single pixel:

        tcf = TCF(land_cover_map = [7])
        tcf.nee_daily(state = [500, 500, 1000], drivers = [...])


    Parameters
    ----------
    land_cover_map : Sequence or numpy.ndarray
        1-dimensional sequence of one or more land-cover types, one for each
        model resolution cell (pixel)
    state : Sequence or numpy.ndarray
        A sequence of 3 values or an (3 x N) array representing the
        initial SOC state in each SOC pool
    '''

    valid_pft = {
        7: 'Cereal Croplands',
        8: 'Broadleaf Croplands'
    }

    def __init__(self, land_cover_map: Sequence, state: Sequence = None):
        self.pft = land_cover_map
        self.state = state


    def gpp_daily(self, drivers: Sequence) -> numpy.ndarray:
        '''
        Calculates daily gross primary production (GPP) under prevailing
        climatic conditions. Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            Root-zone soil moisture wetness, volume proportion [0-1]
            Freeze-thaw (FT) state [0 = Thawed, 1 = Frozen]

        The FT state is optional; if that axis of the data cube is not
        provided, FT state will be calculated from Tmin using a threshold
        of 32 degrees F.

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


    def nee_daily(
            self, drivers: Sequence, state: Sequence = None,
            compute_ft: bool = False
        ) -> numpy.ndarray:
        '''
        Calculates the net ecosystem CO2 exchange (NEE) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.
        Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            Root-zone soil moisture wetness, volume proportion [0-1]
            Freeze-thaw (FT) state [0 = Thawed, 1 = Frozen]
            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness, volume proportion [0-1]

        Even if compute_ft = True, the axis of "drivers" corresponding to FT
        state must be provided; it could be all NaNs.

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 3D data cube of
            shape (P x N x T), for N pixels, and T time steps
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            initial SOC state in each SOC pool
        compute_ft : bool
            If True, the freeze-thaw (FT) state is computed based on Tmin; if
            False, FT must be provided among drivers.

        Returns
        -------
        numpy.ndarray
        '''
        pass


    def rh_daily(self, drivers: Sequence, state: Sequence = None) -> numpy.ndarray:
        '''
        Calculates daily heterotrophic respiration (RH) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.
        Order of driver variables should be:

            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness, volume proportion [0-1]

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 3D data cube of
            shape (P x N x T), for N pixels, and T time steps
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            initial SOC state in each SOC pool

        Returns
        -------
        numpy.ndarray
        '''
        pass
