'''
The core carbon flux model(s). Currently, this is the Terrestrial Carbon
Flux (TCF) model.
'''

import numpy as np
from numbers import Number
from typing import Sequence
from fieldcarb import Namespace
from fieldcarb.utils import arrhenius, linear_constraint


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
    params : dict
        Dictionary of model parameters; keys should be among
        `TCF.required_parameters` and values should be sequences ordered by
        PFT code, either 9 values (value at index 0 being `np.nan`) or as many
        values as there are keys in `TCF.valid_pft`
    land_cover_map : Sequence or numpy.ndarray
        1-dimensional sequence of one or more land-cover types, one for each
        model resolution cell (pixel)
    state : Sequence or numpy.ndarray
        A sequence of 3 values or an (3 x N) array representing the
        initial SOC state in each SOC pool
    '''

    required_parameters = [
        'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0',
        'CUE', 'tsoil', 'smsf0', 'smsf1', 'decay_rates'
    ]
    valid_pft = {
        7: 'Cereal Croplands',
        8: 'Broadleaf Croplands'
    }

    def __init__(
            self, params: dict, land_cover_map: Sequence,
            state: Sequence = None
        ):
        self.params = Namespace()
        self.pft = land_cover_map
        self.state = state
        # Each parameter should be accessed, e.g., tcf.params.LUE
        for key, value in params.items():
            self.params.add(key, value)

    def _rescale_smrz(self, smrz0, smrz_min, smrz_max = 1):
        r'''
        Rescales root-zone soil-moisture (SMRZ) to increase plant sensitivity to
        very low water availability.

        $$
        \hat{\theta} &= 100 \times\left(
        \frac{\theta - \theta_{WP}}{\text{max}(\theta) - \theta_{WP}}
        \right) + 1\\
        \theta_{RZ} &= 0.95 \times
        \frac{\text{ln}(\hat{\theta} \times 100)}{\text{ln(101)}} + 0.05
        $$

        Parameters
        ----------
        smrz0 : numpy.ndarray
            (N x T) array of original SMRZ data, in proportion saturation [0-1]
            units for N sites and T time steps
        smrz_min : numpy.ndarray or float
            Site-level long-term minimum SMRZ (proportion saturation)
        smrz_max : numpy.ndarray or float
            Site-level long-term maximum SMRZ (proportion saturation); can
            optionally provide a fixed upper-limit on SMRZ

        Returns
        -------
        numpy.ndarray
        '''
        # Clip input SMRZ to the lower, upper bounds
        smrz0 = np.where(smrz0 < smrz_min, smrz_min, smrz0)
        smrz0 = np.where(smrz0 > smrz_max, smrz_max, smrz0)
        smrz_norm = np.divide(
            np.subtract(smrz0, smrz_min),
            np.subtract(smrz_max, smrz_min)) + 0.01
        # Log-transform normalized data and rescale to range between
        #   5.0 and 100% saturation)
        return np.add(
            np.multiply(0.95, np.divide(np.log(smrz_norm * 100), np.log(101))), 0.05)

    def forward_run(self, drivers: Sequence) -> np.ndarray:
        '''
        Runs the TCF model forward in time. This is the recommended interface
        for most users.
        '''
        pass

    def gpp(self, drivers: Sequence) -> np.ndarray:
        '''
        Calculates gross primary production (GPP) under prevailing climatic
        climatic conditions. Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            Root-zone soil moisture wetness, volume proportion [0-1]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]

        The FT state is optional; if that axis of the data cube is not
        provided, FT state will be calculated from Tmin using a threshold
        of 32 degrees F (273.15 deg K).

        Unit of time should be consistent with the units of PAR. For example,
        if PAR is given as [MJ m-2 day-1], then GPP will be in units of
        [g C m-2 day-1]. It's assumed the other driver data are representative
        of that time step (e.g., daily averages). GPP doesn't depend on model
        state, so it can be estimated for an arbitrary number of time steps.

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables; a 2D (P x N) array for
            N pixels, or a 3D data cube of shape (P x N x T) for T time steps

        Returns
        -------
        numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Gross primary production (GPP) in [g C m-2 time-1] where time is
            the time step of the PAR data, e.g., [g c m-2 day-1]
        '''
        if drivers.shape[0] == 5:
            fpar, par, tmin, vpd, smrz0 = drivers
            ft0 = np.where(tmin < 273.15, 0, 1)
        else:
            fpar, par, tmin, vpd, smrz0, ft0 = drivers
        # Rescale root-zone soil moisture
        smrz = self._rescale_smrz(smrz0, np.nanmin(smrz0, axis = -1))
        # Convert freeze-thaw flag to a multiplier (always 1 when thawed but
        #   potentially non-zero and less than 1 when thawed)
        ft = np.where(ft0 == 0, self.params.ft0, 1)
        # Get a function that constrains each met. driver to [0, 1]
        f_tmin = linear_constraint(self.params.tmin0, self.params.tmin1)
        f_vpd = linear_constraint(
            self.params.vpd0, self.params.vpd1, 'reversed')
        f_smrz = linear_constraint(self.params.smrz0, self.params.smrz1)
        # Compute the environmental constraint
        e_mult = ft * f_tmin(tmin) * f_vpd(vpd) * f_smrz(smrz)
        return par * fpar * e_mult * self.params.LUE

    def nee(
            self, drivers: Sequence, state: Sequence = None,
            compute_ft: bool = False
        ) -> np.ndarray:
        '''
        Calculates the net ecosystem CO2 exchange (NEE) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.
        Order of driver variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            Root-zone soil moisture wetness, volume proportion [0-1]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]
            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness, volume proportion [0-1]

        Even if compute_ft = True, the axis of "drivers" corresponding to FT
        state must be provided; it could be all NaNs.

        Unit of time should be consistent with the units of PAR and the
        turnover time (`decay_rate`). It's assumed that PAR is denominated
        by daily time steps, so NEE would be given in [g C m-2 day-1].

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 2D data cube of
            shape (P x N), for N pixels
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool
        compute_ft : bool
            If True, the freeze-thaw (FT) state is computed based on Tmin; if
            False, FT must be provided among drivers.

        Returns
        -------
        numpy.ndarray
        '''
        if hastattr(drivers, 'ndim'):
            assert drivers.ndim <= 2
        gpp = self.gpp(drivers)
        npp = self.params.CUE * gpp
        # Total the RH flux from each SOC pool
        rh = self.rh(drivers, state).sum(axis = 0)
        return rh - npp

    def rh(
            self, drivers: Sequence, state: Sequence = None
        ) -> np.ndarray:
        '''
        Calculates heterotrophic respiration (RH) based on the available
        soil organic carbon (SOC) state and prevailing climatic conditions.
        Order of driver variables should be:

            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness, volume proportion [0-1]

        Unit of time should be consistent with the units of PAR and the
        turnover time (`decay_rate`). It's assumed that PAR is denominated
        by daily time steps, so RH would be given in [g C m-2 day-1].

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables or a 2D data cube of
            shape (P x N), for N pixels
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool

        Returns
        -------
        numpy.ndarray
            A (3 x N) array representing the RH flux from each SOC pool
        '''
        if hastattr(drivers, 'ndim'):
            assert drivers.ndim <= 2
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state
        tsoil, smsf = drivers # Unpack met. drivers
        f_smsf = linear_constraint(self.params.smsf0, self.params.smsf1)
        tmult = arrhenius(tsoil, self.params.tsoil)
        wmult = f_smsf(smsf)
        rh = self.decay_rates * wmult * tmult * soc
        # "the adjustment...to account for material transferred into the slow
        #   pool during humification" (Jones et al. 2017, TGARS, p.5); note
        #   that this is a loss FROM the "medium" (structural) pool
        rh[1,...] = rh[1,...] * (1 - self.params.f_structural)
        return rh
