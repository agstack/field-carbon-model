'''
The core carbon flux model(s). Currently, this is the Terrestrial Carbon
Flux (TCF) model.

Notes for potential future improvements:

- Currently, in `TCF.forward_run()`, `TCF.state.soc` does not record the
    history of dynamic SOC change, only the current SOC state; i.e., the state
    as of the most recent time step. This reduces the memory demand but it
    may be preferable in the future to be able to retrieve the history of
    change in SOC.
'''

import numpy as np
from numbers import Number
from typing import Sequence
from tqdm import tqdm
from fieldcarb import Namespace
from fieldcarb.utils import arrhenius, linear_constraint, climatology365


class TCF(object):
    '''
    The Terrestrial Carbon Flux (TCF) model, a starting point for field-scale
    carbon (CO2) flux modeling. TCF incorporates these basic assumptions:

    1. Carbon assimilation (gross primary production) is linearly related to
        the amount of solar radiation and green photosynthetic land cover.
    2. There is a maximum rate of carbon assimilation. Low water availability,
        cold or freezing temperatures, and high atmospheric demand for water
        vapor each reduce the efficiency of carbon assimilation.
    3. Soil organic carbon (SOC) decomposes at a different rates based on the
        type of material. The optimal rate of decomposition is fixed and
        depends only on the land-cover type.
    4. The efficiency of soil organic carbon decomposition is reduced under
        low soil temperatures or low soil moisture.

    A complete description is available in Kimball et al. (2009) and in Jones
    et al. (2017). Soil organic carbon (SOC) and soil decomposition are
    discussed in Endsley et al. (2020) and Endsley et al. (2022).

    If initial model "state" is provided it should be a 2D array with the
    first axis containing, in order:

    - Initial state of "metabolic" SOC pool
    - Initial state of "structural" SOC pool
    - Initial state of "recalcitrant" SOC pool

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
        model resolution cell (pixel). These types should be represented as
        integers (unsigned 16-bit type or lower).
    state : Sequence or numpy.ndarray or None
        A sequence of 3 values or a 2D (S x N) array representing the state
        variables for each of N pixels. The first three (3) state variables
        should be the initial states of each SOC pool.
    litterfall : Sequence or numpy.ndarray or None
        A sequence values or 1D array representing average daily litterfall
        for each model resolution cell (pixel)
    '''

    required_parameters = [
        'LUE', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'smrz0', 'smrz1', 'ft0',
        'CUE', 'tsoil', 'smsf0', 'smsf1', 'decay_rates', 'f_structural',
        'f_metabolic'
    ]
    valid_pft = {
        7: 'Cereal Croplands',
        8: 'Broadleaf Croplands'
    }

    def __init__(
            self, params: dict, land_cover_map: Sequence,
            state: Sequence = None, litterfall: Sequence = None
        ):
        self.constants = Namespace()
        self.state = Namespace()
        self.params = Namespace() # Parameters accessed, e.g., tcf.params.LUE
        self.pft = np.array(land_cover_map, dtype = np.uint16)
        # Load mean daily litterfall rates
        if litterfall is not None:
            litterfall = np.array(litterfall, dtype = np.float32)
        self.constants.add('litterfall', litterfall)
        # Load soil organic carbon (SOC) state
        if state is not None:
            if hasattr(state, 'ndim'):
                assert state.ndim <= 2, '"state" should have at most 2 dimensions'
            # "state" either begins as or is converted to a numpy.ndarray
            state = np.array(state, dtype = np.float32)
            assert len(state) == 3, 'Expected one "state" value for each SOC pool'
            self.state.add('soc', np.array(state))
        # Create a parameters vector; e.g., for a land-cover map:
        #   array([  1,   2,   1,   3])
        # We create an array of parameters:
        #   array([1.5, 3.0, 1.5, 1.0])
        # Where the unique parameter for each land-cover class is copied as
        #   many times as that class appears; result is an array that is the
        #   same shape and size as the land-cover array
        for key, value in params.items():
            if key not in self.required_parameters:
                continue
            try:
                p_vector = value[:,self.pft]
            except TypeError:
                p_vector = np.array([value])
            if key == 'decay_rates':
                if p_vector.shape == (1, 3):
                    p_vector = p_vector.reshape((3, 1))
            elif p_vector.ndim == 2 and p_vector.shape[0] == 1:
                # Reshape for compatibility with (N x T) arrays
                p_vector = p_vector.swapaxes(0, 1)
            self.params.add(key, p_vector)

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
        smrz_min = np.array(smrz_min)
        if smrz_min.ndim == 1:
            smrz_min = smrz_min[:,np.newaxis]
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

    def forward_run(
            self, drivers: Sequence, state: Sequence = None,
            dates: Sequence = None, verbose: bool = True
        ) -> np.ndarray:
        '''
        Runs the TCF model forward in time for daily time steps. This is the
        recommended interface for most users. If `npp_sum` was not provided to
        `TCF` at initialization, it will be necessary to provide at least 365
        daily steps and the `years` of each time step. Order of driver
        variables should be:

            Fraction of PAR intercepted (fPAR) [0-1]
            Photosynthetically active radation (PAR) [MJ m-2 day-1]
            Minimum temperature (Tmin) [deg K]
            Vapor pressure deficit (VPD) [Pa]
            Root-zone soil moisture wetness, volume proportion [0-1]
            Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]
            Soil temperature in the top (0-5 cm) layer [deg K]
            Surface soil moisture wetness, volume proportion [0-1]

        GPP calculation is vectorized but RH and NEE calculation proceed
        step-wise because they depend on the model state (SOC).

        Parameters
        ----------
        drivers : Sequence or numpy.ndarray
            Either a 1D sequence of P driver variables; a 2D (P x N) array for
            N pixels, or a 3D data cube of shape (P x N x T) for T time steps
        state : Sequence or numpy.ndarray or None
            A sequence of 3 values or an (3 x N) array representing the
            SOC state in each SOC pool
        dates : Sequence or numpy.ndarray or None
            If `npp_sum` was not provided to `TCF` during initialization, you
            must provide a sequence of `datetime.date` instances, of length T
            for T time steps, indicating the current year of each time step.
        verbose : bool
            True to show a progress bar and other messages (Default: True)

        Returns
        -------
        tuple
            A 3-element tuple of (NEE, GPP, RH)
        '''
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state.soc
        litter = self.constants.litterfall
        if litter is None:
            assert dates is not None,\
                'Either: "litterfall" must be provided to TCF() or "dates" must be provided to TCF.forward_run()'
            assert len(dates) >= 365 and drivers.shape[-1] >= 365,\
                'At least 365 daily time steps must be provided to allow computation of annual NPP sum'
            assert hasattr(dates[0], 'year') and hasattr(dates[0], 'strftime'),\
                'The values of "dates" must be datetime.date or datetime.datetime instances'
        # GPP can be computed matrix-wise, in a single time step
        gpp = self.gpp(drivers[0:6])
        npp = self.params.CUE * gpp
        # Compute litterfall from the mean annual NPP sum
        if litter is None:
            npp_sum = climatology365(gpp.swapaxes(0, 1), dates).sum(axis = 0)
            # Litterfall is equal daily fraction of average annual NPP
            litter = npp_sum / 365
            self.constants.add('litterfall', litter)
        # Pre-allocate output arrays
        rh = np.ones((3, *gpp.shape), dtype = np.float32) # (3 x N x T)
        nee = np.ones((*gpp.shape,), dtype = np.float32) # (N x T)
        # Pre-compute environmental constraints for soil RH
        tsoil, smsf = drivers[-2:]
        f_smsf = linear_constraint(self.params.smsf0, self.params.smsf1)
        # Swap axes here only to make time the major (first) axis
        tmult = arrhenius(tsoil, self.params.tsoil).swapaxes(0, 1)
        wmult = f_smsf(smsf).swapaxes(0, 1)
        # Forward time steps
        steps = range(0, drivers.shape[-1])
        for t in tqdm(steps, disable = not verbose):
            rh_t = np.empty((3, litter.shape[0])) # Allocate RH(t) array
            for pool in range(0, soc.shape[0]):
                rh_t[pool] = self.params.decay_rates[pool] *\
                    wmult[t] * tmult[t] * soc[pool]
            # Compute SOC change
            dc1 = (litter * self.params.f_metabolic) - rh_t[0,...]
            dc2 = (litter * (1 - self.params.f_metabolic)) - rh_t[1,...]
            dc3 = (self.params.f_structural * rh_t[1,...]) - rh_t[2,...]
            for i, delta in enumerate([dc1, dc2, dc3]):
                soc[i] += delta
            # "the adjustment...to account for material transferred into the slow
            #   pool during humification" (Jones et al. 2017, TGARS, p.5); note
            #   that this is a loss FROM the "medium" (structural) pool
            rh_t[1,...] = rh_t[1,...] * (1 - self.params.f_structural)
            # Record RH and NEE at this time step
            rh[...,t] = rh_t
            nee[...,t] = rh_t.sum(axis = 0) - npp[...,t]
        return (nee, gpp, rh)

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
            Net ecosystem exchange (NEE) in [g C m-2 time-1] where time is
            the time step of the PAR data, e.g., [g c m-2 day-1]
        '''
        if hasattr(drivers, 'ndim'):
            assert drivers.ndim <= 2,\
                'TCF.nee() computes a single time step; only (P x N) driver arrays should be provided'
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state.soc
        gpp = self.gpp(drivers[0:6])
        npp = self.params.CUE * gpp
        # Total the RH flux from each SOC pool
        rh = self.rh(drivers[-2:], state).sum(axis = 0)
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
            A (3 x N) array representing the RH flux from each SOC pool, in
            units of [g C m-2] per unit time, most likely [g C m-2 day-1]
        '''
        if hasattr(drivers, 'ndim'):
            assert drivers.ndim <= 2,\
                'TCF.rh() computes a single time step; only (P x N) driver arrays should be provided'
        # NOTE: Allowing for state variables other than SOC to be included in
        #   a later version
        soc = state
        if soc is None:
            soc = self.state.soc
        tsoil, smsf = drivers # Unpack met. drivers
        f_smsf = linear_constraint(self.params.smsf0.T, self.params.smsf1.T)
        tmult = arrhenius(tsoil, self.params.tsoil.T)
        wmult = f_smsf(smsf)
        rh = wmult * tmult * self.params.decay_rates * soc
        # "the adjustment...to account for material transferred into the slow
        #   pool during humification" (Jones et al. 2017, TGARS, p.5); note
        #   that this is a loss FROM the "medium" (structural) pool
        rh[1,...] = rh[1,...] * (1 - self.params.f_structural.T)
        return rh
