'''
Utilities for creating consistent model driver data. The common format for
raw driver data is a (P x N x T) data cube of P raw fields, in order:

    Abbreviation    Description
    ------------    -----------
    fpar
    swrad
    tmean
    qv2m
    ps
    tmin
    smrz
    tsoil
    smsf
'''

import numpy as np


def drivers_for_tcf(drivers: np.ndarray) -> np.ndarray:
    '''
    Generates a drivers data cube for TCF, which expects the following
    drivers, in order:

        Fraction of PAR intercepted (fPAR) [0-1]
        Photosynthetically active radation (PAR) [MJ m-2 day-1]
        Minimum temperature (Tmin) [deg K]
        Vapor pressure deficit (VPD) [Pa]
        Root-zone soil moisture wetness, volume proportion [0-1]
        Freeze-thaw (FT) state of soil [0 = Frozen, 1 = Thawed]
        Soil temperature in the top (0-5 cm) layer [deg K]
        Surface soil moisture wetness, volume proportion [0-1]

    Parameters
    ----------
    drivers : numpy.ndarray
        Input raw driver datasets, either a (P x N x T) array or a (P x T)
        array of P driver fields, N pixels, and T time steps

    Returns
    -------
    numpy.ndarray
    '''
    fpar, swrad, tmean, qv2m, ps, tmin, smrz, tsoil, smsf = drivers
    assert np.nanmin(smrz) >= 0 and np.nanmax(smrz) <= 1,\
        'Surface soil moisture (SMSF) is not bounded by [0,1]'
    assert np.nanmin(smrz) >= 0 and np.nanmax(smrz) <= 1,\
        'Root-zone soil moisture (SMRZ) is not bounded by [0,1]'
    assert np.nanmin(tmin) >= 200 and np.nanmax(smrz) <= 320,\
        'Minimum temperature is probably not in deg K; should be bounded [200,320]'
    assert np.nanmin(tmean) >= 200 and np.nanmax(tmean) <= 320,\
        'Mean temperature is probably not in deg K; should be bounded [200,320]'
    par = par_from_shortwave(swrad)
    vpd = vapor_pressure_deficit(qv2m, ps, tmean)
    ft = np.where(tmin < 0, 0, 1)
    result = np.stack([
        fpar, par, tmin, vpd, smrz, ft, tsoil, smsf
    ], axis = 0)
    if result.ndim == 2:
        return result[:,np.newaxis,:]
    return result


def par_from_shortwave(swrad):
    '''
    Photosynthetically active radiation (PAR), as a fraction of downwelling
    short-wave radiation (MERRA-2 SWGDN); after calculating the portion of
    SWGDN that is PAR (0.45), convert from [W m-2] to [MJ m-2 d-1].

    Parameters
    ----------
    swrad : Number or numpy.ndarray

    Returns
    -------
    numpy.ndarray
    '''
    # Take 24-hour mean of SWGDN, then convert from W m-2 to MJ m-2 day-1;
    # 11.5741 is derived as 1 / ((60 secs * 60 mins * 24 hrs) / 1e6),
    #   given that 1 W == 1 J sec-1
    return np.divide(np.multiply(0.45, swrad), 11.5741)


def vapor_pressure_deficit(qv2m, ps, temp_k):
	r'''
    Calculates vapor pressure deficit (VPD).

    $$
    \mathrm{VPD} = 610.7 \times \mathrm{exp}\left(
    \frac{17.38 \times T_C}{239 + T_C}
    \right) - \frac{(P \times [\mathrm{QV2M}]}{0.622 + (0.378 \times [\mathrm{QV2M}])}
    $$

    Where P is the surface pressure (Pa), QV2M is the water vapor mixing
    ratio at 2-meter height, and T is the temperature in degrees C (though
    this function requires units of Kelvin when called). Using MERRA-2, the
    corresponding input datasets are:

        Water vapor mixing ratio at 2-m height (QV2M) [kg kg-1]
        Surface air pressure [Pa]
        Minimum temperature (Tmin) [deg K]

    NOTE: A variation on this formula can be found in the text:

        Monteith, J. L. and M. H. Unsworth. 1990.
        Principles of Environmental Physics, 2nd. Ed. Edward Arnold Publisher.

    See also:
        https://glossary.ametsoc.org/wiki/Mixing_ratio

    Parameters
    ----------
    qv2m : numpy.ndarray or float
        QV2M, the water vapor mixing ratio at 2-m height
    ps : numpy.ndarray or float
        The surface pressure, in Pascals
    temp_k : numpy.ndarray or float
        The temperature at 2-m height in degrees Kelvin

    Returns
    -------
    numpy.ndarray or float
        VPD in Pascals
    '''
	temp_c = temp_k - 273.15 # Convert temperature to degrees C
	avp = np.divide(np.multiply(qv2m, ps), 0.622 + (0.378 * qv2m))
	x = np.divide(17.38 * temp_c, (239 + temp_c))
	esat = 610.7 * np.exp(x)
	return np.subtract(esat, avp)
