import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from . import config
from . import dispersion
from .decorators import dispatcher, ufunc
from .integrate import Integrate
from .twoscaledmodel import TwoScaledModel

logger = logging.getLogger(__name__)


__all__ = ["spectrum"]

__bibtex__ = [
    """
@article{ryabkova:2019,
author ={Ryabkova, M. and Karaev, V. and Guo, J. and Titchenko, Yu.},
title = {A Review of Wave Spectrum Models as Applied to the Problem of Radar Probing of the Sea Surface},
year = 2019,
journal = {Journal of Geophysical Research: Oceans},
pages = {7104--7134}
""",
    """
@article{JONSWAP,
author ={Hasselmann K., T.P. Barnett, E. Bouws, H. Carlson, D.E. Cartwright, K. Enke, J.A. Ewing, H. Gienapp, D.E. Hasselmann, P. Kruseman, A. Meerburg, P. Mller, D.J. Olbers, K. Richter, W. Sell, and H. Walden.},
title = {Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP)},
year = 1973,
journal = {Ergnzungsheft zur Deutschen Hydrographischen Zeitschrift Reihe},
pages = {95},
number = {12}
""",

]


g = config['Constants']['GravityAcceleration']


class spectrum(Integrate, TwoScaledModel):
    """
    Ryabkova's model of the elevations spectrum, see __bibtex__[0]

    TODO:
        * Add a swell modification 

    Attributes
    ----------
    peak : float
        frequency of the spectral maximum in units [rad/m]
    omega_m : float
        frequency of the spectral maximum in units [Hz]
    lambda_m : float
        frequency of the spectral maximum in units [m]
    bounds : list
        simulated spectrum boundaries


    Methods
    -------
    __call__
        Depending on the arguments, it returns:
          - One-dimensional spectrum
          - Two-dimensional spectrum

    JONSWAP
        The base of the Ryabkova's model

    azimuthal_distribution 
        Azimuthal distribution used for 2D spectrum. See __bibtex__[0]

    quad
        Calculation of the statistical moments from a one-dimensional spectrum.
        For more details see module seawavepy.spectrum.integrate.

    dblquad
        Calculation of the statistical moments from a two-dimensional spectrum.
        For more details see module seawavepy.spectrum.integrate.
    """

    def __init__(self, **kwargs: Any) -> None:

        for Key, Value in kwargs.items():
            if isinstance(Value, dict):
                for key, value in Value.items():
                    config[Key][key] = value

        super(Integrate, self).__init__()

        # two-scaled model
        self.__tsm__ = False
        if config['Surface']['TwoScaledModel']:
            self.__tsm__ = True
            super(TwoScaledModel, self).__init__()

        self._x = config['Surface']['NonDimWindFetch']
        self._U = config['Wind']['Speed']
        self._wavelength = config['Radar']['WaveLength']
        self._band = config["Radar"]["WaveLength"]
        self._peak = None
        self.KT = np.array([1.49e-2, 2000])
        self._k = np.logspace(np.log10(self.KT.min()),
                              np.log10(self.KT.max()), 10**3+1)

    @property
    @dispatcher
    def bounds(self):
        return self.KT
    
    @property
    @dispatcher
    def peak(self):
        return self._peak

    @property
    @dispatcher
    def k(self) -> np.ndarray:
        return self._k

    @dispatcher
    def __call__(self, k: Optional[ArrayLike] = None, phi: Optional[ArrayLike] = None) -> np.ndarray:
        """
        The main function of the spectrum class. 
        Calculates one-dimensional and two-dimensional spectra of sea waves elevations.

        Parameters
        ----------
        k: ArrayLike, optional
            Arbitrary array of wave numbers. 
            If not defined, points are selected based on the current wave parameters

        phi: ArrayLike
            Array of azimuthal directions in radians.
            The 2D spectrum is not computed until this argument is specified.

        Returns
        -------
        np.ndarray with shape `(k.size)`
            One-dimensional spectrum. If azimuthal directions `phi` is not defined.
        np.ndarray with shape `(phi.size, k.size)`
            Two-dimensional spectrum. If azimuthal directions is specified. 
        """

        if not isinstance(k, np.ndarray):
            k = np.array(k)
            if k.ndim == 0:
                k = np.array([k.tolist()])

        if not isinstance(phi, np.ndarray):
            phi = np.array(phi)
            if phi.ndim == 0:
                phi = np.array([phi.tolist()])

        if np.isnan(k.astype(float)).all():
            k = self._k

        k = np.abs(k)

        limit = np.array([0, *self.limit_k, np.inf])

        spectrum1d = np.zeros(k.size, dtype=np.float64)

        for j in range(1, limit.size):
            self.__piecewise_spectrum__(
                j-1, k, where=(limit[j-1] <= k) & (k <= limit[j]), out=spectrum1d)

        if not np.isnan(phi.astype(float)).all():
            spectrum = self.azimuthal_distribution(k, phi, dtype='Wind')
            spectrum = spectrum1d * spectrum.T

        else:
            spectrum = spectrum1d

        if not np.isnan(phi.astype(float)).all() and config['Swell']['Enable']:
            swell1d = self.swell_spectrum(k)
            swell = swell1d * \
                self.azimuthal_distribution(k, phi, dtype="Swell").T
            spectrum = spectrum + swell.T

        return spectrum

    def azimuthal_distribution(self, k: ArrayLike, phi: ArrayLike, dtype: str = "Wind") -> np.ndarray:
        """
        The base of the Ryabkova's model.
        Calculates one-dimensional one-dimensional spectrum of sea waves elevations. See __bibtex__[1].

        Parameters
        ----------
        k: ArrayLike
            Positive array of wave numbers. 

        phi: ArrayLike
            Azimuthal directions in radians.

        dtype: str
            Future feature. Coming soon.

        Returns
        -------
        np.ndarray 
            Azimuthal distribution with shape (k.size, phi.size)


        """

        phi = np.angle(np.exp(1j*phi))
        phi -= np.deg2rad(config[dtype]["Direction"])
        phi = np.angle(np.exp(1j*phi))
        km = self._peak

        B0 = np.power(10, self.__az_exp_arg__(k, km))

        A0 = self.__az_normalization__(B0)

        phi = phi[np.newaxis]
        Phi = A0/np.cosh(2*B0*phi.T)
        return Phi.T

    def JONSWAP(self, k: ArrayLike):
        """
        The base of the Ryabkova's model.
        Calculates one-dimensional one-dimensional spectrum of sea waves elevations. See __bibtex__[1].

        Parameters
        ----------
        k: ArrayLike
            Positive array of wave numbers. 

        Returns
        -------
        np.ndarray with shape (k.size)
            One-dimensional spectrum of elevations 

        """
        return JONSWAP_vec(k, self._peak, self._alpha, self._gamma)

    @staticmethod
    def Gamma(x: float) -> float:
        """
        Parameters
        ----------
        x: float
            Non-dimensional wind fetch

        Returns
        -------
        float 
            Non-dimensional gamma-coefficient from the article __bibtex__[0]
        """

        if x >= 20170:
            return 1.0
        gamma = (
            +5.253660929
            + 0.000107622*x
            - 0.03778776*np.sqrt(x)
            - 162.9834653/np.sqrt(x)
            + 253251.456472*x**(-3/2)
        )
        return gamma

    @staticmethod
    def Alpha(x: float) -> float:
        """
        Parameters
        ----------
        x: float
            Non-dimensional wind fetch

        Returns
        -------
        float 
            Non-dimensional alpha-coefficient from the article __bibtex__[0]
        """
        if x >= 20170:
            return 0.0081
        else:
            alpha = (+0.0311937
                     - 0.00232774 * np.log(x)
                     - 8367.8678786/x**2
                     )
        return alpha

    @staticmethod
    def Omega(x: float) -> float:
        """
        Parameters
        ----------
        x: float
            Non-dimensional wind fetch

        Returns
        -------
        float 
            Non-dimensional frequency. See __bibtex__[0]
        """

        if x >= 20170:
            return 0.835

        omega_tilde = (0.61826357843576103
                       + 3.52883010586243843e-06*x
                       - 0.00197508032233982112*np.sqrt(x)
                       + 62.5540113059129759/np.sqrt(x)
                       - 290.214120684236224/x
                       )
        return omega_tilde

    """
    This is where public methods end. The further code will be documented for myself as necessary.
    """
    def __update__(self):
        logger.info('Refresh old spectrum parameters')
        x = config['Surface']['NonDimWindFetch']
        U = config['Wind']['Speed']
        Udir = config['Wind']['Direction']

        logger.info('Start modeling with U=%.1f, Udir=%.1f, x=%.1f, lambda=%s' % (
            U, Udir, x, str(config["Radar"]["WaveLength"])))
        # self.dblquad.cache_clear()

        # коэффициент gamma (см. спектр JONSWAP)
        self._gamma = self.Gamma(x)
        # коэффициент alpha (см. спектр JONSWAP)
        self._alpha = self.Alpha(x)

        # координата пика спектра по волновому числу
        self._peak = (self.Omega(x) / U)**2 * g

        # координата пика спектра по частоте
        self.omega_m = self.Omega(x) * g / U
        # длина доминантной волны
        self.lambda_m = 2 * np.pi / self._peak
        logger.info('Set peak\'s parameters: kappa=%.4f, omega=%.4f, lambda=%.4f' % (
            self._peak, self.omega_m, self.lambda_m))

        limit = np.zeros(5)
        limit[0] = 1.2 * self.omega_m
        limit[1] = (0.8*np.log(U) + 1) * self.omega_m
        limit[2] = 20.0
        limit[3] = 81.0
        limit[4] = 500.0

        __limit_k = np.array([dispersion.k(limit[i])
                              for i in range(limit.size)])
        self.limit_k = __limit_k[np.where(__limit_k <= 2000)]
        del __limit_k, limit

        waveLength = config['Radar']["WaveLength"]
        if waveLength != None and self.__tsm__:
            logger.info(
                'Calculate bounds of modeling for radar wave lenghts: %s' % str(waveLength))
            self.KT = self.kEdges(waveLength)

        elif not self.__tsm__:
            self.KT = np.array([0, 2000])

        # массив с границами моделируемого спектра.
        self._k = np.logspace(np.log10(self._peak/4),
                              np.log10(self.KT.max()), 10**3+1)

        logger.info('Set bounds of modeling %s' % str(np.round(self.KT, 2)))


    @staticmethod
    def __az_exp_arg__(k: ArrayLike, km: float) -> np.ndarray:
        k[np.where(k/km < 0.4)] = km * 0.4
        b = (
            -0.28+0.65*np.exp(-0.75*np.log(k/km))
            + 0.01*np.exp(-0.2+0.7*np.log10(k/km))
        )
        return b

    @staticmethod
    def __az_normalization__(B: float) -> float:
        return B/np.arctan(np.sinh(2*np.pi*B))



    @ufunc(3, 1)
    def __piecewise_spectrum__(self, n: int, k: ArrayLike) -> np.ndarray:
        power = [
            4,
            5,
            7.647*np.power(self._U, -0.237),
            0.0007*np.power(self._U, 2) - 0.0348*self._U + 3.2714,
            5,
        ]

        if n == 0:
            return self.JONSWAP(k)

        else:
            omega0 = dispersion.omega(self.limit_k[n-1])
            beta0 = self.__piecewise_spectrum__(n-1, self.limit_k[n-1]) * \
                omega0**power[n-1]/dispersion.det(self.limit_k[n-1])

            omega0 = dispersion.omega(k)
            return beta0 * np.power(omega0, -power[n-1]) * dispersion.det(k)

    @ufunc(2, 1)
    def swell_spectrum(self, k: ArrayLike) -> np.ndarray:

        omega_m = self.Omega(20170) * g/config['Swell']['Speed']
        W = np.power(omega_m/dispersion.omega(k), 5)

        sigma_sqr = 0.0081 * g**2 * np.exp(-0.05) / (6 * omega_m**4)

        spectrum = 6 * sigma_sqr * W / \
            dispersion.omega(k) * np.exp(-1.2 * W) * dispersion.det(k)
        return spectrum



def JONSWAP_vec(k: ArrayLike, km: float, alpha: float, gamma: float) -> np.ndarray:
    if k == 0:
        return 0

    if k >= km:
        sigma = 0.09
    else:
        sigma = 0.07

    Sw = (alpha/2 *
          np.power(k, -3) *
          np.exp(-1.25 * np.power(km/k, 2)) *
          np.power(gamma,
                   np.exp(- (np.sqrt(k/km) - 1)**2 / (2*sigma**2))
                   )
          )
    return Sw
