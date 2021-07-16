from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
from scipy import integrate

from . import dispersion
from .decorators import dispatcher


class Integrate():
    """
    The parent class for the `spectrum`. Cannot be used separately
    """
    def __init__(self) -> None:
        pass


    @lru_cache()
    def quad(self, a: int, b: int, k0 = None, k1 = None, **kwargs) -> float:
        """
        Calculation of the mixed statistical moment of the sea wave elevations 1D-spectrum.
        The integral is calculated over a function of the form:
        `func(k,a,b) = k**a * omega(k)**b * S(k)`

        Parameters
        ----------
        a, b : int
           Powers for k and omega(k) 

        k0 : float
            Lower limit of integration (minimum of self.bounds by default).
        k1 : float
            Upper limit of integration (maximum of self.bounds by default).

        kwargs : any
            All keyword arguments of scipt.integrate.quad function.

        Returns
        -------
        var : float
            The resultant integral.
        """
        if k0==None:
            k0 = self.bounds[0]

        if k1==None:
            k1 = self.bounds[-1]

        S = lambda k: self.__call__(k, dispatcher=False) * k**a * dispersion.omega(k)**b
        var = integrate.quad(S, k0, k1, **kwargs)[0]

        return var




    @dispatcher
    @lru_cache()
    def dblquad(self, a, b, c, d = 0, k0 = None, k1 = None, phi0 = None, phi1 = None, **kwargs):

        """
        Calculation of the mixed statistical moment of the sea wave elevations 2D-spectrum.
        The integral is calculated over a function of the form:
        `func(k,a,b,c,d) = k**(a+b-c) * omega(k)**d * S(k, phi) * cos(phi)**a * sin(phi)**b`

        Parameters
        ----------
        func : callable
            A Python function or method of at least two variables: y must be the
            first argument and x the second argument.
        a, b, c, d: int 
           Powers for k, omega(k), sin(phi), cos(phi)
            
        k0 : float, optional
            Lower limit of integration by wavenumbers (minimum of `self.bounds` by default).
        k1 : float, optional
            Upper limit of integration by wavenumbers (maximum of `self.bounds` by default).

        phi0: float, optional
            Lower limit of integration by azimuth (`-np.pi` by default).
        phi1: float, optional
            Upper limit of integration by azimuth (`+np.pi` by default).

        kwargs : any
            Extra arguments to pass to `scipy.integrate.dblquad`

        Returns
        -------
        var : float
            The resultant integral.

        Examples
        --------

        Calculate the variance of the slopes along the x-axis direction
        >>> from seawave import spectrum
        >>> spectrum.dblquad(2, 0, 0)
        0.011070716562565735

        """
        limit = np.array([self.KT[0], *self.bounds, self.KT[-1]])

        if k0==None:
            k0 = self.KT[0]

        if k1==None:
            k1 = self.KT[-1]

        if phi0==None:
            phi0 = -np.pi
        
        if phi1==None:
            phi1 = np.pi
        

        S = lambda phi, k:  self.__call__(k, phi, dispatcher=False) *  \
            k**(a+b-c) * np.cos(phi)**a * np.sin(phi)**b * dispersion.omega(k)**d

        var = integrate.dblquad(S,
                a=k0, b=k1,
                gfun=lambda phi: phi0, 
                hfun=lambda phi: phi1, **kwargs)
        
        return var[0]

    def cov(self):

        cov = np.zeros((2, 2))
        cov[0, 0] = self.dblquad(2, 0, 0)
        cov[1, 1] = self.dblquad(0, 2, 0)
        cov[1, 0] = self.dblquad(1, 1, 0)
        cov[0, 1] = cov[1, 0]

    #     return cov



    # # def quad(self, a,b, k0=None, k1=None):
    #     S = lambda k, rho: self.get_spectrum()(k) *  np.cos(k*rho)
    #     limit = np.array([self.KT[0], *self.limit_k, self.KT[-1]])
    #     # k0 = np.logspace( np.log10(self.KT[0]), np.log10(self.KT[-1]), 2**10 + 1)
    #     k0 = np.linspace( self.KT[0], self.KT[-1], 2**11 + 1)
    #     k0[0] = self.KT[0]
    #     k0[-1] = self.KT[-1]

    #     integral=np.zeros(len(rho))
    #     for i in range(len(rho)):
    #         # integral[i] = integrate.quad(S, limit[0], limit[-1],args=(rho[i],))[0]
    #         # integral[i] = integrate.romb(S(k0, rho[i]), np.diff(k0[:2]))
    #         integral[i] =integrate.trapz(S(k0, rho[i]), k0)

    #     return integral

    # def fftstep(self, x):
    #     return np.pi/x


    # def fftfreq(self, xmax):
    #     step = self.fftstep(xmax)
    #     k = np.arange(-self.KT[-1], self.KT[-1], step)
    #     d = np.diff(k[0:2])
    #     return np.linspace(-np.pi/d, +np.pi/d, k.size)

    # def fftcorrelate(self, xmax, a=0, b=0, c=1):

    #     xkorr = 2*np.pi/self.KT[0]

    #     x = xmax
    #     # if xmax <= xkorr:
    #     #     x = 2*xkorr

    #     step = self.fftstep(x)
    #     k = np.arange(-self.KT[-1], self.KT[-1], step)
    #     D = k.max()

    #     S = lambda k: k**a * dispersion.omega(k)**b * self.ryabkova(k)**c

    #     S = fft.fftshift(S(k))
    #     K = fft.ifft(S) * D

    #     ind = int(np.ceil(S.size/2))
    #     # K = K[:ind]
    #     K = fft.fftshift(K)
    #     return K


    # def pdf_heights(self, z=None, dtype=None):
    #     if z is None:
    #         sigma0 = self.quad(0,0)
    #         z = np.linspace(-3*np.sqrt(sigma0), +3*np.sqrt(sigma0), 128)
    #         pdf = 1/np.sqrt(2*np.pi*sigma0) * np.exp(-1/2*z**2/sigma0)
    #     else:
    #         pdf, z = np.histogram(z, density=True, bins = "auto")
    #         z = z[:-1]

    #     if dtype == "cwm":
    #         sigma0 = self.quad(0,0)
    #         sigma1 = self.quad(1,0)
    #         pdf *= (1 - sigma1/sigma0*z)

    #     return pdf, z

    # def cdf_heights(self, *args, **kwargs):
    #     cdf = np.cumsum(self.pdf_heights(*args, **kwargs))
    #     return cdf

    # def cdf_slopes(self, *args, **kwargs):
    #     cdf = np.cumsum(self.pdf_slopes(*args, **kwargs))
    #     return cdf

    # def pdf_slopes(self, z=None, dtype="default"):

    #     if z is None:
    #         sigma0 = self.quad(2,0)
    #         z = np.linspace(-3*np.sqrt(sigma0), +3*np.sqrt(sigma0), 128)
    #         pdf = 1/np.sqrt(2*np.pi*sigma0) * np.exp(-1/2*z**2/sigma0)

    #     else:
    #         pdf, z = np.histogram(z, density=True, bins = "auto")
    #         z = z[:-1]

    #     if dtype == "cwm":
    #         sigma2 = self.quad(2,0)
    #         pdf = (
    #             np.exp(-1/(2*sigma2))/( np.pi*(1 + z**2)**2 )  +
    #                 (sigma2*(1+z**2) + 1)/np.sqrt(2*np.pi*sigma2)/(1+z**2)**(5/2) * 
    #                 erf(1/np.sqrt(2*sigma2*(1+z**2))) * np.exp(-1/(2*sigma2)*(z**2/(1+z**2)))
    #         )

    #     return pdf, z
