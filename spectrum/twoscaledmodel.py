from . import config
from scipy import optimize
import numpy as np
from . import integrate
from functools import lru_cache

class TwoScaledModel():
    """
    The parent class for the `spectrum`. Cannot be used separately
    """
    def __init__(self) -> None:
        pass

    def curv_criteria(self, band='Ku'):
        speckwargs = dict(dispatcher=False)
        # Сейчас попробуем посчитать граничное волновое число фактически из экспериментальных данных
        # Из работы Панфиловой известно, что полная дисперсия наклонов в Ku-диапазоне задается формулой

        # Дисперсия наклонов из статьи
        if band == "Ku":
            # var = lambda U10: 0.0101 + 0.0022*np.sqrt(U10)
            var = lambda U10: 0.0101 + 0.0022*U10
            radarWaveLength = 0.022

        elif band == "Ka":
            var = lambda U10: 0.0101 + 0.0034*U10
            radarWaveLength = 0.008

        # Необходимо найти верхний предел интегрирования, чтобы выполнялось равенство
        # интеграла по спектру и экспериментальной формулы для дисперсии
        # Интеграл по спектру наклонов


        epsabs = 1.49e-6
        Func = lambda k_bound: self.quad(2, 0, 0, k1=k_bound, epsabs=epsabs, ) - var(self._U)
        # Поиск граничного числа 
        # (Ищу ноль функции \integral S(k) k^2 dk = var(U10) )
        opt = optimize.root_scalar(Func, bracket=[0, 2000]).root

        # Значение кривизны 
        curv0 = self.quad(4,0,0,opt, epsabs=epsabs)

        # Критерий выбора волнового числа
        eps = np.power(radarWaveLength/(2*np.pi) * np.sqrt(curv0), 1/3)

        return eps

    def _find_k_bound(self, radarWaveLength,  **kwargs):
        eps = self.curv_criteria()
        Func = lambda k_bound: np.power( radarWaveLength/(2*np.pi) * np.sqrt(self.quad(4,0,0, k1=k_bound, epsabs=1.49e-6, )), 1/3 ) - eps
        # root = optimize.root_scalar(Func, bracket=[self.KT[0], self.KT[-1]]).root
        root = optimize.root_scalar(Func, bracket=[0, 2000]).root
        return root


    def kEdges(self, band, ):
        """
        Границы различных электромагнитных диапазонов согласно спецификации IEEE
        
        Band        Freq, GHz            WaveLength, cm         BoundaryWaveNumber, 
        Ka          26-40                0.75 - 1.13            2000 
        Ku          12-18                1.6  - 2.5             80 
        X           8-12                 2.5  - 3.75            40
        C           4-8                  3.75 - 7.5             10

        """
        bands = {"C":1, "X":2, "Ku":3, "Ka":4}

        k_m = self._peak

        if isinstance(band, str):
            bands_edges = [

                lambda k_m: k_m/4,

                lambda k_m: (
                    2.74 - 2.26*k_m + 15.498*np.sqrt(k_m) + 1.7/np.sqrt(k_m) -
                    0.00099*np.log(k_m)/k_m**2
                ),


                lambda k_m: (
                    25.82 + 25.43*k_m - 16.43*k_m*np.log(k_m) + 1.983/np.sqrt(k_m)
                    + 0.0996/k_m**1.5
                ),


                lambda k_m: (
                    + 68.126886 + 72.806451 * k_m  
                    + 12.93215 * np.power(k_m, 2) * np.log(k_m) 
                    - 0.39611989*np.log(k_m)/k_m 
                    - 0.42195393/k_m
                ),

                lambda k_m: (
                    #   24833.0 * np.power(k_m, 2) - 2624.9*k_m + 570.9
                    2000
                )

            ]
            edges = np.array([ bands_edges[i](k_m) for i in range(bands[band]+1)])
        else:

            edges = np.zeros(len(band)+1)
            edges[0] = k_m/4
            for i in range(1, len(edges)):
                if band[i-1] == 0:
                    edges[i] = 2000
                else:
                    edges[i] = self._find_k_bound(band[i-1], )

        return edges