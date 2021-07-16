from . import dispersion
import numpy as np
from . import config

def dispatcher(func):
    """
    Декоратор обновляет необходимые переменные при изменении
    разгона или скорости ветра
    """
    def wrapper(*args, dispatcher=True, **kwargs):
        # self = spectrum

        if dispatcher:
            self = args[0]
            x = config['Surface']['NonDimWindFetch']
            U = config['Wind']['Speed']
            waveLength = config['Radar']['WaveLength']

            if self._x != x or self._U != U or self._peak == None or \
            (self._wavelength != waveLength):
                self.__update__()

                if hasattr(func, 'cache_clear'): func.cache_clear()

            self._x, self._U = x, U
            self._wavelength = config["Radar"]["WaveLength"]



        return func(*args, **kwargs)
    return wrapper

def ufunc(nin, nout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ufunc = np.frompyfunc(func, nin, nout)
            return ufunc(*args, **kwargs)
        return wrapper
    return decorator