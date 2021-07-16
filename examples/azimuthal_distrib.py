import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from spectrum import spectrum, config


config["Wind"]["Speed"] = 7
config["Wind"]["Direction"] =  45

k = np.array([spectrum.peak])
phi = np.linspace(-np.pi, np.pi, 256)
S = spectrum.azimuthal_distribution(k, phi)

plt.figure()
plt.polar(phi, S.T)

plt.savefig("examples/azimuthal_distribution.png")
