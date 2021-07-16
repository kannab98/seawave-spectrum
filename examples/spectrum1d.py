import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from seawave_spectrum import spectrum, config


config["Wind"]["Speed"] = 7

S = spectrum()
plt.figure()
plt.loglog(spectrum.k, S)

plt.savefig("examples/spectrum1d.png")
