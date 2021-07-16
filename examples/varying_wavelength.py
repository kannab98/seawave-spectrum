import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from spectrum import spectrum, config


config["Radar"]["WaveLength"] = "Ku"

print(spectrum.bounds)
