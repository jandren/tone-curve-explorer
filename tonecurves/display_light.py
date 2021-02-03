import numpy as np

def inv_srgb(u):
    output = np.zeros_like(u)
    mask = u <= 0.04045
    output[mask] = u[mask] / 12.92
    output[~mask] = pow((u[~mask] + 0.055) / 1.055, 2.4)
    return output

def inv_hlg(u):
    output = np.zeros_like(u)
    mask = u <= 0.5
    output[mask] = pow(2.0 * u[mask], 2.0)
    a = 0.17883277
    b = 1 - 4*a
    c = 0.5 - a * np.log(4*a)
    output[~mask] = np.exp((u[~mask] - c) / a) + b
    return output
