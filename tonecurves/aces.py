import numpy as np
import pandas as pd
import tonecurves

class Aces(object):
    def __init__(self):
        extracted_data = pd.read_csv("./tonecurves/aces_curves.csv")
        self.intensity = extracted_data["luma"]
        self.srgb_offset = pow(2.0, -0.5480658487457724)
        self.hlg_offset = pow(2.0, 0.8545276465490544)
        max16bit = pow(2.0, 16.0)
        self.srgb_linearized = tonecurves.inv_srgb(extracted_data["sRGB"] / max16bit)
        self.hlg_linearized = tonecurves.inv_hlg(extracted_data["HLG"] / max16bit)

    def apply_srgb(self, x):
        return np.interp(x, self.srgb_offset * self.intensity, self.srgb_linearized)

    def apply_hlg(self, x):
        return np.interp(x, self.hlg_offset * self.intensity, self.hlg_linearized)
