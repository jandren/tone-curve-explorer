import numpy as np
import pandas as pd
import tonecurves

class Blender(object):
    def __init__(self):
        self.data = pd.read_csv("./tonecurves/blender_curves.csv")
        # From the OCIO config file in Blender
        self.scene_intensity = pow(2.0, np.linspace(-12.473931188, 4.026068812, 4096))
        # 4.02... should possible be changed to 12.526068812 but this result makes more sense.

    def apply(self, x, curve_name):
        curve_linear_brightness = tonecurves.inv_srgb(self.data[curve_name])
        offset = 0.1845 / np.interp(0.1845, curve_linear_brightness, self.scene_intensity)
        return np.interp(x, offset * self.scene_intensity, curve_linear_brightness)
