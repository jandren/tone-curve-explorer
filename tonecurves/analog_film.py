import numpy as np
import pandas as pd

class AnalogFilm(object):
    def __init__(self):
        extracted_data = pd.read_csv("./tonecurves/analog_film.csv")
        print_density = np.array(extracted_data["PrintDensity"])
        # Normalize max transmission as 1
        self.print_transmission = pow(10.0, print_density[-1] - print_density)

        # Find 18 % grey exposure level and 
        log_intensity = extracted_data["LogExposure"]
        middle_exposure = np.interp(0.1845, self.print_transmission, log_intensity)
        self.intensity = pow(10.0, log_intensity - middle_exposure + np.log10(0.1845))

    def apply_print(self, x):
        return np.interp(x, self.intensity, self.print_transmission)
