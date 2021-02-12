import numpy as np

class LogLogistic(object):
    def __init__(self):
        self.scene_grey = 0.1845
        self.display_black = 0.0
        self.display_grey = 0.1845
        self.display_white = 1.0
        self.contrast = 1.65
        self.skew = 0.0

    def __settings(self):
        self.skew_power = pow(5.0, -self.skew)
        self.power = pow(self.contrast, 1.0 / self.skew_power)

        self.magnitude = self.display_white
        T = pow(self.display_white / self.display_grey, 1.0 / self.skew_power) - 1.0
        if self.display_black > 0.0:
            z = pow(self.display_white / self.display_black, 1.0 / self.skew_power) - 1.0
            self.fog = self.scene_grey * pow(T, 1.0 / self.power) / (pow(z, 1.0 / self.power) - pow(T, 1.0 / self.power))
        else:
            self.fog = 0.0
        self.paper_e = pow(self.fog + self.scene_grey, self.power) * T        

    def apply(self, x):
        self.__settings()
        return self.magnitude * pow(1.0 + self.paper_e * pow(self.fog + x, -self.power), -self.skew_power)
