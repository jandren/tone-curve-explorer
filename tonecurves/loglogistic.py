import numpy as np


class LogLogisticSettings(object):
    def __init__(self):
        self.scene_grey = 0.1845
        self.display_black = 0.0
        self.display_grey = 0.1845
        self.display_white = 1.0
        self.contrast = 1.65
        self.skew = 0.0


class LogLogistic(object):
    def __init__(self):
        pass

    def __transfer_settings(self, settings):
        self.power = settings.contrast
        self.skew = pow(5.0, -settings.skew)

        self.magnitude = settings.display_white
        T = pow(settings.display_white / settings.display_grey, 1.0 / self.skew) - 1.0
        if settings.display_black > 0.0:
            z = pow(settings.display_white / settings.display_black, 1.0 / self.skew) - 1.0
            self.fog = settings.scene_grey * pow(T, 1.0 / self.power) / (pow(z, 1.0 / self.power) - pow(T, 1.0 / self.power))
        else:
            self.fog = 0.0
        self.paper_e = pow(self.fog + settings.scene_grey, self.power) * T        

    def apply(self, x, settings):
        self.__transfer_settings(settings)
        return self.magnitude * pow(1.0 + self.paper_e * pow(self.fog + x, -self.power), -self.skew)
