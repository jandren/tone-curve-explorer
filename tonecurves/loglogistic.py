import numpy as np


class LogLogisticSettings(object):
    def __init__(self):
        self.scene_grey = 0.1845
        self.display_black = 0.0
        self.display_grey = 0.1845
        self.display_white = 1.0
        self.contrast = 1.65


class LogLogistic(object):
    def __init__(self):
        self.__transfer_settings(LogLogisticSettings())

    def __transfer_settings(self, settings):
        self.threshold = settings.scene_grey
        self.offset = settings.display_black
        self.denominator = (settings.display_black - settings.display_grey) / (settings.display_grey - settings.display_white)
        self.magnitude = -self.denominator * (settings.display_black - settings.display_white)
        self.power = settings.contrast

    def apply(self, x, settings):
        self.__transfer_settings(settings)
        scaled_x = x / self.threshold
        return self.offset + self.magnitude / (self.denominator + pow(scaled_x, -self.power))
