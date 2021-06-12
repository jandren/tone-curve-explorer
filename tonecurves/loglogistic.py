import numpy as np

class LogLogistic(object):
    def __init__(self):
        self.scene_grey = 0.1845
        self.display_black = 0.0
        self.display_grey = 0.1845
        self.display_white = 1.0
        self.contrast = 1.65
        self.skew = 0.0
    
    def find_film_power(self):
        # Actual wanted slope should be the same as for without skew
        # and with constant display targets
        self.paper_power = 1.0
        self.film_power = self.contrast
        self.magnitude = 1.0
        self.fog = 0.0
        self.paper_e = pow(self.scene_grey, self.film_power) * ((self.magnitude / self.display_grey) - 1.0)
        target_slope = self.__calc_slope_at(self.scene_grey)

        # Add skew
        self.paper_power = pow(5.0, -self.skew)

        # Slope at low film power
        self.film_power = 1.0
        self.__settings()
        self.fog = 0.0
        slope_1 = self.__calc_slope_at(self.scene_grey)

        # Figure out where it fulfills the target slope
        # (linear when assuming display_black=0.0)
        self.film_power = target_slope / slope_1

        # Redo settings such that black target can be taken into consideration
        self.__settings()
        return self.film_power

    def __calc_slope_at(self, point):
        dt = 0.00000001
        return (self.__evaluate(self.scene_grey + dt) - self.__evaluate(self.scene_grey - dt)) / 2 / dt

    def __settings(self):
        self.magnitude = self.display_white
        T = pow(self.display_white / self.display_grey, 1.0 / self.paper_power) - 1.0
        if self.display_black > 0.0:
            z = pow(self.display_white / self.display_black, 1.0 / self.paper_power) - 1.0
            self.fog = self.scene_grey * pow(T, 1.0 / self.film_power) / (pow(z, 1.0 / self.film_power) - pow(T, 1.0 / self.film_power))
        else:
            self.fog = 0.0
        self.paper_e = pow(self.fog + self.scene_grey, self.film_power) * T

    def __evaluate(self, x):
        #return self.magnitude * pow(1.0 + self.paper_e * pow(self.fog + x, -self.film_power), -self.paper_power)
        film = pow(self.fog + x, self.film_power)
        return self.magnitude * pow(film / (self.paper_e + film), self.paper_power)

    def apply(self, x):
        self.find_film_power()
        return self.__evaluate(x)
