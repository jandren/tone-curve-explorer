import numpy as np


class FilmicSettings(object):
    def __init__(self):
        ### Settings
        self.scene_white = 4.0
        self.scene_black = -8.0

        self.display_white = 1.0
        self.display_black = 0.0152 / 100.0
        
        self.scene_grey = 0.1845
        self.display_grey = 0.1845

        self.contrast = 1.35
        self.latitude = 0.25
        self.balance = 0.0
        self.hardness = 2.2

class Filmic(object):
    def __init__(self):
        self.__transfer_settings(FilmicSettings())

    def __transfer_settings(self, settings):
        self.scene_white = settings.scene_white
        self.scene_black = settings.scene_black

        self.display_white = settings.display_white
        self.display_black = settings.display_black
        
        self.scene_grey = settings.scene_grey
        self.display_grey = settings.display_grey

        self.contrast = settings.contrast
        self.latitude = settings.latitude
        self.balance = settings.balance
        self.hardness = settings.hardness

    def __log_tonemapping_v2(self, x):
        dynamic_range = self.scene_white - self.scene_black
        return np.clip((np.log2(x / self.scene_grey) - self.scene_black) / dynamic_range, 0.0, 1.0)

    def __fit_splines(self):
        self.output_power = self.hardness #np.log(self.display_grey) / np.log(-self.scene_black / (self.scene_white - self.scene_black))
        #print("Gamma: ", self.output_power)
        self.display_grey_gamma = pow(self.display_grey, 1.0 / self.output_power)
    
        self.display_black_gamma = pow(np.clip(self.display_black, 0.0, self.display_grey), 1.0 / (self.output_power))
        self.display_white_gamma = pow(max(self.display_white, self.display_grey), 1.0 / (self.output_power))

        dynamic_range = (self.scene_white - self.scene_black)
        self.grey_log = abs(self.scene_black) / dynamic_range
        
        # nodes for mapping from log encoding to desired target luminance
        # X coordinates
        self.toe_log = self.grey_log - self.latitude * abs(self.scene_black / dynamic_range)
        self.shoulder_log = self.grey_log + self.latitude * abs(self.scene_white / dynamic_range)

        # interception
        linear_intercept = self.display_grey_gamma - (self.contrast * self.grey_log)

        # y coordinates
        self.toe_display = (self.toe_log * self.contrast + linear_intercept)
        self.shoulder_display = (self.shoulder_log * self.contrast + linear_intercept)

        # Apply the highlights/shadows balance as a shift along the contrast slope
        norm = np.sqrt(self.contrast * self.contrast + 1.0)

        # negative values drag to the left and compress the shadows, on the UI negative is the inverse
        coeff = -(2.0 * self.latitude) * self.balance
        self.toe_display += coeff * self.contrast / norm
        self.shoulder_display += coeff * self.contrast / norm
        self.toe_log += coeff / norm
        self.shoulder_log += coeff / norm

        self.latitude_min = self.toe_log
        self.latitude_max = self.shoulder_log

        """
        * Now we have 3 segments :
        *  - x = [0.0 ; toe_log], curved part
        *  - x = [toe_log ; grey_log ; shoulder_log], linear part
        *  - x = [shoulder_log ; 1.0] curved part
        *
        * BUT : in case some nodes overlap, we need to remove them to avoid
        * degenerating of the curve
        **/

        // Build the curve from the nodes
        spline->x[0] = black_log;
        spline->x[1] = toe_log;
        spline->x[2] = grey_log;
        spline->x[3] = shoulder_log;
        spline->x[4] = white_log;

        spline->y[0] = black_display;
        spline->y[1] = toe_display;
        spline->y[2] = grey_display;
        spline->y[3] = shoulder_display;
        spline->y[4] = white_display;

        spline->latitude_min = spline->x[1];
        spline->latitude_max = spline->x[3];

        /**
        * For background and details, see :
        * https://eng.aurelienpierre.com/2018/11/30/filmic-darktable-and-the-quest-of-the-hdr-tone-mapping/#filmic_s_curve
        *
        **/"""
        Tl = self.toe_log
        Tl2 = Tl * Tl
        Tl3 = Tl2 * Tl
        Tl4 = Tl3 * Tl

        Sl = self.shoulder_log
        Sl2 = Sl * Sl
        Sl3 = Sl2 * Sl
        Sl4 = Sl3 * Sl

        """
        // Each polynomial is following the same structure :
        // y = M5 * x⁴ + M4 * x³ + M3 * x² + M2 * x¹ + M1 * x⁰
        // We then compute M1 to M5 coeffs using the imposed conditions over the curve.
        // M1 to M5 are 3×1 vectors, where each element belongs to a part of the curve.
        """
        # solve the linear central part - affine function
        self.lat_poly = np.poly1d([self.contrast, self.toe_display - self.contrast * self.toe_log])

        # solve the toe part
        # fourth order polynom - only mode in darktable 3.0.0
        toe_A = np.array([[0.,        0.,       0.,      0., 1.],   # position in 0
                          [0.,        0.,       0.,      1., 0.],   # first derivative in 0
                          [Tl4,       Tl3,      Tl2,     Tl, 1.],   # position at toe node
                          [4. * Tl3,  3. * Tl2, 2. * Tl, 1., 0.],   # first derivative at toe node
                          [12. * Tl2, 6. * Tl,  2.,      0., 0.]])  # second derivative at toe node

        toe_b = np.array([self.display_black_gamma, 0., self.toe_display, self.contrast, 0. ])
        self.toe_poly = np.poly1d(np.linalg.solve(toe_A, toe_b))

        # solve the shoulder part
        # fourth order polynom - only mode in darktable 3.0.0
        shoulder_A = np.array([[1.,        1.,       1.,      1., 1.],   # position in 1
                               [4.,        3.,       2.,      1., 0.],   # first derivative in 1
                               [Sl4,       Sl3,      Sl2,     Sl, 1.],   # position at shoulder node
                               [4. * Sl3,  3. * Sl2, 2. * Sl, 1., 0.],   # first derivative at shoulder node
                               [12. * Sl2, 6. * Sl,  2.,      0., 0.]])  # second derivative at shoulder node

        shoulder_b = np.array([self.display_white_gamma, 0., self.shoulder_display, self.contrast, 0. ])
        self.shoulder_poly = np.poly1d(np.linalg.solve(shoulder_A, shoulder_b))

    def apply(self, x, settings):
        self.__transfer_settings(settings)
        self.__fit_splines()
        x_log = self.__log_tonemapping_v2(x)
        output = np.polyval(self.lat_poly, x_log)

        # Replace Toe
        mask = x_log < self.latitude_min
        output[mask] = np.polyval(self.toe_poly, x_log[mask])

        # Replace Shoulder
        mask = x_log > self.latitude_max
        output[mask] = np.polyval(self.shoulder_poly, x_log[mask])

        display_value = pow(output, self.output_power)
        display_slope = np.zeros_like(x)
        display_slope[1:] = (display_value[1:] - display_value[:-1])# / (x[1:] - x[:-1])
        return display_value
    
    def get_default_view(self, settings, n_points):
        grey_log = np.log2(self.scene_grey)
        x = pow(2.0, np.linspace(self.scene_black + grey_log, self.scene_white + grey_log, n_points))
        x_log = self.__log_tonemapping_v2(x)
        output = np.polyval(self.lat_poly, x_log)

        # Replace Toe
        mask = x_log < self.latitude_min
        output[mask] = np.polyval(self.toe_poly, x_log[mask])

        # Replace Shoulder
        mask = x_log > self.latitude_max
        output[mask] = np.polyval(self.shoulder_poly, x_log[mask])
        return x_log, output