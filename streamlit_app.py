import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tonecurves

### Sidebar stuff
st.sidebar.header("Scene Settings")
view_range = st.sidebar.slider('View Range', min_value=-16.0, max_value=16.0,value=[-6.0, 6.0])
view_mode = st.sidebar.selectbox('View Scaling', ["EV (log2)", "Linear", "Log2Log2"])
if not view_mode == "Log2Log2":
    view_dydx_mode = st.sidebar.selectbox('Slope Base', ["EV (log2)", "Linear"])
else:
    view_dydx_mode = "EV (log2)"
view_resolution = st.sidebar.slider('View Resolution', min_value=100, max_value=10000,value=400)

st.sidebar.header("Display Settings")
display_black = pow(2.0, st.sidebar.slider("Target Black Luminance", -16.0, -5.0, -13.0))
display_white = st.sidebar.slider("Target White Luminance", 0.2, 16.0, 1.0)

st.sidebar.header("Log-Logistic Settings")
loglogistic = tonecurves.LogLogistic()
loglogistic.enable = st.sidebar.checkbox('Show Log-Logistic', value=True)
if loglogistic.enable:
    loglogistic.contrast = st.sidebar.slider("Contrast", 0.5, 5.0, loglogistic.contrast)
    loglogistic.skew = st.sidebar.slider("Skew", -2.0, 1.0, loglogistic.skew)
    loglogistic.display_white = display_white
    loglogistic.display_black = display_black

st.sidebar.header("Filmic Settings")
filmic = tonecurves.Filmic()
filmic.enable = st.sidebar.checkbox('Show Filmic', value=True)
if filmic.enable:
    filmic.scene_white = st.sidebar.slider("White Relative Exposure", 2.0, 8.0, filmic.scene_white)
    filmic.scene_black = st.sidebar.slider("Black Relative Exposure", -14.0, -3.0, filmic.scene_black)
    filmic.contrast = st.sidebar.slider("Contrast", 1.0, 2.0, filmic.contrast)

    if st.sidebar.checkbox('Auto Adjust Hardness', value=True):
        filmic.hardness = np.log(filmic.display_grey) / np.log(-filmic.scene_black / (filmic.scene_white - filmic.scene_black))
        st.sidebar.code(str(filmic.hardness))
    else:
        filmic.hardness = st.sidebar.slider("Hardness", 1.0, 10.0, filmic.hardness)
        

    filmic.latitude = st.sidebar.slider("Latitude", 5.0, 50.0, 100.0*filmic.latitude) / 100.0
    filmic.balance = st.sidebar.slider("Shadows/Highlights Balance", -50.0, 50.0, 100.0*filmic.balance) / 100.0
    filmic.display_black = display_black
    filmic.display_white = display_white

st.sidebar.header("Other Curves")
# Double Logistic
double_logistic = tonecurves.DoubleLogistic()
double_logistic.enable = st.sidebar.checkbox('Show Double Logistic', value=False)
if double_logistic.enable:
    double_logistic.L = st.sidebar.slider("L", 0.0, 2.0, 0.1845)
    double_logistic.c = st.sidebar.slider("c", 1.0, 10.0, 2.0)

# Ask if other curves should be displayed
show_base_curve = st.sidebar.checkbox('Show average basecurve', value=False)
show_aces_srgb = st.sidebar.checkbox('Show ACES sRGB 100 nits', value=False)
show_aces_hlg = st.sidebar.checkbox('Show ACES HLG 1000 nits', value=False)
show_analog_print = st.sidebar.checkbox('Show Analog Print', value=True)
show_analog_negative = st.sidebar.checkbox('Show inverted Analog Negative', value=False)
blender_names = ['None', 'Very Low Contrast', 'Low Contrast', 'Medium Low Contrast', "Medium Contrast", "Medium High Contrast", "High Contrast", "Very High Contrast"]
show_blender_filmic = st.sidebar.selectbox("Show Blender Filmic Curve", blender_names)

### Calculations
# Setup x axis for the plots
@st.cache
def gen_xaxis(mode, vrange, resolution):
    if mode == "Linear":
        return tonecurves.scene_light_linear(vrange, mode, resolution)
    else:
        return tonecurves.scene_light_log(vrange, resolution)
view_x_axis = gen_xaxis(view_mode, view_range, view_resolution)
value_plot = pd.DataFrame(index=view_x_axis)
slope_plot = pd.DataFrame(index=view_x_axis)

intensity_x_axis = tonecurves.scene_light_linear(view_range, view_mode, view_resolution)
if view_dydx_mode == "Linear":
    dydx_x_axis = intensity_x_axis
else:
    dydx_x_axis = np.log2(intensity_x_axis)
log_epsilon = pow(2.0, -16)

@st.cache
def derivative(x, y):
    dydx = np.zeros_like(y)
    dydx[1:] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dydx[0] = np.nan
    return dydx

# Add Loglogistic to the DataFrame
if loglogistic.enable:
    loglogistic_value = loglogistic.apply(intensity_x_axis)
    if view_mode == "Log2Log2":
        loglogistic_value = np.log2(loglogistic_value + log_epsilon)
    loglogistic_slope = derivative(dydx_x_axis, loglogistic_value)
    value_plot["LogLogistic"] = loglogistic_value
    slope_plot["LogLogistic"] = loglogistic_slope

# Add Filmic to the DataFrame
if filmic.enable:
    filmic_value = filmic.apply(intensity_x_axis)
    if view_mode == "Log2Log2":
        filmic_value = np.log2(filmic_value + log_epsilon)
    filmic_slope = derivative(dydx_x_axis, filmic_value)
    value_plot["darktable Filmic"] = filmic_value
    slope_plot["darktable Filmic"] = filmic_slope

    # For the filmic default view
    norm_x, pregamma_y = filmic.get_default_view(view_resolution)
    filmic_plot = pd.DataFrame(index=norm_x)
    filmic_plot["darktable Filmic"] = pregamma_y

    darktable_filmic_fig = plt.figure("Filmic in darktable")
    plt.plot(norm_x, pregamma_y)
    plt.plot(0.0, filmic.display_black_gamma, 'o')
    plt.plot(filmic.toe_log, filmic.toe_display, 'o')
    plt.plot(filmic.grey_log, filmic.display_grey_gamma, 'o')
    plt.plot(filmic.shoulder_log, filmic.shoulder_display, 'o')
    plt.plot(1.0, filmic.display_white_gamma, 'o')

if double_logistic.enable:
    double_logistic_value = double_logistic.apply(intensity_x_axis)
    if view_mode == "Log2Log2":
        double_logistic_value = np.log2(double_logistic_value + log_epsilon)
    double_logistic_slope = derivative(dydx_x_axis, double_logistic_value)
    value_plot["DoubleLogistic"] = double_logistic_value
    slope_plot["DoubleLogistic"] = double_logistic_slope

# Add Base Average Base Curve if requested
@st.cache
def get_base_curve(mode, intensity):
    base_curve = tonecurves.BaseCurve()
    value = base_curve.get_average_curve(intensity)
    slope = derivative(intensity, value)
    return value, slope

if show_base_curve:
    base_display, base_slope = get_base_curve(view_mode, intensity_x_axis)
    if view_mode == "Log2Log2":
        base_display = np.log2(base_display + log_epsilon)
    base_slope = derivative(dydx_x_axis, base_display)
    value_plot["Average Base Curve"] = base_display
    slope_plot["Average Base Curve"] = base_slope

# Add aces curve if requested
if show_aces_srgb or show_aces_hlg:
    aces = tonecurves.Aces()

if show_aces_srgb:
    aces_srgb_display = aces.apply_srgb(intensity_x_axis)
    if view_mode == "Log2Log2":
        aces_srgb_display = np.log2(aces_srgb_display + log_epsilon)
    aces_srgb_slope = derivative(dydx_x_axis, aces_srgb_display)
    value_plot["ACES sRGB"] = aces_srgb_display
    slope_plot["ACES sRGB"] = aces_srgb_slope

if show_aces_hlg:
    aces_hlg_display = aces.apply_hlg(intensity_x_axis)
    if view_mode == "Log2Log2":
        aces_hlg_display = np.log2(aces_hlg_display + log_epsilon)
    aces_hlg_slope = derivative(dydx_x_axis, aces_hlg_display)
    value_plot["ACES HLG"] = aces_hlg_display
    slope_plot["ACES HLG"] = aces_hlg_slope

# Add aces curve if requested
if show_analog_print or show_analog_negative:
    analog = tonecurves.AnalogFilm()

if show_analog_print:
    analog_print_display = analog.apply_print(intensity_x_axis)
    if view_mode == "Log2Log2":
        analog_print_display = np.log2(analog_print_display + log_epsilon)
    analog_print_slope = derivative(dydx_x_axis, analog_print_display)
    value_plot["Analog Print"] = analog_print_display
    slope_plot["Analog Print"] = analog_print_slope

if show_analog_negative:
    analog_negative_display = analog.apply_negative(intensity_x_axis)
    if view_mode == "Log2Log2":
        analog_negative_display = np.log2(analog_negative_display + log_epsilon)
    analog_negative_slope = derivative(dydx_x_axis, analog_negative_display)
    value_plot["Analog Negative"] = analog_negative_display
    slope_plot["Analog Negative"] = analog_negative_slope

# Add Blender filmic curve if requested
@st.cache
def get_blender_curve(curve_name):
    blender = tonecurves.Blender()
    return blender.apply(intensity_x_axis, curve_name)

if not show_blender_filmic == "None":
    blender_display = get_blender_curve(show_blender_filmic)
    if view_mode == "Log2Log2":
        blender_display = np.log2(blender_display + log_epsilon)
    blender_slope = derivative(dydx_x_axis, blender_display)
    value_plot[show_blender_filmic] = blender_display
    slope_plot[show_blender_filmic] = blender_slope


### Main window stuff
st.title("Tone Curve Playground")
st.write("""Welcome to this simple 1D plaground for Scene to Display mapping Tone Curves.
It's my personal attempt at getting a better understanding of the curves used in darktable.
I hope some you will find it interesting as well!""")
st.header("Usage")
st.write("""The view range on the left is for the x-axis luminance range, which is defined in EV relative to
middle grey, fixed at 0.1845. The view scaling is only used for scaling the plots. However all calculations are always
done on the linear light intensity. The basis for calculation of the the derivative can be picked as linear or EV
indenpendetly of the view scaling. Log2Log2 forces log2 basis for the derivative.

Display settings are the same as the display tab in the filmic module with 1.0=100%.
The defualt values of 0 and 1 equals the range of a normal computer screen. A HLG HDR screen will typically have a target white luminance = 12.
An example of output medium that wants a smaller range than a SDR screen are prints on paper.

The Log-Logistic curve is modified to fulfill the following conditions:
* f(0) = Target Black Luminance
* f(0.1845) = 0.1845
* f(inf) = Target White Luminance
* The contrast setting is the power of the function.
* Skew works pushes the peak contrast towards shadows or highlights, the slope at middle grey remains constant.

The filmic settings are the same as in darktable 3.4, please check out available documentation for more info.
""")
st.header("Value Graph")
st.write("""X-axis scene intensity for the applied color channel, typically R, G, B, or Luminance.
The Y-axis is the display luminance that the x value maps to.

What to look for:
* Does it converge to the target luminances?
* How does it converge?
* Does it go outside of the target luminances?
""")
st.line_chart(value_plot)

st.header("Slope Graph")
st.line_chart(slope_plot)
st.write("""
In my eyes the slope graph is more interesting than the value graph. It shows the derivative (dy/dx) of the value plot with dx selectable as log2 or linear.
The best way to describe the visual impact of the slope is image contrast, higher slope means higher contrast. A different name for this graph could be Contrast Graph.

What to look for:
* Number of peaks, a good curve will have only one peak (I do not have any references for this yet but I'm feeling reasonably certain about this).
* Position of the maximum of the slope, the maximum of the slope graph is also the luminance level at which the highest image contrast will be.
* Tails, we want a smooth convergence towards display black and display white. Longer tails will yield smoother but potentially duller transitions.
* Negative values! A no, no, no! These curves has to always be positive as these transfer curves are supposed to be monotonic.
* When does the slope reach zero? These points marks the dynamic range of that particular curve.
""")

if filmic.enable:
    st.header("About the Filmic curve")
    st.subheader("Why does it look different from the darktable plot?")
    st.write("""The default view of the filmic curve in darktable looks different to this but
    is very useful for finding a suitable curve. It does however hide the actual effect of changing
    black and white point as well as what the hardness factor does to the end result.
    I found the hardness factor particulary hard to understand. It seems to act like
    a normalizer for the middle grey point which is a crucial part of expanding the space of usable splines.
    The default darktable view is shown below for reference. Note that this view is on a normalized x axis
    mapping [black_point, white_point] -> [0, 1] and the y axis is shown as before the hardness function is applied.""")


    st.pyplot(darktable_filmic_fig)
    st.subheader("Disclaimer on the filmic curve")
    st.write("""
    The Filmic curve implementation is based on the documentation provided here:
    https://eng.aurelienpierre.com/2018/11/filmic-darktable-and-the-quest-of-the-hdr-tone-mapping/#filmic_s_curve
    and porting of the darktable C-code. Errors might have been made, I appoligize for that in advance.

    The following modification have been done to make it behave like the other options:
    The output clamp before applying the hardness/gamma power function has been removed.
    It would not be possible to reach values larger than 1.0 otherwise.
    """)
