import numpy as np

SCENE_GREY = 0.1845

def scene_light_linear(ev_range, view_mode, n_points):
    if view_mode == "Linear":
        return np.linspace(pow(2, ev_range[0] + np.log2(SCENE_GREY)), pow(2, ev_range[1] + np.log2(SCENE_GREY)), n_points)
    else:
        return pow(2.0, scene_light_log(ev_range, n_points) + np.log2(SCENE_GREY))

def scene_light_log(ev_range, n_points):
    return np.linspace(ev_range[0], ev_range[1], n_points)