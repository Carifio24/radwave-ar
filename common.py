from os.path import join

import numpy as np
import pandas as pd

__all__ = [
    "N_VISIBLE_PHASES", "N_PHASES", "clip_linear_transformations", "cluster_filepath", "scale"
]

N_VISIBLE_PHASES = 270
N_PHASES = 720
output_directory = "out"


COORDINATES = "galactic"
CLUSTER_FILEPATH = join("data", "RW_cluster_oscillation_phase_updated_galactocentric.csv")


BEST_FIT_FILEPATH = join("data", f"RW_best_fit_oscillation_phase_galactocentric.csv")
BEST_FIT_DOWNSAMPLED_FILEPATH = join("data", f"RW_best_fit_oscillation_phase_{COORDINATES}_downsampled.csv")

N_POINTS = 89
N_BEST_FIT_POINTS = 1500


def get_bounds():
    mins = [np.inf, np.inf, np.inf]
    maxes = [-np.inf, -np.inf, -np.inf]
    cluster_df = pd.read_csv(CLUSTER_FILEPATH)
    best_fit_filepath = BEST_FIT_FILEPATH
    best_fit_df = pd.read_csv(best_fit_filepath)
    for phase in range(N_VISIBLE_PHASES + 1):
        for df in (cluster_df, best_fit_df):
            slice = df[df["phase"] == phase]
            xyz = [slice[c] for c in ["xc", "zc", "yc"]]
            xyz[0] *= -1
            xyz[1] -= 20.8
            for index, coord in enumerate(xyz):
                mins[index] = min(mins[index], min(coord))
                maxes[index] = max(maxes[index], max(coord))
    
    return mins, maxes


def scale(value, lower, upper):
    return (value - lower) / (upper - lower)


def slope_intercept_between(a, b):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = b[1] - slope * b[0]
    return slope, intercept


def clip_linear_transformations(bounds):
    ranges = [abs(bds[1] - bds[0]) for bds in bounds]
    max_range = max(ranges)
    line_data = []
    for bds, rg in zip(bounds, ranges):
        frac = rg / max_range
        half_frac = frac / 2
        line_data.append(slope_intercept_between((bds[0], -half_frac), (bds[1], half_frac)))
    return line_data


def bring_into_clip(data, transforms):
    return np.array([[m * d + b for d in data[idx]] for idx, (m, b) in enumerate(transforms)])

