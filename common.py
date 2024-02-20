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

