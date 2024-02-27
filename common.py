from itertools import product
import math
from os.path import join

import numpy as np
from numpy.random import multivariate_normal
import pandas as pd

__all__ = [
    "N_VISIBLE_PHASES", "N_PHASES", "clip_linear_transformations", "cluster_filepath", "scale"
]

N_VISIBLE_PHASES = 270
N_PHASES = 360 
output_directory = "out"


COORDINATES = "galactic"
CLUSTER_FILEPATH = join("data", "RW_cluster_oscillation_phase_updated_galactocentric.csv")

Y_ROTATION_ANGLE = 165 * math.pi / 180

BEST_FIT_FILEPATH = join("data", f"RW_best_fit_oscillation_phase_galactocentric.csv")
BEST_FIT_DOWNSAMPLED_FILEPATH = join("data", f"RW_best_fit_oscillation_phase_{COORDINATES}_downsampled.csv")

N_POINTS = 89
N_BEST_FIT_POINTS = 1500

def rotate_y_nparrays(xyz, theta):
    x, y, z = xyz
    x_rot = math.cos(theta) * x + math.sin(theta) * z
    z_rot = -math.sin(theta) * x + math.cos(theta) * z
    return [x_rot, y, z_rot]

def rotate_y_list(xyz, theta):
    return [[math.cos(theta) * p[0] + math.sin(theta) * p[2], p[1], -math.sin(theta) * p[0] + math.cos(theta) * p[2]] for p in xyz]

# theta is the azimuthal angle here. Sorry math folks.
# This gives a straightforward "grid"-style triangulation of a sphere with the given center and radius,
# with tunable resolutions in theta and phi.
def sphere_mesh(center, radius, theta_resolution=5, phi_resolution=5):
    nonpole_thetas = [i * math.pi / theta_resolution for i in range(1, theta_resolution-1)]
    phis = [i * 2 * math.pi / phi_resolution for i in range(phi_resolution)]
    points = [(
        center[0] + radius * math.cos(phi) * math.sin(theta),
        center[1] + radius * math.sin(phi) * math.sin(theta),
        center[2] + radius * math.cos(theta)
    ) for theta, phi in product(nonpole_thetas, phis)]
    points = [(center[0], center[1], center[2] + radius)] + points + [(center[0], center[1], center[2] - radius)]

    # TODO: Make a cleaner way to handle "modular" aspect of rows
    # Idea: Make column = column % phi_resolution in `sphere_mesh_index` ?
    triangles = [(int(0), i + 1, i) for i in range(1, phi_resolution)]
    tr, pr = theta_resolution, phi_resolution
    triangles.append((0, 1, theta_resolution))
    for row in range(1, theta_resolution - 2):
        for col in range(phi_resolution):
            rc_index = sphere_mesh_index(row, col, tr, pr)
            triangles.append((rc_index, sphere_mesh_index(row+1, col, tr, pr), sphere_mesh_index(row+1, col-1, tr, pr)))
            triangles.append((rc_index, sphere_mesh_index(row, col+1, tr, pr), sphere_mesh_index(row+1, col, tr, pr)))
        triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row+1, pr-1, tr, pr), sphere_mesh_index(row+1, pr-2, tr, pr)))
        triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row, 0, tr, pr), sphere_mesh_index(row+1, pr-1, tr, pr)))
        
    row = theta_resolution - 2
    last_index = sphere_mesh_index(theta_resolution - 1, 0, tr, pr)
    for col in range(phi_resolution-1):
        triangles.append((sphere_mesh_index(row, col, tr, pr), sphere_mesh_index(row, col+1, tr, pr), last_index))
    triangles.append((sphere_mesh_index(row, pr-1, tr, pr), sphere_mesh_index(row, 0, tr, pr), last_index))

    return points, triangles


def sphere_mesh_index(row, column, theta_resolution, phi_resolution):
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        return phi_resolution * (row - 1) + column + 1


def sample_around(point, n, sigma_val):
    sigma = np.array([sigma_val] * 3)
    cov = np.diag(sigma ** 2)
    return multivariate_normal(mean=point, cov=cov, size=n)


def get_bounds():
    mins = [np.inf, np.inf, np.inf]
    maxes = [-np.inf, -np.inf, -np.inf]
    cluster_df = pd.read_csv(CLUSTER_FILEPATH)
    best_fit_filepath = BEST_FIT_FILEPATH
    best_fit_df = pd.read_csv(best_fit_filepath)
    for phase in range(N_PHASES):
        for df in (cluster_df, best_fit_df):
            slice = df[df["phase"] == phase]
            xyz = [slice[c] for c in ["xc", "zc", "yc"]]
            xyz[0] *= -1
            xyz[1] -= 20.8
            xyz = rotate_y_nparrays(xyz, Y_ROTATION_ANGLE)
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


def clip_linear_transformations(bounds, clip_size=1):
    ranges = [abs(bds[1] - bds[0]) for bds in bounds]
    max_range = max(ranges)
    line_data = []
    for bds, rg in zip(bounds, ranges):
        frac = rg / max_range
        half_frac = frac / 2
        half_target = clip_size * half_frac
        line_data.append(slope_intercept_between((bds[0], -half_target), (bds[1], half_target)))
    return line_data


def bring_into_clip(data, transforms):
    return np.array([[m * d + b for d in data[idx]] for idx, (m, b) in enumerate(transforms)])

