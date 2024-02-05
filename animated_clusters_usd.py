from os.path import join
from numpy import inf
import pandas as pd
from pxr import Gf, Usd, UsdGeom, Vt

from common import *


def bounding_box(center, radius):
    return Vt.Vec3fArray(2, (Gf.Vec3f(*[c - radius for c in center]), Gf.Vec3f(*[c + radius for c in center])))


def get_scaled_positions():
    positions = { pt: [] for pt in range(N_PHASES) }
    cmin = inf
    cmax = -inf
    dfs = []
    for phase in range(N_PHASES+1):
        df = pd.read_csv(cluster_filepath(phase))
        dfs.append(df)
        xyz = [df["x"], df["y"], df["z"]]
        for coord in xyz:
            cmin = min(cmin, min(coord))
            cmax = max(cmax, max(coord))

    for phase in range(N_PHASES+1):
        df = dfs[phase]
        xyz = [df["x"], df["y"], df["z"]]
        xyz = [scale(c, cmin, cmax) for c in xyz]
        for pt in range(df["x"].shape[0]):
            positions[pt].append(tuple(c[pt] for c in xyz))

    return positions


output_directory = "out"

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)
N_POINTS = initial_df.shape[0]
radius = 0.005
time_delta = 0.01
timestamps = [time_delta * i for i in range(1, N_PHASES)]

point_positions = get_scaled_positions()

# Set up the stage for our USD
stage = Usd.Stage.CreateNew(join(output_directory, "radwave.usdc"))

# Create a sphere for each point at phase=0
for index in range(N_POINTS):

    positions = point_positions[index]

    xform_key = f"/xform_{index}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{index}"
    sphere = UsdGeom.Sphere.Define(stage, sphere_key)

    initial_position = positions[0]
    UsdGeom.XformCommonAPI(xform).SetTranslate(initial_position)
    
    extent_attr = sphere.GetExtentAttr()
    radius_attr = sphere.GetRadiusAttr()
    color_attr = sphere.GetDisplayColorAttr()

    radius_attr.Set(radius)
    bbox = bounding_box(initial_position, radius)
    extent_attr.Set(bbox)
    color_attr.Set([(31 / 255, 60 / 255, 241 / 255)])

    translation = sphere.AddTranslateOp()
    translation.set(initial_position)
    for i, time in enumerate(timestamps):
        translation.Set(time=time, value=positions[i])

stage.GetRootLayer().Save()
