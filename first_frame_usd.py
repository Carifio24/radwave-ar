from os.path import join
import pandas as pd
from pxr import Gf, Usd, UsdGeom, Vt

from common import *


def bounding_box(center, radius):
    return Vt.Vec3fArray(
            2,
            Gf.Vec3f(*[c - radius for c in center]),
            Gf.Vec3f(*[c + radius for c in center])
    )


output_directory = "out"

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)
N_POINTS = initial_df.shape[0]
radius = 0.005

positions, translations = get_scaled_positions_and_translations()

# Set up the stage for our USD
stage = Usd.Stage.CreateNew(join(output_directory, "radwave.usda"))

# Create a sphere for each point at phase=0
for index, point in enumerate(positions):

    xform_key = f"/xform_{index}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{index}"
    sphere = UsdGeom.Sphere.Define(stage, sphere_key)
    
    extent_attr = sphere.GetExtentAttr()
    radius_attr = sphere.GetRadiusAttr()

    radius_attr.Set(radius)
    bbox = bounding_box(point, radius)
    extent_attr.Set(bbox)



# xform_prim = UsdGeom.Xform.Define(stage, "/xform")
# sphere_prim = UsdGeom.Sphere.Define(stage, "/xform/sphere")

stage.GetRootLayer().Save()
