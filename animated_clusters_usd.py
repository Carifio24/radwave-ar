from os.path import extsep, join, splitext
from numpy import inf
import pandas as pd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from common import bring_into_clip, clip_linear_transformations, cluster_filepath, N_PHASES, N_POINTS


def bounding_box(center, radius):
    return Vt.Vec3fArray(2, (Gf.Vec3f(*[c - radius for c in center]), Gf.Vec3f(*[c + radius for c in center])))


def get_scaled_positions():
    positions = { pt: [] for pt in range(N_PHASES) }
    cmins = [inf, inf, inf]
    cmaxes = [-inf, -inf, -inf]
    dfs = []
    for phase in range(N_PHASES+1):
        df = pd.read_csv(cluster_filepath(phase))
        dfs.append(df)
        xyz = [df["x"], df["y"], df["z"]]
        for index, coord in enumerate(xyz):
            cmins[index] = min(cmins[index], min(coord))
            cmaxes[index] = max(cmaxes[index], max(coord))

    clip_transforms = clip_linear_transformations(list(zip(cmins, cmaxes)))
    for phase in range(N_PHASES+1):
        df = dfs[phase]
        xyz = [df["x"], df["y"], df["z"]]
        xyz = bring_into_clip(xyz, clip_transforms)
        for pt in range(df["x"].shape[0]):
            positions[pt].append(tuple(c[pt] for c in xyz))

    return positions


output_directory = "out"

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)
N_POINTS = initial_df.shape[0]
radius = 0.005
time_delta = 0.1
timestamps = [time_delta * i for i in range(1, N_PHASES)]

point_positions = get_scaled_positions()

# Set up the stage for our USD
output_filename = "radwave.usdc"
output_filepath = join(output_directory, output_filename)
stage = Usd.Stage.CreateNew(output_filepath)

default_prim = UsdGeom.Xform.Define(stage, "/world").GetPrim()
stage.SetDefaultPrim(default_prim)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

material = UsdShade.Material.Define(stage, "/material")
pbrShader = UsdShade.Shader.Define(stage, "/material/PBRShader")
pbrShader.CreateIdAttr("UsdPreviewSurface")
pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((31 / 255, 60 / 255, 241 / 255))
material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

# Create a sphere for each point at phase=0
for index in range(N_POINTS):

    positions = point_positions[index]

    xform_key = f"/world/xform_{index}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{index}"
    sphere = UsdGeom.Sphere.Define(stage, sphere_key)

    initial_position = positions[0]
    UsdGeom.XformCommonAPI(xform).SetTranslate(initial_position)
    
    extent_attr = sphere.GetExtentAttr()
    radius_attr = sphere.GetRadiusAttr()

    radius_attr.Set(radius)
    bbox = bounding_box(initial_position, radius)
    extent_attr.Set(bbox)

    sphere.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(sphere).Bind(material)

    translation = sphere.AddTranslateOp()
    translation.Set(initial_position)
    for i, time in enumerate(timestamps):
        translation.Set(time=time, value=positions[i])

stage.GetRootLayer().Save()

# Create a USDA file as well
path, _ = splitext(output_filepath)
stage.GetRootLayer().Export(f"{path}{extsep}usda")
