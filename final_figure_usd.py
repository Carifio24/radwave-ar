from os.path import extsep, join, splitext
import pandas as pd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from uuid import uuid4

from common import get_bounds, CLUSTER_FILEPATH, bring_into_clip, clip_linear_transformations, N_PHASES, N_POINTS


# Overall configuration settings
SCALE = True


def bounding_box(center, radius):
    return Vt.Vec3fArray(2, (Gf.Vec3f(*[c - radius for c in center]), Gf.Vec3f(*[c + radius for c in center])))


def get_positions(scale=False, clip_transforms=None):
    positions = { pt: [] for pt in range(N_PHASES) }
    df = pd.read_csv(CLUSTER_FILEPATH)
    for phase in range(N_PHASES + 1):
        slice = df[df["phase"] == phase % 360]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] *= -1
        xyz[1] -= 20.8
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        for pt in range(N_POINTS):
            positions[pt].append(tuple(c[pt] for c in xyz))

    return positions


def add_sphere(stage, positions, timestamps, radius, material):
    xform_key = f"/world/xform_{uuid4()}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{uuid4()}"
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
    for time, position in zip(timestamps, positions):
        translation.Set(time=time, value=position)


output_directory = "out"

radius = 0.01
time_delta = 0.1
mins, maxes = get_bounds()
clip_transforms = clip_linear_transformations(list(zip(mins, maxes)))
timestamps = [time_delta * i for i in range(1, N_PHASES)]

point_positions = get_positions(scale=SCALE)

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
