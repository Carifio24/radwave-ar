import math
from os.path import extsep, join, splitext
import pandas as pd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from uuid import uuid4

from common import BEST_FIT_FILEPATH, N_BEST_FIT_POINTS, get_bounds, CLUSTER_FILEPATH, bring_into_clip, clip_linear_transformations, N_PHASES, N_POINTS, sample_around


# Overall configuration settings
SCALE = True
TRIM_GALAXY = True 
GAUSSIAN_POINTS = 6

sigma_val = 15 / math.sqrt(3)
if SCALE:
    sigma_val /= 1000

def unique_id():
    return uuid4().hex


def bounding_box(center, radius):
    return Vt.Vec3fArray(2, (Gf.Vec3f(*[c - radius for c in center]), Gf.Vec3f(*[c + radius for c in center])))


def get_positions(scale=False, clip_transforms=None):
    df = pd.read_csv(CLUSTER_FILEPATH)

    # We don't need the translations for the USDZ animation
    # but we need them to construct the future positions of the sampled points
    translations = { pt: [] for pt in range(N_POINTS) }
    initial_phase = df[df["phase"] == 0]
    initial_xyz = [-initial_phase["xc"], initial_phase["zc"] - 20.8, initial_phase["yc"]]
    initial_positions = [tuple(c[i] for c in initial_xyz) for i in range(N_POINTS)]
    sampled_positions = []
    for position in initial_positions:
        with_samples = list(sample_around(position, GAUSSIAN_POINTS, sigma_val)) + [position]
        sampled_positions.extend([list(x) for x in with_samples])

    for phase in range(N_PHASES + 1):
        slice = df[df["phase"] == phase % 360]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] *= -1
        xyz[1] -= 20.8
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        diffs = [c - pc for c, pc in zip(xyz, initial_xyz)]
        for pt in range(N_POINTS):
            translations[pt].append(tuple(x[pt] for x in diffs))

    positions = {}
    for pt, pos in enumerate(sampled_positions):
        original_index = pt // (GAUSSIAN_POINTS + 1)
        pt_translations = translations[original_index]
        positions[pt] = [tuple(tc + pc for tc, pc in zip(t, pos)) for t in pt_translations]
        
    return positions


def get_best_fit_positions(scale=False, clip_transforms=None):
    positions = { pt: [] for pt in range(N_BEST_FIT_POINTS) }
    df = pd.read_csv(BEST_FIT_FILEPATH)
    for phase in range(N_PHASES + 1):
        slice = df[df["phase"] == phase % 360]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] *= -1
        xyz[1] -= 20.8
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        for pt in range(N_BEST_FIT_POINTS):
            positions[pt].append(tuple(c[pt] for c in xyz))

    return positions


def add_sphere(stage, positions, timestamps, radius, material):
    xform_key = f"/world/xform_{unique_id()}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{unique_id()}"
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

radius = 0.005
best_fit_radius = 0.002
time_delta = 0.01
mins, maxes = get_bounds()
clip_transforms = clip_linear_transformations(list(zip(mins, maxes)))
timestamps = [time_delta * i for i in range(1, N_PHASES)]

point_positions = get_positions(scale=SCALE, clip_transforms=clip_transforms)
best_fit_positions = get_best_fit_positions(scale=SCALE, clip_transforms=clip_transforms)

# Set up the stage for our USD
output_filename = "radwave.usdc"
output_filepath = join(output_directory, output_filename)
stage = Usd.Stage.CreateNew(output_filepath)

default_prim = UsdGeom.Xform.Define(stage, "/world").GetPrim()
stage.SetDefaultPrim(default_prim)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

material = UsdShade.Material.Define(stage, "/material")
pbr_shader = UsdShade.Shader.Define(stage, "/material/PBRShader")
pbr_shader.CreateIdAttr("UsdPreviewSurface")
pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((31 / 255, 60 / 255, 241 / 255))
material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")

best_fit_material = UsdShade.Material.Define(stage, "/best_fit_material")
bf_pbr_shader = UsdShade.Shader.Define(stage, "/best_fit_material/PBRShader")
bf_pbr_shader.CreateIdAttr("UsdPreviewSurface")
bf_pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
bf_pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
bf_pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((132 / 155, 215 / 255, 245 / 255))
best_fit_material.CreateSurfaceOutput().ConnectToSource(bf_pbr_shader.ConnectableAPI(), "surface")


# Create a sphere for each point at phase=0
for index in range(len(point_positions)):
    positions = point_positions[index]
    add_sphere(stage, positions, timestamps, radius, material)

for index in range(len(best_fit_positions)):
    positions = best_fit_positions[index]
    add_sphere(stage, positions, timestamps, best_fit_radius, best_fit_material)

sun_position = [8121.97336612, 0., 0.]
sun_world_position = sun_position
if SCALE:
    sun_position_columns = [[c] for c in sun_position]
    sun_position_clip = bring_into_clip(sun_position_columns, clip_transforms)
    sun_position = [c[0] for c in sun_position_clip]



# Now we need to set up the galaxy image
galaxy_square_edge = 18_500
shift = sun_world_position[0]
shift_fraction = 0.5 * shift / galaxy_square_edge
if TRIM_GALAXY:
    galaxy_fraction = 0.2
    galaxy_image_edge = galaxy_fraction * galaxy_square_edge
else:
    galaxy_image_edge = galaxy_square_edge

galaxy_points = [
    [galaxy_image_edge, 0, galaxy_image_edge],
    [galaxy_image_edge, 0, -galaxy_image_edge],
    [-galaxy_image_edge, 0, -galaxy_image_edge],
    [-galaxy_image_edge, 0, galaxy_image_edge]
]
if TRIM_GALAXY:
    galaxy_points = [[p[0] + shift, p[1], p[2]] for p in galaxy_points]

# This is the transformation from world space -> galaxy texture space
# We determined that the galaxy image needs a 90 degree rotation
# and so this affine transformation accounts for that.
# It's easier if we do this before we scale
slope = 0.5 / galaxy_square_edge
intercept = slope * galaxy_square_edge
texcoord = lambda x, z: [(-0.5 / galaxy_square_edge) * z + 0.5, (0.5 / galaxy_square_edge) * x + 0.5]
galaxy_texcoords = [texcoord(p[0], p[2]) for p in galaxy_points]

if SCALE:
    galaxy_point_columns = [[c[i] for c in galaxy_points] for i in range(3)]
    galaxy_points_clip = bring_into_clip(galaxy_point_columns, clip_transforms)
    galaxy_points = [tuple(c[i] for c in galaxy_points_clip) for i in range(len(galaxy_points))]

galaxy_prim_key = "/world/galaxy"
galaxy_prim = stage.DefinePrim(galaxy_prim_key)
galaxy_mesh_key = f"{galaxy_prim_key}/mesh"
mesh = UsdGeom.Mesh.Define(stage, galaxy_mesh_key)
mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
mesh.CreatePointsAttr(galaxy_points)
mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(mesh.GetPointsAttr().Get()))
mesh.CreateFaceVertexCountsAttr([4])
mesh.CreateFaceVertexIndicesAttr([0,1,2,3])

    
stage.GetRootLayer().Save()

# Create a USDA file as well
path, _ = splitext(output_filepath)
stage.GetRootLayer().Export(f"{path}{extsep}usda")
