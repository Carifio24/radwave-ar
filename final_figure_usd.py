import math
from os import getcwd
from os.path import extsep, join, splitext
import pandas as pd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from uuid import uuid4

from common import BEST_FIT_FILEPATH, N_BEST_FIT_POINTS, get_bounds, CLUSTER_FILEPATH, bring_into_clip, clip_linear_transformations, N_POINTS, sample_around, sphere_mesh, Y_ROTATION_ANGLE, rotate_y_list, rotate_y_nparrays

N_PHASES = 360

# Overall configuration settings
SCALE = True 
CLIP_SIZE = 30
TRIM_GALAXY = True
GALAXY_FRACTION = 0.09
GAUSSIAN_POINTS = 6
BEST_FIT_DOWNSAMPLE_FACTOR = 2

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
    initial_xyz = rotate_y_nparrays(initial_xyz, Y_ROTATION_ANGLE)
    if scale:
        initial_xyz = bring_into_clip(initial_xyz, clip_transforms)
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
        xyz = rotate_y_nparrays(xyz, Y_ROTATION_ANGLE)
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
    positions = { pt: [] for pt in range(N_BEST_FIT_POINTS // BEST_FIT_DOWNSAMPLE_FACTOR) }
    df = pd.read_csv(BEST_FIT_FILEPATH)
    for phase in range(N_PHASES + 1):
        slice = df[df["phase"] == phase % 360][::BEST_FIT_DOWNSAMPLE_FACTOR]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] *= -1
        xyz[1] -= 20.8
        xyz = rotate_y_nparrays(xyz, Y_ROTATION_ANGLE)
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        for pt in range(N_BEST_FIT_POINTS // BEST_FIT_DOWNSAMPLE_FACTOR):
            positions[pt].append(tuple(c[pt] for c in xyz))

    return positions


all_triangles = 0
def add_sphere(stage, positions, timestamps, radius, material, theta_resolution=5, phi_resolution=5):
    global all_triangles
    initial_position = positions[0]
    points, triangles = sphere_mesh(initial_position, radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    all_triangles += len(triangles)

    xform_key = f"{default_prim_key}/xform_{unique_id()}"
    xform = UsdGeom.Xform.Define(stage, xform_key)
    sphere_key = f"{xform_key}/sphere_{unique_id()}"
    sphere = UsdGeom.Mesh.Define(stage, sphere_key)
    sphere.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    sphere.CreatePointsAttr(points)
    sphere.CreateExtentAttr(bounding_box(initial_position, radius))
    sphere.CreateFaceVertexCountsAttr([3] * len(triangles))
    sphere.CreateFaceVertexIndicesAttr([idx for tri in triangles for idx in tri])

    sphere.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(sphere).Bind(material)

    translation = sphere.AddTranslateOp()
    for time, position in zip(timestamps, positions):
        delta = tuple(p - i for i, p in zip(initial_position, position))
        translation.Set(time=time, value=delta)

cwd = getcwd()
output_directory = join(cwd, "out")

radius = 1.75 * CLIP_SIZE * (0.005 if SCALE else 5)
best_fit_radius = 2 * CLIP_SIZE * math.sqrt(BEST_FIT_DOWNSAMPLE_FACTOR) * (0.0005 if SCALE else 0.5)
time_delta = 0.2
mins, maxes = get_bounds()
clip_transforms = clip_linear_transformations(list(zip(mins, maxes)), clip_size=CLIP_SIZE)
timestamps = [time_delta * i for i in range(1, N_PHASES)]

point_positions = get_positions(scale=SCALE, clip_transforms=clip_transforms)
best_fit_positions = get_best_fit_positions(scale=SCALE, clip_transforms=clip_transforms)

# Set up the stage for our USD
# Note that, just like with glTF, the default is that +y is up
output_filename = "radwave.usdc"
output_filepath = join(output_directory, output_filename)
stage = Usd.Stage.CreateNew(output_filepath)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

default_prim_key = "/world"
default_prim = UsdGeom.Xform.Define(stage, default_prim_key).GetPrim()
stage.SetDefaultPrim(default_prim)

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

sun_material = UsdShade.Material.Define(stage, "/sun_material")
sun_pbr_shader = UsdShade.Shader.Define(stage, "/sun_material/PBRShader")
sun_pbr_shader.CreateIdAttr("UsdPreviewSurface")
sun_pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
sun_pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
sun_pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((255/ 155, 255 / 255, 10 / 255))
sun_material.CreateSurfaceOutput().ConnectToSource(sun_pbr_shader.ConnectableAPI(), "surface")


# Create a sphere for each point at phase=0
for index in range(len(point_positions)):
    positions = point_positions[index]
    add_sphere(stage, positions, timestamps, radius, material, theta_resolution=8, phi_resolution=12)

for index in range(len(best_fit_positions)):
    positions = best_fit_positions[index]
    add_sphere(stage, positions, timestamps, best_fit_radius, best_fit_material, theta_resolution=4, phi_resolution=4)


sun_position = [8121.97336612, 0., 0.]
sun_world_position = sun_position
sun_position = rotate_y_list([sun_position], Y_ROTATION_ANGLE)[0]
if SCALE:
    sun_position_columns = [[c] for c in sun_position]
    sun_position_clip = bring_into_clip(sun_position_columns, clip_transforms)
    sun_position = [c[0] for c in sun_position_clip]

# Add a sphere for the Sun
sun_radius = CLIP_SIZE * (0.01 if SCALE else 10)
add_sphere(stage, [tuple(sun_position)], [], sun_radius, sun_material, theta_resolution=10, phi_resolution=15)

# Now we need to set up the galaxy image
galaxy_square_edge = 18_500
shift = sun_world_position[0]
shift_fraction = 0.5 * shift / galaxy_square_edge
if TRIM_GALAXY:
    galaxy_image_edge = GALAXY_FRACTION * galaxy_square_edge
else:
    galaxy_image_edge = galaxy_square_edge

galaxy_points = [
    [galaxy_image_edge, 0, galaxy_image_edge],
    [galaxy_image_edge, 0, -galaxy_image_edge],
    [-galaxy_image_edge, 0, -galaxy_image_edge],
    [-galaxy_image_edge, 0, galaxy_image_edge]
]
shift_point = [shift, 0, 0]
if TRIM_GALAXY:
    galaxy_points = [[c + sc for c, sc in zip(p, shift_point)] for p in galaxy_points]

# This is the transformation from world space -> galaxy texture space
# We determined that the galaxy image needs a 90 degree rotation
# and so this affine transformation accounts for that.
# It's easier if we do this before we scale
#
# NB: USD texture coordinates are "t-flipped" relative to glTF
# i.e. if our glTF texture coordinates are (s, t)
# then the corresponding USD coordinates are (s, 1-t)
# (and vice versa, since this operation is an involution)
# (https://openusd.org/release/spec_usdpreviewsurface.html#texture-coordinate-orientation-in-usd)
slope = 0.5 / galaxy_square_edge
intercept = slope * galaxy_square_edge
texcoord = lambda x, z: [(-0.5 / galaxy_square_edge) * z + 0.5, 0.5 - (0.5 / galaxy_square_edge) * x]
galaxy_texcoords = [texcoord(p[0], p[2]) for p in galaxy_points]

galaxy_points = rotate_y_list(galaxy_points, Y_ROTATION_ANGLE)
if SCALE:
    galaxy_point_columns = [[c[i] for c in galaxy_points] for i in range(3)]
    galaxy_points_clip = bring_into_clip(galaxy_point_columns, clip_transforms)
    galaxy_points = [tuple(c[i] for c in galaxy_points_clip) for i in range(len(galaxy_points))]

galaxy_prim_key = f"{default_prim_key}/galaxy"
galaxy_prim = stage.DefinePrim(galaxy_prim_key)
galaxy_mesh_key = f"{galaxy_prim_key}/mesh"
galaxy_mesh = UsdGeom.Mesh.Define(stage, galaxy_mesh_key)
galaxy_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
galaxy_mesh.CreatePointsAttr(galaxy_points)
galaxy_mesh.CreateExtentAttr([(-galaxy_image_edge, 0, -galaxy_image_edge), (galaxy_image_edge, 0, galaxy_image_edge)])
galaxy_mesh.CreateFaceVertexCountsAttr([4])
galaxy_mesh.CreateFaceVertexIndicesAttr([0,1,2,3])

galaxy_bottom_prim_key = f"{default_prim_key}/galaxy_bottom"
galaxy_bottom_prim = stage.DefinePrim(galaxy_bottom_prim_key)
galaxy_bottom_mesh_key = f"{galaxy_bottom_prim_key}/bottom_mesh"
galaxy_bottom_mesh = UsdGeom.Mesh.Define(stage, galaxy_bottom_mesh_key)
galaxy_bottom_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
galaxy_bottom_mesh.CreatePointsAttr(galaxy_points)
galaxy_bottom_mesh.CreateExtentAttr([(-galaxy_image_edge, 0, -galaxy_image_edge), (galaxy_image_edge, 0, galaxy_image_edge)])
galaxy_bottom_mesh.CreateFaceVertexCountsAttr([4])
galaxy_bottom_mesh.CreateFaceVertexIndicesAttr([3,2,1,0])

tex_coords = UsdGeom.PrimvarsAPI(galaxy_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
tex_coords.Set(galaxy_texcoords)

bottom_tex_coords = UsdGeom.PrimvarsAPI(galaxy_bottom_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
bottom_tex_coords.Set(galaxy_texcoords)

galaxy_material_key = f"{galaxy_prim_key}/material"
galaxy_material = UsdShade.Material.Define(stage, galaxy_material_key)
galaxy_pbr_shader = UsdShade.Shader.Define(stage, f"{galaxy_prim_key}/PBRShader")
galaxy_pbr_shader.CreateIdAttr("UsdPreviewSurface")
galaxy_pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
galaxy_pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
galaxy_pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.7)
galaxy_material.CreateSurfaceOutput().ConnectToSource(galaxy_pbr_shader.ConnectableAPI(), "surface")

galaxy_st_reader = UsdShade.Shader.Define(stage, f"{galaxy_material_key}/stReader")
galaxy_st_reader.CreateIdAttr("UsdPrimvarReader_float2")

galaxy_diffuse_texture_sampler = UsdShade.Shader.Define(stage, f"{galaxy_material_key}/diffuseTexture")
galaxy_diffuse_texture_sampler.CreateIdAttr("UsdUVTexture")
galaxy_image_path = "milkywaybar.jpg"
galaxy_diffuse_texture_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(galaxy_image_path)
galaxy_diffuse_texture_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(galaxy_st_reader.ConnectableAPI(), "result")
galaxy_diffuse_texture_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
galaxy_pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(galaxy_diffuse_texture_sampler.ConnectableAPI(), "rgb")

galaxy_st_input = galaxy_material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
galaxy_st_input.Set("st")
galaxy_st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(galaxy_st_input)

galaxy_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
UsdShade.MaterialBindingAPI(galaxy_mesh).Bind(galaxy_material)

galaxy_bottom_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
UsdShade.MaterialBindingAPI(galaxy_bottom_mesh).Bind(galaxy_material)

    
stage.GetRootLayer().Save()

# Create a USDA file as well
path, _ = splitext(output_filepath)
stage.GetRootLayer().Export(f"{path}{extsep}usda")

print(all_triangles)
