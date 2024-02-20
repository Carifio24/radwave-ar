from itertools import product
import math
from os.path import join
import pandas as pd
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource
from gltflib import Accessor, AccessorType, Asset, BufferTarget, BufferView, Image, PBRMetallicRoughness, Primitive, \
    ComponentType, GLTFModel, Node, Sampler, Scene, Attributes, Mesh, Buffer, \
    Animation, AnimationSampler, Channel, Target, Material, Texture, TextureInfo, interpolation 
from numpy import inf
import operator
import struct

from common import N_VISIBLE_PHASES, N_PHASES, N_POINTS, BEST_FIT_FILEPATH, BEST_FIT_DOWNSAMPLED_FILEPATH, bring_into_clip, CLUSTER_FILEPATH, clip_linear_transformations 

BEST_FIT_PATH = BEST_FIT_FILEPATH
SCALE = False 


# Note that there are occasionally some funky coordinate things throughout
# glTF is a right-handed +y coordinate system
# which should be equivalent to (x, -z, y)


def sphere_mesh_index(row, column, theta_resolution, phi_resolution):
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        return phi_resolution * (row - 1) + column + 1


def get_bounds():
    mins = [inf, inf, inf]
    maxes = [-inf, -inf, -inf]
    cluster_df = pd.read_csv(CLUSTER_FILEPATH)
    best_fit_filepath = BEST_FIT_PATH
    best_fit_df = pd.read_csv(best_fit_filepath)
    for phase in range(N_VISIBLE_PHASES + 1):
        slice = cluster_df[cluster_df["phase"] == phase]
        xyz = [slice[c] for c in ["xc", "zc", "yc"]]
        xyz[0] = -xyz[0]
        for index, coord in enumerate(xyz):
            mins[index] = min(mins[index], min(coord))
            maxes[index] = max(maxes[index], max(coord))
    
        best_fit_phase = best_fit_df[best_fit_df["phase"] == phase]
        best_fit_xyz = [best_fit_phase[c] for c in ["xc", "zc", "yc"]]
        best_fit_xyz[0] = -best_fit_xyz[0]
        for index, coord in enumerate(best_fit_xyz):
            mins[index] = min(mins[index], min(coord))
            maxes[index] = max(maxes[index], max(coord))

    return mins, maxes


# Note that these need to be overall translations (i.e. x(t) - x(0))
# NOT per-timestep translations (e.g. x(t) - x(t-dt))
def get_positions_and_translations(scale=True, clip_transforms=None):
    df = pd.read_csv(CLUSTER_FILEPATH)
    initial_phase = df[df["phase"] == 0]
    initial_xyz = [-initial_phase["xc"], initial_phase["zc"], initial_phase["yc"]]
    translations = { pt: [] for pt in range(N_POINTS) }

    if scale:
        initial_xyz = bring_into_clip(initial_xyz, clip_transforms)
    for phase in range(1, N_PHASES + 1):
        slice = df[df["phase"] == phase % 360]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] = -xyz[0]
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        diffs = [c - pc for c, pc in zip(xyz, initial_xyz)]
        for pt in range(initial_phase.shape[0]):
            translations[pt].append(tuple(x[pt] for x in diffs))
    
    positions = [tuple(c[i] for c in initial_xyz) for i in range(N_POINTS)]
    return positions, translations


# Note that these need to be overall translations (i.e. x(t) - x(0))
# NOT per-timestep translations (e.g. x(t) - x(t-dt))
def get_best_fit_positions_and_translations(scale=True, clip_transforms=None):
    df = pd.read_csv(BEST_FIT_PATH)
    initial_phase = df[df["phase"] == 0]
    initial_xyz = [-initial_phase["xc"], initial_phase["zc"], initial_phase["yc"]]
    translations = { pt: [] for pt in range(initial_phase.shape[0]) }

    if scale:
        initial_xyz = bring_into_clip(initial_xyz, clip_transforms)
    for phase in range(1, N_PHASES + 1):
        bf_phase = phase % 360
        slice = df[df["phase"] == bf_phase]
        xyz = [slice[c].to_numpy() for c in ["xc", "zc", "yc"]]
        xyz[0] = -xyz[0]
        if scale:
            xyz = bring_into_clip(xyz, clip_transforms)
        diffs = [c - pc for c, pc in zip(xyz, initial_xyz)]
        for pt in range(slice.shape[0]):
            translations[pt].append(tuple(x[pt] for x in diffs))
    
    positions = [tuple(c[i] for c in initial_xyz) for i in range(initial_phase.shape[0])]
    return positions, translations


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


output_directory = "out"


# Let's set up our arrays and any constant values
radius = 5 * (0.005 if SCALE else 5)
theta_resolution = 10
phi_resolution = 15
POINTS_PER_SPHERE = phi_resolution * (theta_resolution - 2) + 2
buffers = []
buffer_views = []
accessors = []
nodes = []
meshes = []
file_resources = []
animation_samplers = []
invisible_animation_samplers = []
channels = []
invisible_channels = []
animations = []
materials = [
    # Cluster spheres
    Material( pbrMetallicRoughness=PBRMetallicRoughness(baseColorFactor=[31 / 255, 94 / 255, 241 / 255, 1])),
    # Best-fit spheres
    Material(pbrMetallicRoughness=PBRMetallicRoughness(baseColorFactor=[132 / 255, 215 / 255, 245 / 255, 1], metallicFactor=0, roughnessFactor=0)),
]

# Set up some stuff that we'll need for the animations
# In particular, set up out timestamp buffer/view/accessor
time_delta = 0.01
timestamps = [time_delta * i for i in range(1, N_PHASES)]
min_time = min(timestamps)
max_time = max(timestamps)
mins, maxes = get_bounds()
clip_transforms = clip_linear_transformations(list(zip(mins, maxes)))
positions, translations = get_positions_and_translations(scale=SCALE, clip_transforms=clip_transforms)

time_barr = bytearray()
for time in timestamps:
    time_barr.extend(struct.pack('f', time))
time_bin = "time.bin"
file_resources.append(FileResource(time_bin, data=time_barr))
time_buffer = Buffer(byteLength=len(time_barr), uri=time_bin)
buffers.append(time_buffer)
time_buffer_view = BufferView(buffer=len(buffers)-1, byteLength=len(time_barr))
buffer_views.append(time_buffer_view)
time_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.FLOAT.value, count=len(timestamps),
                         type=AccessorType.SCALAR.value, min=[min_time], max=[max_time])
accessors.append(time_accessor)
time_accessor_index = len(accessors) - 1

# "Animation" times and scales for the non-visible timestamps
zero_scale = [0.0, 0.0, 0.0]
one_scale = [1.0, 1.0, 1.0]
invisible_scales = [one_scale for _ in range(1, N_VISIBLE_PHASES + 1)] + [zero_scale for _ in range(N_PHASES-N_VISIBLE_PHASES-1)]
invisible_barr = bytearray()
for scale in invisible_scales:
    for coord in scale:
        invisible_barr.extend(struct.pack('f', coord))
invisible_bin = "invisible.bin"
invisible_buffer = Buffer(byteLength=len(invisible_barr), uri=invisible_bin)
buffers.append(invisible_buffer)
invisible_view = BufferView(buffer=len(buffers)-1, byteLength=len(invisible_barr))
buffer_views.append(invisible_view)
invisible_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.FLOAT.value, count=len(invisible_scales),
                              type=AccessorType.VEC3.value, min=zero_scale, max=one_scale)
accessors.append(invisible_accessor)
invisible_scale_accessor_index = len(accessors) - 1
file_resources.append(FileResource(invisible_bin, data=invisible_barr))


# Create a sphere for each point at phase=0
for index, point in enumerate(positions):

    points, triangles = sphere_mesh(point, radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    point_mins = [min([operator.itemgetter(i)(pt) for pt in points]) for i in range(3)]
    point_maxes = [max([operator.itemgetter(i)(pt) for pt in points]) for i in range(3)]

    arr = bytearray()
    for point in points:
        for coord in point:
            arr.extend(struct.pack('f', coord))
    triangles_offset = len(arr)
    for triangle in triangles:
        for idx in triangle:
            arr.extend(struct.pack('I', idx))

    # Set up the position and indices (triangulation) for this cluster
    # The main thing here is to make sure that our indices to buffers/views/accessors are correct
    bin = f"buf_{index}.bin"
    buffer = Buffer(byteLength=len(arr), uri=bin)
    buffers.append(buffer)
    positions_view = BufferView(buffer=len(buffers)-1, byteLength=triangles_offset, target=BufferTarget.ARRAY_BUFFER.value)
    indices_view = BufferView(buffer=len(buffers)-1, byteLength=len(arr)-triangles_offset, byteOffset=triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
    buffer_views.append(positions_view)
    buffer_views.append(indices_view)
    positions_accessor = Accessor(bufferView=len(buffer_views)-2, componentType=ComponentType.FLOAT.value, count=POINTS_PER_SPHERE, type=AccessorType.VEC3.value, min=point_mins, max=point_maxes)
    indices_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.UNSIGNED_INT.value, count=len(triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[POINTS_PER_SPHERE-1])
    accessors.append(positions_accessor)
    accessors.append(indices_accessor)
    file_resources.append(FileResource(bin, data=arr))
    meshes.append(Mesh(primitives=[Primitive(attributes=Attributes(POSITION=len(buffer_views)-2), indices=len(buffer_views)-1, material=0)]))
    nodes.append(Node(mesh=len(meshes)-1))

    # Set up the buffer/view/accessor for the animation data for this point
    # TODO: We definitely want separate BufferViews and Accessors for each point,
    # but maybe all of the animation data could live in one buffer? And just use the correct offset.
    # Would this even be any better?
    diffs = translations[index]
    diff_bin = f"diff_{index}.bin"
    diff_barr = bytearray()
    for diff in diffs:
        for value in diff:
            diff_barr.extend(struct.pack('f', value))
    file_resources.append(FileResource(diff_bin, data=diff_barr))
    diff_buffer = Buffer(byteLength=len(diff_barr), uri=diff_bin)
    buffers.append(diff_buffer)
    diff_buffer_view = BufferView(buffer=len(buffers)-1, byteLength=len(diff_barr))
    buffer_views.append(diff_buffer_view)
    diff_mins = [min([operator.itemgetter(i)(diff) for diff in diffs]) for i in range(3)]
    diff_maxes = [max([operator.itemgetter(i)(diff) for diff in diffs]) for i in range(3)]
    diff_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.FLOAT.value, count=N_PHASES-1,
                             type=AccessorType.VEC3.value, min=diff_mins, max=diff_maxes)
    accessors.append(diff_accessor)
    target = Target(node=len(nodes)-1, path="translation")
    sampler = AnimationSampler(input=time_accessor_index, interpolation="LINEAR", output=len(accessors)-1)
    animation_samplers.append(sampler)
    channel = Channel(target=target, sampler=len(animation_samplers)-1)
    channels.append(channel)

    # Make the "invisible" animation for each point
    invisible_target = Target(node=len(nodes)-1, path="scale")
    invisible_sampler = AnimationSampler(input=time_accessor_index, interpolation="LINEAR", output=invisible_scale_accessor_index)
    invisible_animation_samplers.append(invisible_sampler)
    channel = Channel(target=invisible_target, sampler=len(invisible_animation_samplers)-1)
    invisible_channels.append(channel)
    
# Now we're going to do the same for the best-fit
# except with larger spheres
best_fit_radius = 0.5 * (0.001 if SCALE else 10)
bf_positions, bf_translations = get_best_fit_positions_and_translations(scale=SCALE, clip_transforms=clip_transforms)

for index, point in enumerate(bf_positions):
    points, triangles = sphere_mesh(point, best_fit_radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution)
    point_mins = [min([operator.itemgetter(i)(pt) for pt in points]) for i in range(3)]
    point_maxes = [max([operator.itemgetter(i)(pt) for pt in points]) for i in range(3)]
        
    arr = bytearray()
    for point in points:
        for coord in point:
            arr.extend(struct.pack('f', coord))
    triangles_offset = len(arr)
    for triangle in triangles:
        for idx in triangle:
            arr.extend(struct.pack('I', idx))

    # Set up the position and indices (triangulation) for this cluster
    # The main thing here is to make sure that our indices to buffers/views/accessors are correct
    bin = f"bf_buf_{index}.bin"
    buffer = Buffer(byteLength=len(arr), uri=bin)
    buffers.append(buffer)
    positions_view = BufferView(buffer=len(buffers)-1, byteLength=triangles_offset, target=BufferTarget.ARRAY_BUFFER.value)
    indices_view = BufferView(buffer=len(buffers)-1, byteLength=len(arr)-triangles_offset, byteOffset=triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
    buffer_views.append(positions_view)
    buffer_views.append(indices_view)
    positions_accessor = Accessor(bufferView=len(buffer_views)-2, componentType=ComponentType.FLOAT.value, count=POINTS_PER_SPHERE, type=AccessorType.VEC3.value, min=point_mins, max=point_maxes)
    indices_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.UNSIGNED_INT.value, count=len(triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[POINTS_PER_SPHERE-1])
    accessors.append(positions_accessor)
    accessors.append(indices_accessor)
    file_resources.append(FileResource(bin, data=arr))
    meshes.append(Mesh(primitives=[Primitive(attributes=Attributes(POSITION=len(buffer_views)-2), indices=len(buffer_views)-1, material=1)]))
    nodes.append(Node(mesh=len(meshes)-1))

    # Set up the buffer/view/accessor for the animation data for this point
    # TODO: We definitely want separate BufferViews and Accessors for each point,
    # but maybe all of the animation data could live in one buffer? And just use the correct offset.
    # Would this even be any better?
    diffs = bf_translations[index]
    diff_bin = f"bf_diff_{index}.bin"
    diff_barr = bytearray()
    for diff in diffs:
        for value in diff:
            diff_barr.extend(struct.pack('f', value))
    file_resources.append(FileResource(diff_bin, data=diff_barr))
    diff_buffer = Buffer(byteLength=len(diff_barr), uri=diff_bin)
    buffers.append(diff_buffer)
    diff_buffer_view = BufferView(buffer=len(buffers)-1, byteLength=len(diff_barr))
    buffer_views.append(diff_buffer_view)
    diff_mins = [min([operator.itemgetter(i)(diff) for diff in diffs]) for i in range(3)]
    diff_maxes = [max([operator.itemgetter(i)(diff) for diff in diffs]) for i in range(3)]
    diff_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.FLOAT.value, count=N_PHASES-1, type=AccessorType.VEC3.value, min=diff_mins, max=diff_maxes)
    accessors.append(diff_accessor)
    target = Target(node=len(nodes)-1, path="translation")
    sampler = AnimationSampler(input=time_accessor_index, interpolation="LINEAR", output=len(accessors)-1)
    animation_samplers.append(sampler)
    channel = Channel(target=target, sampler=len(animation_samplers)-1)
    channels.append(channel)

animation = Animation(name="Oscillating", channels=channels, samplers=animation_samplers)
animations.append(animation)
# animationinvisible_animation = Animation(name="Hiding", channels=invisible_channels, samplers=invisible_animation_samplers)
# animations.append(invisible_animation)

# Finally, let's create the galaxy texture

# The "radius" of the galaxy image in parsecs
# We had to figure this out by trial-and-error
galaxy_square_side = 18_500
galaxy_points = [
    [galaxy_square_side, 0, galaxy_square_side],
    [galaxy_square_side, 0, -galaxy_square_side],
    [-galaxy_square_side, 0, -galaxy_square_side],
    [-galaxy_square_side, 0, galaxy_square_side]
]
galaxy_points = [[p[0], p[1], p[2]] for p in galaxy_points]
galaxy_triangles= [[0, 1, 2], [2, 3, 0], [0, 2, 1], [2, 0, 3]]
galaxy_texcoords = [
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0]
]
galaxy_image_path = join("images", "milkywaybar.jpg")
galaxy_image = Image(uri=galaxy_image_path)
file_resources.append(FileResource(galaxy_image_path))
galaxy_sampler = Sampler()
samplers = [galaxy_sampler]
galaxy_texture = Texture(source=0, sampler=len(samplers)-1)
galaxy_texture_info = TextureInfo(index=0)
materials.append(Material(alphaMode="BLEND", pbrMetallicRoughness=PBRMetallicRoughness(baseColorFactor=[1, 1, 1, 0.75], baseColorTexture=galaxy_texture_info, metallicFactor=0, roughnessFactor=1)))

galaxy_barr = bytearray()
for point in galaxy_points:
    for coord in point:
        galaxy_barr.extend(struct.pack('f', coord))
galaxy_triangles_offset = len(galaxy_barr)
for triangle in galaxy_triangles:
    for idx in triangle:
        galaxy_barr.extend(struct.pack('I', idx))
galaxy_texcoords_offset = len(galaxy_barr)
for vertex in galaxy_texcoords:
    for coord in vertex:
        galaxy_barr.extend(struct.pack('f', coord))

galaxy_bin = "galaxy.bin"
galaxy_buffer = Buffer(byteLength=len(galaxy_barr), uri=galaxy_bin)
buffers.append(galaxy_buffer)
galaxy_positions_view = BufferView(buffer=len(buffers)-1, byteLength=galaxy_triangles_offset, target=BufferTarget.ARRAY_BUFFER.value)
galaxy_indices_view = BufferView(buffer=len(buffers)-1, byteLength=galaxy_texcoords_offset-galaxy_triangles_offset, byteOffset=galaxy_triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
galaxy_texcoords_view = BufferView(buffer=len(buffers)-1, byteLength=len(galaxy_barr)-galaxy_texcoords_offset, byteOffset=galaxy_texcoords_offset, target=BufferTarget.ARRAY_BUFFER.value)
buffer_views.append(galaxy_positions_view)
buffer_views.append(galaxy_indices_view)
buffer_views.append(galaxy_texcoords_view)
galaxy_positions_accessor = Accessor(bufferView=len(buffer_views)-3, componentType=ComponentType.FLOAT.value, count=len(galaxy_points), type=AccessorType.VEC3.value, min=galaxy_points[2], max=galaxy_points[0])
galaxy_indices_accessor = Accessor(bufferView=len(buffer_views)-2, componentType=ComponentType.UNSIGNED_INT.value, count=len(galaxy_triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[3])
galaxy_texcoords_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.FLOAT.value, count=len(galaxy_texcoords), type=AccessorType.VEC2.value, min=[0,0], max=[1,1])
accessors.append(galaxy_positions_accessor)
accessors.append(galaxy_indices_accessor)
accessors.append(galaxy_texcoords_accessor)
file_resources.append(FileResource(galaxy_bin, data=galaxy_barr))
meshes.append(Mesh(primitives=[Primitive(attributes=Attributes(POSITION=len(accessors)-3, TEXCOORD_0=len(accessors)-1), indices=len(accessors)-2, material=len(materials)-1)]))
nodes.append(Node(mesh=len(meshes)-1))

# Finally, set up our model and export
node_indices = [_ for _ in range(len(nodes))]
model = GLTFModel(
    asset=Asset(version='2.0'),
    scenes=[Scene(nodes=node_indices)],
    nodes=nodes,
    meshes=meshes,
    buffers=buffers,
    bufferViews=buffer_views,
    accessors=accessors,
    materials=materials,
    animations=animations,
    samplers=samplers,
    images=[galaxy_image],
    textures=[galaxy_texture]
)
gltf = GLTF(model=model, resources=file_resources)
gltf.export(join(output_directory, "radwave.gltf"))
gltf.export(join(output_directory, "radwave.glb"))
