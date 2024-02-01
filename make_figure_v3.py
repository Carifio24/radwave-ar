from itertools import product
import math
from os.path import join
import pandas as pd
import pyvista as pv
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource
from gltflib import Accessor, AccessorType, Asset, BufferTarget, BufferView, PBRMetallicRoughness, Primitive, \
    ComponentType, GLTFModel, Node, Scene, Attributes, Mesh, Buffer, \
    Animation, AnimationSampler, Channel, Target, Material
import operator
import struct

N_PHASES = 270


def cluster_filepath(phase):
    return join("data", f"RW_cluster_oscillation_{phase}_updated.csv")


def sphere_mesh_index(row, column, theta_resolution, phi_resolution):
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        return phi_resolution * (row - 1) + column + 1

# Note that these need to be overall translations (i.e. x(t) - x(0))
# NOT per-timestep translations (e.g. x(t) - x(t-dt))
def get_translations():
    translations = { pt: [] for pt in range(N_PHASES) }
    initial_df = pd.read_csv(cluster_filepath(0))
    initial_xyz = [initial_df["x"], initial_df["y"], initial_df["z"]]
    for phase in range(1, N_PHASES+1):
        df = pd.read_csv(cluster_filepath(phase))
        xyz = [df["x"], df["y"], df["z"]]
        diffs = [c - pc for c, pc in zip(xyz, initial_xyz)]
        for pt in range(df.shape[0]):
            translations[pt].append(tuple(x[pt] for x in diffs))

    return translations


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

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)

# Let's set up our arrays and any constant values
radius = 5
theta_resolution = 10
phi_resolution = 15
POINTS_PER_SPHERE = phi_resolution * (theta_resolution - 2) + 2
buffers = []
buffer_views = []
accessors = []
nodes = []
meshes = []
file_resources = []
materials = [Material(pbrMetallicRoughness=PBRMetallicRoughness(baseColorFactor=[31 / 255, 60 / 255, 241 / 255, 1]))]
samplers = []
channels = []

# Set up some stuff that we'll need for the animations
# In particular, set up out timestamp buffer/view/accessor
time_delta = 0.01
timestamps = [time_delta * i for i in range(1, N_PHASES)]
min_time = min(timestamps)
max_time = max(timestamps)
translations = get_translations()
time_barr = bytearray()
for time in timestamps:
    time_barr.extend(struct.pack('f', time))
time_bin = "time.bin"
file_resources.append(FileResource(time_bin, data=time_barr))
time_buffer = Buffer(byteLength=len(time_barr), uri=time_bin)
time_buffer_view = BufferView(buffer=0, byteLength=len(time_barr))
time_accessor = Accessor(bufferView=0, componentType=ComponentType.FLOAT.value, count=N_PHASES-1,
                         type=AccessorType.SCALAR.value, min=[min_time], max=[max_time])
buffers.append(time_buffer)
buffer_views.append(time_buffer_view)
accessors.append(time_accessor)

# Create a sphere for each point at phase=0
N_POINTS = initial_df.shape[0]
positions = [tuple(initial_df[c][i] for c in ["x", "y", "z"]) for i in range(N_POINTS)]
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
    nodes.append(Node(mesh=index))

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
    target = Target(node=index, path="translation")
    sampler = AnimationSampler(input=0, interpolation="LINEAR", output=len(accessors)-1)
    samplers.append(sampler)
    channel = Channel(target=target, sampler=index)
    channels.append(channel)
    
animation = Animation(channels=channels, samplers=samplers)

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
    animations=[animation]
)
gltf = GLTF(model=model, resources=file_resources)
gltf.export(join(output_directory, "v3.gltf"))
gltf.export(join(output_directory, "v3.glb"))
