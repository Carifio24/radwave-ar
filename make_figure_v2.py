from itertools import product
import math
from os.path import join
import pandas as pd
import pyvista as pv
from gltflib.gltf import GLTF
from gltflib.gltf_resource import FileResource
from gltflib import Accessor, AccessorType, Asset, BufferTarget, BufferView, Primitive, \
    ComponentType, GLTFModel, Node, Scene, Attributes, Mesh, Buffer, \
    Animation, AnimationSampler, Channel, Target
import operator
import struct


def cluster_filepath(phase):
    return join("data", f"RW_cluster_oscillation_{phase}_updated.csv")


def sphere_mesh_index(row, column, theta_resolution, phi_resolution):
    if row == 0:
        return 0
    elif row == theta_resolution - 1:
        return (theta_resolution - 2) * phi_resolution + 1
    else:
        return phi_resolution * (row - 1) + column + 1


# theta is the azimuthal angle here. Sorry math folks
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

    bin = f"buf_{index}.bin"
    buffer = Buffer(byteLength=len(arr), uri=bin)
    buffers.append(buffer)
    positions_view = BufferView(buffer=index, byteLength=triangles_offset, target=BufferTarget.ARRAY_BUFFER.value)
    indices_view = BufferView(buffer=index, byteLength=len(arr)-triangles_offset, byteOffset=triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
    buffer_views.append(positions_view)
    buffer_views.append(indices_view)
    positions_accessor = Accessor(bufferView=len(buffer_views)-2, componentType=ComponentType.FLOAT.value, count=POINTS_PER_SPHERE, type=AccessorType.VEC3.value, min=point_mins, max=point_maxes)
    indices_accessor = Accessor(bufferView=len(buffer_views)-1, componentType=ComponentType.UNSIGNED_INT.value, count=len(triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[POINTS_PER_SPHERE-1])
    accessors.append(positions_accessor)
    accessors.append(indices_accessor)
    file_resources.append(FileResource(bin, data=arr))
    meshes.append(Mesh(primitives=[Primitive(attributes=Attributes(POSITION=2*index), indices=2*index+1)]))
    nodes.append(Node(mesh=index))

node_indices = [_ for _ in range(len(nodes))]
model = GLTFModel(
    asset=Asset(version='2.0'),
    scenes=[Scene(nodes=node_indices)],
    nodes=nodes,
    meshes=meshes,
    buffers=buffers,
    bufferViews=buffer_views,
    accessors=accessors
)
gltf = GLTF(model=model, resources=file_resources)
gltf.export(join(output_directory, "test.gltf"))
gltf.export(join(output_directory, "test.glb"))
