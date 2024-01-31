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
    triangles = [(int(0), i, i + 1) for i in range(1, phi_resolution)]
    tr, pr = theta_resolution, phi_resolution
    triangles.append((0, theta_resolution, 1))
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


center = (0, 0, 0)
radius = 1
theta_resolution = 5
phi_resolution = 5
points, triangles = sphere_mesh(center, radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution)

for t in triangles:
    print(t)

point_mins = [min([operator.itemgetter(i)(point) for point in points]) for i in range(3)]
point_maxes = [max([operator.itemgetter(i)(point) for point in points]) for i in range(3)]

output_directory = "out"

N_POINTS = phi_resolution * (theta_resolution - 2) + 2
arr = bytearray()
for point in points:
    for coord in point:
        arr.extend(struct.pack('f', coord))

triangles_offset = len(arr)
for triangle in triangles:
    for index in triangle:
        arr.extend(struct.pack('I', index))

buffer = Buffer(byteLength=len(arr), uri="buf.bin")
buffer_views = [
    BufferView(buffer=0, byteLength=triangles_offset, target=BufferTarget.ARRAY_BUFFER.value),
    BufferView(buffer=0, byteOffset=triangles_offset, byteLength=len(arr)-triangles_offset, target=BufferTarget.ELEMENT_ARRAY_BUFFER.value)
]
accessors = [
    Accessor(bufferView=0, componentType=ComponentType.FLOAT.value, count=N_POINTS, type=AccessorType.VEC3.value, min=point_mins, max=point_maxes),
    Accessor(bufferView=1, componentType=ComponentType.UNSIGNED_INT.value, count=len(triangles) * 3, type=AccessorType.SCALAR.value, min=[0], max=[N_POINTS-1])
]
file_resources = [FileResource("buf.bin", data=arr)]

model = GLTFModel(
    asset=Asset(version='2.0'),
    scenes=[Scene(nodes=[0])],
    nodes=[Node(mesh=0)],
    meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])],
    buffers=[buffer],
    bufferViews=buffer_views,
    accessors=accessors
)
gltf = GLTF(model=model, resources=file_resources)
gltf.export(join(output_directory, "test.gltf"))
gltf.export(join(output_directory, "test.glb"))
