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

output_directory = "out"

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)

N_PHASES = 270
N_POINTS = initial_df.shape[0]

points = [tuple(initial_df[c][i] for c in ["x", "y", "z"]) for i in range(N_POINTS)]
points_bytearray = bytearray()
for point in points:
    for coord in point:
        points_bytearray.extend(struct.pack('f', coord))

timestamps = [0.1 * i for i in range(N_PHASES)]
time_bytearray = bytearray()
for time in timestamps:
    time_bytearray.extend(struct.pack('f', time))

prev_xyz = [initial_df["x"], initial_df["y"], initial_df["z"]]
forward_differences = { pt: [] for pt in range(N_POINTS) }
dfs = {}
for phase in range(N_PHASES):
    df = pd.read_csv(cluster_filepath(phase))
    dfs[phase] = df
    xyz = [df["x"], df["y"], df["z"]]
    diffs = [c - pc for c, pc in zip(xyz, prev_xyz)]
    for pt in range(N_POINTS):
        forward_differences[pt].append(tuple(x[pt] for x in diffs))
    prev_xyz = xyz

forward_diff_bytearrays = []
for point in range(N_POINTS):
    bytearr = bytearray()
    for diff in forward_differences[point]:
        for value in diff:
            bytearr.extend(struct.pack('f', value))
    forward_diff_bytearrays.append(bytearr)

time_bin = "time.bin"
time_buffer = Buffer(byteLength=len(time_bytearray), uri=time_bin)
points_bin = "points.bin"
points_buffer = Buffer(byteLength=len(points_bytearray), uri=points_bin)

time_min = min(timestamps)
time_max = max(timestamps)
pt_mins = [min([point[i] for point in points]) for i in range(3)]
pt_maxs = [max([point[i] for point in points]) for i in range(3)]
buffers = [time_buffer, points_buffer]
buffer_views = [
    BufferView(buffer=0, byteLength=len(time_bytearray)),
    BufferView(buffer=1, byteLength=len(points_bytearray))
]
accessors = [
    Accessor(bufferView=0, componentType=ComponentType.FLOAT.value, count=N_PHASES, type=AccessorType.SCALAR.value, min=[time_min], max=[time_max]),
    Accessor(bufferView=1, componentType=ComponentType.FLOAT.value, count=N_POINTS, type=AccessorType.VEC3.value, min=pt_mins, max=pt_maxs)
]
file_resources = [
    FileResource(time_bin, data=time_bytearray),
    FileResource(points_bin, data=points_bytearray)
]

meshes = [Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0))])]
nodes = [Node(mesh=0) for _ in range(N_POINTS)]
targets = [Target(node=idx, path="translation") for idx in range(len(nodes))]
samplers = [AnimationSampler(input=0, interpolation="LINEAR", output=point+2) for point in range(N_POINTS)]
channels = [Channel(target=t, sampler=0) for i, t in enumerate(targets)]
animations = [Animation(channels=[c], samplers=[s]) for c, s in zip(channels, samplers)]

samplers = []
channels = []
for point, bytearr in enumerate(forward_diff_bytearrays):
    uri = f"diffs_{point}.bin"
    buff = Buffer(byteLength=len(bytearr), uri=uri)
    buffers.append(buff)
    mins = []
    maxs = []
    for i in range(3):
        vals = [diff[i] for diff in forward_differences[point]]
        mins.append(min(vals))
        maxs.append(max(vals))
    if point == 0:
        print(mins)
        print(maxs)
        print("====")
        for diff in forward_differences[point]:
            print(diff)
    # The extra 2 accounts for the time and point buffers added above
    buffer_views.append(BufferView(buffer=point+2, byteLength=len(bytearr)))
    accessors.append(Accessor(bufferView=point+2, componentType=ComponentType.FLOAT.value, count=N_POINTS, type=AccessorType.VEC3.value, min=mins, max=maxs))
    file_resources.append(FileResource(uri, data=bytearr))

model = GLTFModel(asset=Asset(version='2.0'),
                  scenes=[Scene(nodes=list(range(N_POINTS)))],
                  meshes=meshes,
                  nodes=nodes,
                  buffers=buffers,
                  bufferViews=buffer_views,
                  accessors=accessors,
                  # animations=animations
        )

gltf = GLTF(model=model, resources=file_resources)
gltf.export(join(output_directory, "test.gltf"))
gltf.export(join(output_directory, "test.glb"))
