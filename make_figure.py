from os.path import join
import pandas as pd
import pyvista as pv
from gltflib import GLTF, Accessor, AccessorType, Asset, BufferView, Primitive, \
    ComponentType, GLTFModel, Node, Scene, Attributes, Mesh, Buffer
import struct

def cluster_filepath(phase):
    return join("data", f"RW_cluster_oscillation_{phase}_updated.csv")

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)

N_PHASES = 270
N_POINTS = initial_df.shape[0]

points = [tuple(initial_df[c][i] for c in ["x", "y", "z"]) for i in range(N_POINTS)]
points_bytearray = bytearray()
for point in points:
    for coord in point:
        points_bytearray.extend(struct.pack('f', coord))

prev_xyz = [initial_df["x"], initial_df["y"], initial_df["z"]]
forward_differences = { pt: [] for pt in range(N_POINTS) }
for phase in range(N_PHASES):
    df = pd.read_csv(cluster_filepath(phase))
    xyz = [df["x"], df["y"], df["z"]]
    diffs = [c - pc for c, pc in zip(xyz, prev_xyz)]
    for pt in range(N_POINTS):
        forward_differences[pt].append(tuple(x[pt] for x in diffs))
    prev_xyz = xyz

timestamps = [0.1 * i for i in range(N_PHASES)]

gltf = GLTFModel(asset=Asset(version='2.0'),
                 scenes=[Scene(nodes=list(range(N_POINTS)))],
                 nodes=[Node(mesh=0) for _ in range(N_POINTS)],
                 meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0))])],
                 buffers=[Buffer
        )
