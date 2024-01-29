from os.path import join
import pandas as pd
import pyvista as pv
from gltflib.gltf import GLTF

def cluster_filepath(phase):
    return join("data", f"RW_cluster_oscillation_{phase}_updated_radec.csv")

initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(cluster_filepath(initial_filepath))


