from os.path import join
import pandas as pd

__all__ = [
    "N_PHASES", "cluster_filepath", "scale"
]

N_PHASES = 270
output_directory = "out"


def cluster_filepath(phase):
    return join("data", f"RW_cluster_oscillation_{phase}_updated.csv")


initial_filepath = cluster_filepath(0)
initial_df = pd.read_csv(initial_filepath)
N_POINTS = initial_df.shape[0]


def scale(value, lower, upper):
    return (value - lower) / (upper - lower)

