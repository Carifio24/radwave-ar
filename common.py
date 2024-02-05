from os.path import join
import pandas as pd

__all__ = [
    "N_PHASES", "cluster_filepath", "scale",
    "get_scaled_positions_and_translations"
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


# Note that these need to be overall translations (i.e. x(t) - x(0))
# NOT per-timestep translations (e.g. x(t) - x(t-dt))
def get_scaled_positions_and_translations():
    translations = { pt: [] for pt in range(N_PHASES) }
    initial_df = pd.read_csv(cluster_filepath(0))
    initial_xyz = [initial_df["x"], initial_df["y"], initial_df["z"]]
    cmin = min([min(c) for c in initial_xyz])
    cmax = max([max(c) for c in initial_xyz])
    dfs = []
    for phase in range(1, N_PHASES+1):
        df = pd.read_csv(cluster_filepath(phase))
        dfs.append(df)
        xyz = [df["x"], df["y"], df["z"]]
        for coord in xyz:
            cmin = min(cmin, min(coord))
            cmax = max(cmax, max(coord))

    initial_xyz = [scale(c, cmin, cmax) for i, c in enumerate(initial_xyz)]
    for df in dfs:
        xyz = [df["x"], df["y"], df["z"]]
        xyz = [scale(c, cmin, cmax) for i, c in enumerate(xyz)]
        diffs = [c - pc for c, pc in zip(xyz, initial_xyz)]
        for pt in range(df.shape[0]):
            translations[pt].append(tuple(x[pt] for x in diffs))
    
    positions = [tuple(c[i] for c in initial_xyz) for i in range(N_POINTS)]
    return positions, translations
