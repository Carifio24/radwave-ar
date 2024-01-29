from os.path import extsep, join, splitext

import pandas as pd
import astropy.units as u
from astropy.coordinates import Galactic, SkyCoord


def convert_coordinates(df):
    out = pd.DataFrame()
    out["phase"] = df["phase"]
    out["d"] = df["d"]
    ra_dec = zip(df["ra"], df["dec"])
    coords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(Galactic) for ra, dec in ra_dec]
    ls = []
    bs = []
    for coord in coords:
        ls.append(coord.l.value)
        bs.append(coord.b.value)
    out["l"] = ls 
    out["b"] = bs 

    return out


def convert(filepath):
    df = pd.read_csv(filepath)
    new_df = convert_coordinates(df)
    base, ext = splitext(filepath)
    base = base.replace("_radec", "")
    path = base + ext
    new_df.to_csv(path, index=False)


if __name__ == "__main__":
    for phase in range(271):
        fpath = join("data", f"RW_cluster_oscillation_{phase}_updated_radec.csv")
        convert(fpath)
