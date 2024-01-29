from math import cos, sin
from os.path import join, splitext

import pandas as pd
import astropy.units as u
from astropy.coordinates import Galactic, SkyCoord


def lbd_to_xyz(l, b, d):
    x = d * sin(l) * cos(b)
    y = d * sin(l) * sin(b)
    z = d * cos(l)
    return x, y, z


def convert_coordinates(df):
    out = pd.DataFrame()
    out["phase"] = df["phase"]
    out["d"] = df["d"]
    ra_dec = zip(df["ra"], df["dec"])
    coords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(Galactic) for ra, dec in ra_dec]
    ls = []
    bs = []
    xs = []
    ys = []
    zs = []
    for coord, d in zip(coords, df["d"]):
        l = coord.l.to(u.rad).value
        b = coord.b.to(u.rad).value
        x, y, z = lbd_to_xyz(l, b, d)
        ls.append(l)
        bs.append(b)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    out["l"] = ls 
    out["b"] = bs 
    out["x"] = xs
    out["y"] = ys
    out["z"] = zs

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
