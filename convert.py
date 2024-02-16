from math import cos, sin
from os.path import join, splitext

import pandas as pd
import astropy.units as u
from astropy.coordinates import Galactic, Galactocentric, SkyCoord


def lbd_to_xyz(l, b, d):
    x = d * sin(l) * cos(b)
    y = d * sin(l) * sin(b)
    z = d * cos(l)
    return x, y, z


def convert_coordinates_galactocentric(df):
    out = pd.DataFrame()
    out["phase"] = df["phase"]
    ra_dec_d = zip(df["ra"], df["dec"], df["d"])
    print("About to convert")
    coords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=d * u.pc).transform_to(Galactocentric) for ra, dec, d in ra_dec_d]
    xs = []
    ys = []
    zs = []
    for coord in coords:
        xs.append(coord.x.value)
        ys.append(coord.y.value)
        zs.append(coord.z.value)

    out["x"] = xs
    out["y"] = ys
    out["z"] = zs

    return out 

def convert_coordinates_galactic(df):
    out = pd.DataFrame()
    out["phase"] = df["phase"]
    out["d"] = df["d"]
    ra_dec = zip(df["ra"], df["dec"])
    print("About to convert")
    coords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(Galactic) for ra, dec in ra_dec]
    ls = []
    bs = []
    xs = []
    ys = []
    zs = []
    i = 0
    print("Converting")
    for coord, d in zip(coords, df["d"]):
        if i % 1000 == 0:
            print(i)
        i += 1
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
    print("Read CSV")
    new_df = convert_coordinates_galactic(df)
    base, ext = splitext(filepath)
    base = base.replace("_radec", "")
    path = base + ext
    new_df.to_csv(path, index=False)


if __name__ == "__main__":
    # for phase in range(271):
    #     print(f"Phase: {phase}")
    #     fpath = join("data", f"RW_cluster_oscillation_{phase}_updated_radec.csv")
    #     convert(fpath)

    print("Downsampled")
    convert(join("data", "RW_best_fit_oscillation_phase_radec_downsampled.csv"))
    print("Regular")
    convert(join("data", "RW_best_fit_oscillation_phase_radec.csv"))
