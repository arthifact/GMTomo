"""A synthetic subduction zone to try GMTomo on: data/example/{stations, events, picks}.csv and truth.npz.

A slab 15 km thick dips at 30° and is 6% fast; a sphere of 15 km radius in the crust is 8% slow in P and 10%
in S. 36 stations, 400 earthquakes (three quarters in the slab), P picks with 0.05 s of noise and S picks
on 70% of records with 0.1 s. The times are computed on a 0.75 km grid, finer than the inversion's. The
catalogue locations are off by 2 km horizontally and 4 km in depth (standard deviations), as a routine
catalogue located in a 1-D model might be; truth.npz keeps the true ones.

    uv run src/example.py
"""
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter

from data import EARTH_KM
from eikonal import field
from model import Grid

DEPTH, VP, VS = [0, 20, 35, 60], [5.8, 6.5, 8.0, 8.2], [3.35, 3.75, 4.6, 4.7]   # the start in config.toml
DIP, TRENCH, THICKNESS = math.radians(30), -50., 15.                          # the slab top meets the surface at x = -50 km
BODY, RADIUS = np.array([15., 10., 12.]), 15.


def truth(grid):
    """True vp, vs [nx, ny, nz] (km/s) on a grid."""
    x, y, z = np.meshgrid(*grid.axes, indexing='ij')
    below = z * math.cos(DIP) - (x - TRENCH) * math.sin(DIP)                  # depth below the slab top
    slab = ((below > 0) & (below < THICKNESS)).astype(float)
    body = ((x - BODY[0]) ** 2 + (y - BODY[1]) ** 2 + (z - BODY[2]) ** 2 < RADIUS ** 2).astype(float)
    slab, body = (gaussian_filter(m, 2 / grid.spacing) for m in (slab, body))
    return grid.layered(DEPTH, VP) * (1 + .06 * slab - .08 * body), grid.layered(DEPTH, VS) * (1 + .06 * slab - .1 * body)


def make(folder='data/example', stations=36, events=400, spacing=.75, seed=0):
    rng = np.random.default_rng(seed)
    side = round(stations ** .5)
    xy = np.stack(np.meshgrid(*[np.linspace(-45, 45, side)] * 2), -1).reshape(-1, 2) + rng.uniform(-5, 5, (side ** 2, 2))
    xy -= xy.mean(0)                                                            # centred, as data.py centres them
    sites = np.c_[xy, -rng.uniform(0, 1, len(xy))]                              # 0-1000 m above sea level
    slab = []
    while len(slab) < 3 * events // 4:
        along, inside, y = rng.uniform(5, 120), rng.uniform(1, 10), rng.uniform(-45, 45)
        p = [TRENCH + along * math.cos(DIP) - inside * math.sin(DIP), y, along * math.sin(DIP) + inside * math.cos(DIP)]
        if 5 < p[2] < 58 and abs(p[0]) < 50:
            slab.append(p)
    crust = np.c_[rng.uniform(-45, 45, (events - len(slab), 2)), rng.uniform(2, 20, events - len(slab))]
    quakes = np.vstack((slab, crust))[rng.permutation(events)]
    grid = Grid(SimpleNamespace(stations=sites, events=quakes), spacing, depth=60, margin=20)
    vp, vs = truth(grid)
    origin = rng.uniform(0, 86400, events)                                     # unknown to the inversion
    picks = []
    for s, site in enumerate(sites):
        for phase, speed, noise, share in (('P', vp, .05, 1.), ('S', vs, .1, .7)):
            times = grid.sample(field(speed, grid, site), quakes)
            for e in np.flatnonzero(rng.random(events) < share):
                picks.append((f'EQ{e:04d}', f'ST{s:02d}', phase, f'{origin[e] + times[e] + rng.normal(0, noise):.3f}'))
        grid.corrections.clear()
    catalogue = quakes + rng.normal(0, [2, 2, 4], quakes.shape)
    catalogue[:, 2] = np.clip(catalogue[:, 2], .5, 58)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    degrees = lambda km: math.degrees(km / EARTH_KM)                            # the origin is at 0°N 0°E
    write(folder / 'stations.csv', ['station', 'latitude', 'longitude', 'elevation_m'],
          [(f'ST{s:02d}', f'{degrees(p[1]):.5f}', f'{degrees(p[0]):.5f}', f'{-1000 * p[2]:.0f}') for s, p in enumerate(sites)])
    write(folder / 'events.csv', ['event', 'latitude', 'longitude', 'depth_km'],
          [(f'EQ{e:04d}', f'{degrees(p[1]):.5f}', f'{degrees(p[0]):.5f}', f'{p[2]:.2f}') for e, p in enumerate(catalogue)])
    write(folder / 'picks.csv', ['event', 'station', 'phase', 'time'], picks)
    np.savez(folder / 'truth.npz', x_km=grid.axes[0], y_km=grid.axes[1], z_km=grid.axes[2], vp=vp, vs=vs, events_km=quakes)
    return folder


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(header)
        out.writerows(rows)


if __name__ == '__main__':
    print(f'-> {make()}/')
