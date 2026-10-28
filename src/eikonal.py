"""Travel times and rays through a 3-D model, from the eikonal equation solved at each station.

By reciprocity, one time field from a station gives its arrival time from every event, and the ray from an
event back to the station is the steepest descent through that field. Along a ray dt/dv = -∫ dl / v², so a
Gaussian's effect on an arrival is that integral weighted by the Gaussian.
"""
import numpy as np
import skfmm
from scipy import sparse

CHUNK = 20000                                          # ray samples per sensitivity product


def march(speed, grid, source):
    """Fast marching from `source` [3] (km) through speed [nx, ny, nz] (km/s); the front starts as an exact
    sphere of 3 nodes. Returns the times (s) and the distance of every node from the source."""
    x, y, z = (a.reshape(shape) for a, shape in zip(grid.axes, ((-1, 1, 1), (1, -1, 1), (1, 1, -1))))
    r = np.sqrt((x - source[0]) ** 2 + (y - source[1]) ** 2 + (z - source[2]) ** 2)
    radius = 3 * grid.spacing
    v0 = float(grid.sample(speed, source[None])[0])
    outside = np.abs(np.asarray(skfmm.travel_time(r - radius, speed, dx=grid.spacing, order=2)))
    return np.where(r <= radius, r / v0, outside + radius / v0), r


def field(speed, grid, source):
    """Arrival times (s) at every node from `source` [3] (km), through speed [nx, ny, nz] (km/s).

    Fast marching errs by up to 1% along directions the grid favours. In a uniform model the times are
    exactly distance / speed, so that error is measured once per source and divided out: three times
    smaller on average (tests/test_gmtomo.py)."""
    key = tuple(np.round(source, 6))
    if key not in grid.corrections:
        times, r = march(np.ones(grid.shape), grid, source)
        grid.corrections[key] = np.where(r > 0, r / np.maximum(times, 1e-12), 1.).astype(np.float32)
    return march(speed, grid, source)[0] * grid.corrections[key]


def rays(times, grid, sources, end, step=1.):
    """Steepest-descent paths through `times` from each source [n, 3] to `end` [3], in steps of `step` km.
    Returns sample points [m, 3], the path length each stands for [m] (km) and its source [m]."""
    gradient = np.gradient(times, grid.spacing)
    h = min(step, grid.spacing)
    position, alive = np.array(sources, dtype=float), np.ones(len(sources), dtype=bool)
    points, lengths, owners = [], [], []
    for _ in range(int(4 * sum(a[-1] - a[0] for a in grid.axes) / h)):
        which = np.flatnonzero(alive)
        if not len(which):
            break
        p = position[which]
        gap = np.linalg.norm(p - end, axis=1)
        close = gap <= 2 * grid.spacing                                 # the last stretch is straight
        points.append((p[close] + end) / 2)
        lengths.append(gap[close])
        owners.append(which[close])
        alive[which[close]] = False
        which, p = which[~close], p[~close]
        g = np.stack([grid.sample(component, p) for component in gradient], 1)
        g /= np.linalg.norm(g, axis=1, keepdims=True).clip(1e-12)
        points.append(p - h / 2 * g)
        lengths.append(np.full(len(p), h))
        owners.append(which)
        position[which] = p - h * g
    which = np.flatnonzero(alive)                                       # never arrived: straight to the end
    points.append((position[which] + end) / 2)
    lengths.append(np.linalg.norm(position[which] - end, axis=1))
    owners.append(which)
    return np.concatenate(points), np.concatenate(lengths), np.concatenate(owners)


def linearise(speed, grid, gaussians, data, phase):
    """For every pick of `phase`: its index, predicted time (s, from origin time 0), and sensitivity to every
    Gaussian [picks, K] (s per km/s); plus the ray length through every node (km)."""
    picks = np.flatnonzero(data.phase == phase)
    predicted, G = np.zeros(len(picks)), np.zeros((len(picks), len(gaussians)))
    coverage = np.zeros(grid.shape)
    for station in np.unique(data.station[picks]):
        rows = np.flatnonzero(data.station[picks] == station)
        times = field(speed, grid, data.stations[station])
        sources = data.events[data.event[picks[rows]]]
        predicted[rows] = grid.sample(times, sources)
        points, lengths, owners = rays(times, grid, sources, data.stations[station])
        weight = -lengths / grid.sample(speed, points) ** 2
        for i in range(0, len(points), CHUNK):
            part = slice(i, i + CHUNK)
            n = len(weight[part])
            spread = sparse.csr_matrix((weight[part], (owners[part], np.arange(n))), shape=(len(rows), n))
            G[rows] += spread @ gaussians.at(points[part])
        node = np.clip(np.round(grid.index(points)).astype(int).T, 0, np.array(grid.shape) - 1)
        np.add.at(coverage, tuple(node.T), lengths)
    return picks, predicted, G, coverage
