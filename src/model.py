"""The grid, the 1-D start and the Gaussians.

A velocity model is the start plus a sum of Gaussians:

    v(r) = start(z) + Σ_k c_k exp(-|r - μ_k|² / 2σ_k²)

The Gaussians sit on a lattice for each width, spaced 1.5 widths apart; the inversion finds every c_k.
"""
import itertools

import numpy as np
from scipy.ndimage import map_coordinates


class Grid:
    """Nodes every `spacing` km over the stations and events plus `margin`, from the highest station down
    to `depth` km (or below the deepest event). Arrays are indexed [x, y, z]."""

    def __init__(self, data, spacing=2., depth=60., margin=20.):
        points = np.vstack((data.stations, data.events))
        low = np.floor((points.min(0) - [margin, margin, 0]) / spacing) * spacing
        high = np.ceil((points.max(0) + [margin, margin, 0]) / spacing) * spacing
        high[2] = max(depth, high[2] + 2 * spacing)
        self.spacing = spacing
        self.axes = [np.arange(lo, hi + spacing / 2, spacing) for lo, hi in zip(low, high)]
        self.shape = tuple(len(a) for a in self.axes)
        self.corrections = {}                          # per source: the fast-marching error to divide out

    def index(self, points):
        """Fractional node indices [3, n] of points [n, 3] (km)."""
        return ((np.asarray(points) - [a[0] for a in self.axes]) / self.spacing).T

    def sample(self, values, points):
        """Values [nx, ny, nz] at points [n, 3], linearly interpolated."""
        return map_coordinates(values, self.index(points), order=1, mode='nearest')

    def layered(self, depth_km, values):
        """A 1-D model on the grid: `values` at `depth_km`, linear in between, constant beyond."""
        return np.broadcast_to(np.interp(self.axes[2], depth_km, values), self.shape).copy()


class Gaussians:
    """Unit Gaussians of each width (km) on a lattice spaced 1.5 widths apart over the grid. They are
    separable, exp(-|r - μ|²/2σ²) = gx(x) gy(y) gz(z), so three tables [nodes, K] hold all of them."""

    def __init__(self, grid, widths):
        centres, sigma = [], []
        for width in widths:
            lattice = [np.linspace(a[0], a[-1], max(2, round((a[-1] - a[0]) / (1.5 * width)) + 1)) for a in grid.axes]
            for centre in itertools.product(*lattice):
                centres.append(centre)
                sigma.append(width)
        self.grid, self.centres, self.widths = grid, np.array(centres), np.array(sigma, dtype=float)
        self.tables = [np.exp(-.5 * ((a[:, None] - self.centres[None, :, i]) / self.widths) ** 2)
                       for i, a in enumerate(grid.axes)]

    def __len__(self):
        return len(self.widths)

    def render(self, c):
        """Σ c_k g_k on the grid [nx, ny, nz]."""
        return np.einsum('ik,jk,lk,k->ijl', *self.tables, c, optimize=True)

    def at(self, points):
        """Every Gaussian at points [n, 3]: [n, K], float32, linear between nodes."""
        out = None
        for table, f in zip(self.tables, self.grid.index(points)):
            i = np.clip(np.floor(f).astype(int), 0, len(table) - 2)
            t = np.clip(f - i, 0, 1)[:, None].astype(np.float32)
            value = table[i].astype(np.float32) * (1 - t) + table[i + 1].astype(np.float32) * t
            out = value if out is None else out * value
        return out

    def gram(self):
        """[K, K] overlaps Σ_nodes g_k g_l: the squared size of a coefficient change on the grid."""
        return np.prod([t.T @ t for t in self.tables], axis=0)
