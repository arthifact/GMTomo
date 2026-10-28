"""Figures of a run: the velocity change from the start, in map view at three depths and in a section
along x through the network's centre. Grey: no ray passes within 2 km.

    uv run src/plot.py      -> outputs/<name>/figures/
"""
import numpy as np
import matplotlib
from scipy.ndimage import binary_dilation

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from data import Data  # noqa: E402

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def change(model, phase):
    """Percent change from the start, blank where no ray passes."""
    p = phase.lower()
    out = 100 * (model[f'v{p}'] / model[f'v{p}_start'] - 1)
    return np.where(binary_dilation(model[f'coverage_{p}_km'] > 0, iterations=2), out, np.nan)


def figures(run, data, depths=(10, 25, 40)):
    model = np.load(run / 'model.npz')
    x, y, z = model['x_km'], model['y_km'], model['z_km']
    phases = [p for p in 'PS' if np.isfinite(model[f'v{p.lower()}']).all()]
    (run / 'figures').mkdir(exist_ok=True)
    grey = plt.get_cmap('RdBu').copy()
    grey.set_bad('#e6e6e6')
    limit = {p: max(1., float(np.nanpercentile(np.abs(change(model, p)), 99))) for p in phases}

    fig, axes = plt.subplots(len(phases), len(depths), figsize=(2.6 * len(depths), 2.5 * len(phases)), squeeze=False,
                             layout='constrained')
    for row, phase in zip(axes, phases):
        values = change(model, phase)
        for ax, depth in zip(row, depths):
            k = int(np.argmin(np.abs(z - depth)))
            image = ax.imshow(values[:, :, k].T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), cmap=grey,
                              vmin=-limit[phase], vmax=limit[phase])
            near = np.abs(data.events[:, 2] - depth) < 5
            ax.plot(*data.events[near, :2].T, '.', ms=1.5, color='k', alpha=.5)
            ax.plot(*data.stations[:, :2].T, 'v', ms=3, color='k')
            ax.set_title(f'V{phase.lower()} at {z[k]:.0f} km')
            ax.set_xlabel('x east (km)')
        row[0].set_ylabel('y north (km)')
        fig.colorbar(image, ax=row.tolist(), shrink=.8, label='change (%)')
    fig.savefig(run / 'figures' / 'depth_slices.png', dpi=160, bbox_inches='tight')

    fig, axes = plt.subplots(len(phases), 1, figsize=(6, 2.2 * len(phases)), squeeze=False, layout='constrained')
    j = int(np.argmin(np.abs(y)))
    for ax, phase in zip(axes[:, 0], phases):
        image = ax.imshow(change(model, phase)[:, j, :].T, origin='upper', extent=(x[0], x[-1], z[-1], z[0]), cmap=grey,
                          vmin=-limit[phase], vmax=limit[phase], aspect='auto')
        near = np.abs(data.events[:, 1] - y[j]) < 5
        ax.plot(data.events[near, 0], data.events[near, 2], '.', ms=1.5, color='k', alpha=.5)
        ax.set_title(f'V{phase.lower()}, section at y = {y[j]:.0f} km')
        ax.set_ylabel('depth (km)')
        fig.colorbar(image, ax=ax, label='change (%)')
    axes[-1, 0].set_xlabel('x east (km)')
    fig.savefig(run / 'figures' / 'section.png', dpi=160, bbox_inches='tight')
    return run / 'figures'


if __name__ == '__main__':
    c = config.load()
    print(f'-> {figures(c["run"], Data(c["data"]["folder"]))}/')
