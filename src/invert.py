"""Invert P and S arrival times for 3-D velocity: damped least squares over Gaussians, self-calibrated.

Each iteration: rays through the current model give every pick's sensitivity to every Gaussian and to moving
its event. Every event is relocated with its P and S picks. Then, for each phase, whatever an origin time or a
small move of the events could explain is projected out of the data, the damping is chosen on fake earths whose
data are one matrix product away, and one solve updates the model.

    uv run src/invert.py      -> outputs/<name>/
"""
import csv
import json
import time

import numpy as np

import config
from data import Data
from eikonal import linearise
from model import Gaussians, Grid

DAMPINGS = 10. ** np.arange(-6, 1.01, .25)            # relative to the mean eigenvalue of GᵀG
FAKES, ROUNDS = 100, 4                                 # fake earths per round; rounds of noise and damping
STEP_KM = 5.                                           # the furthest an event moves in one iteration


class Separation:
    """Projects each event's picks off [1, dt/dx, dt/dy, dt/dz] at the event: what is left, no origin time and no
    small move of the event can explain (parameter separation). Without `moving`, only the origin times: each
    event's mean is removed. Works on vectors and on rows."""

    def __init__(self, events, moving=None):
        self.groups = [np.flatnonzero(events == e) for e in np.unique(events)]
        self.bases = [np.ones((len(g), 1)) if moving is None else np.c_[np.ones(len(g)), moving[g]] for g in self.groups]
        self.inverses = [np.linalg.pinv(b, rcond=1e-6) for b in self.bases]
        self.absorbed = sum(min(len(g), b.shape[1]) for g, b in zip(self.groups, self.bases))

    def __call__(self, values):
        out = np.array(values, dtype=float)
        for g, b, inverse in zip(self.groups, self.bases, self.inverses):
            out[g] -= b @ (inverse @ out[g])
        return out


def relocate(fits, events, positions, grid):
    """Moves every event to best fit its P and S picks, each phase weighted by its noise: least squares for
    [origin time, dx, dy, dz], at most STEP_KM per call, staying inside the grid. fits: {phase: (picks, residual,
    moving, noise)}. Returns the shifts [events, 3] (km) and the residuals {phase: residual} after them."""
    picks = np.concatenate([f[0] for f in fits.values()])
    r = np.concatenate([f[1] for f in fits.values()])
    moving = np.concatenate([f[2] for f in fits.values()])
    weight = np.concatenate([np.full(len(f[0]), 1 / f[3]) for f in fits.values()])
    owner = np.concatenate([events[f[0]] for f in fits.values()])
    shifts = np.zeros_like(positions)
    for e in np.unique(owner):
        rows = np.flatnonzero(owner == e)
        if len(rows) < 5:                                                # four unknowns need five picks
            continue
        A = np.c_[np.ones(len(rows)), moving[rows]] * weight[rows, None]
        step = np.linalg.lstsq(A, r[rows] * weight[rows], rcond=1e-6)[0][1:]
        shifts[e] = step * min(1., STEP_KM / max(np.linalg.norm(step), 1e-12))
    inside = np.clip(positions + shifts, [a[0] for a in grid.axes], [a[-1] - grid.spacing for a in grid.axes])
    shifts = inside - positions
    after = {phase: f[1] - np.sum(f[2] * shifts[events[f[0]]], 1) for phase, f in fits.items()}
    return shifts, after


def calibrate(G, r, separate, gram, rng):
    """Solve G c ≈ r with the damping that best recovers fake earths. Each fake earth is a few random Gaussians
    as strong as the data suggest, with noise as large as the data show, passed through the same separation;
    both are re-estimated each round. Returns c, its standard deviation, the noise (s) and the damping."""
    n, K = G.shape
    w, U = np.linalg.eigh(G.T @ G)
    w = w.clip(0)
    scale = w.mean()

    def solve(d, dampings):
        return U @ ((U.T @ (G.T @ d))[:, None] / (w[:, None] + np.asarray(dampings)[None] * scale))

    damping, few = 1e-2, max(3, K // 30)
    for _ in range(ROUNDS):
        c = solve(r, [damping])[:, 0]
        dof = float((w / (w + damping * scale)).sum())
        noise = float(np.linalg.norm(r - G @ c)) / max(n - dof - separate.absorbed, 1) ** .5
        signal = max(float(r @ r) - n * noise ** 2, 1e-12)
        probes = np.zeros((K, 20))
        for j in range(20):
            probes[rng.choice(K, few, replace=False), j] = rng.normal(size=few)
        size = (signal / float(np.mean((G @ probes) ** 2) * n)) ** .5
        truth = np.zeros((K, FAKES))
        for j in range(FAKES):
            truth[rng.choice(K, few, replace=False), j] = size * rng.normal(size=few)
        d = G @ truth + noise * separate(rng.normal(size=(n, FAKES)))
        z = U.T @ (G.T @ d)                                                       # [K, fakes]
        e = U @ (z[:, :, None] / (w[:, None, None] + DAMPINGS[None, None] * scale)).reshape(K, -1)
        e = e - np.repeat(truth, len(DAMPINGS), axis=1)
        errors = (e * (gram @ e)).sum(0).reshape(FAKES, len(DAMPINGS)).sum(0)
        damping = float(DAMPINGS[np.argmin(errors)])
    c = solve(r, [damping])[:, 0]
    std = noise * np.sqrt((U ** 2 / (w + damping * scale)).sum(1))
    return c, std, noise, damping


def invert(data, grid, gaussians, start, iterations=2, relocation=True, seed=0, log=print):
    """Velocity models {phase: [nx, ny, nz]} (km/s) from `start`, the Gaussian coefficients with their standard
    deviations, the ray coverage, the final residuals, a report per iteration, and the event positions [E, 3]
    (km; moved only with `relocation`)."""
    rng = np.random.default_rng(seed)
    gram = gaussians.gram()
    phases = [p for p in ('P', 'S') if (data.phase == p).any()]
    speed = {p: start[p].copy() for p in phases}
    total = {p: np.zeros(len(gaussians)) for p in phases}
    positions = data.events.copy()
    noise = {p: None for p in phases}
    std, report = {}, []

    def fit_all():
        """Rays through the current model from the current positions; then, with relocation, every event moved."""
        fits, rays = {}, {}
        for phase in phases:
            picks, predicted, moving, G, coverage = linearise(speed[phase], grid, gaussians, data, phase, positions)
            r = Separation(data.event[picks])(data.time[picks] - predicted)
            fits[phase] = (picks, r, moving, noise[phase] or float(np.sqrt(np.mean(r ** 2))))
            rays[phase] = (G, coverage)
        if not relocation:
            return fits, rays, np.zeros_like(positions)
        shifts, after = relocate(fits, data.event, positions, grid)
        for phase in phases:
            picks, _, moving, sigma = fits[phase]
            fits[phase] = (picks, after[phase], moving, sigma)
        return fits, rays, shifts

    for iteration in range(1, iterations + 1):
        t = time.time()
        fits, rays, shifts = fit_all()
        positions += shifts
        log(f'iteration {iteration}: events moved {np.sqrt(np.mean(np.sum(shifts ** 2, 1))):.2f} km (rms), '
            f'{time.time() - t:.0f} s')
        for phase in phases:
            t = time.time()
            picks, r, moving, _ = fits[phase]
            G = rays[phase][0]
            separate = Separation(data.event[picks], moving if relocation else None)
            c, std[phase], noise[phase], damping = calibrate(separate(G), separate(r), separate, gram, rng)
            speed[phase] = np.maximum(speed[phase] + gaussians.render(c), .1)
            total[phase] += c
            report.append(dict(iteration=iteration, phase=phase, picks=len(picks), residual_s=float(np.sqrt(np.mean(r ** 2))),
                               noise_s=noise[phase], damping=damping, seconds=round(time.time() - t, 1)))
            log(f'iteration {iteration} {phase}: {len(picks)} picks, residual {report[-1]["residual_s"]:.3f} s, '
                f'noise {noise[phase]:.3f} s, damping {damping:.0e}, {report[-1]["seconds"]} s')
    fits, rays, shifts = fit_all()                                     # the final fit, after a last relocation
    positions += shifts
    coverage = {p: rays[p][1] for p in phases}
    residual = {p: (fits[p][0], Separation(data.event[fits[p][0]])(fits[p][1])) for p in phases}
    for phase in phases:
        log(f'final {phase}: residual {np.sqrt(np.mean(residual[phase][1] ** 2)):.3f} s')
    return speed, total, std, coverage, residual, report, positions


def save(run, data, grid, gaussians, start, speed, total, std, coverage, residual, report, positions, config_text):
    """outputs/<name>/: model.npz, gaussians.csv, events.csv, residuals.csv, summary.json and the config used."""
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.toml').write_text(config_text)
    blank = np.full(grid.shape, np.nan)
    np.savez(run / 'model.npz', x_km=grid.axes[0], y_km=grid.axes[1], z_km=grid.axes[2],
             origin=[data.latitude, data.longitude],
             **{f'v{p.lower()}': speed.get(p, blank) for p in 'PS'}, **{f'v{p.lower()}_start': start[p] for p in 'PS'},
             **{f'coverage_{p.lower()}_km': coverage.get(p, blank) for p in 'PS'})
    latitude, longitude = data.geographic(gaussians.centres[:, 0], gaussians.centres[:, 1])
    with open(run / 'gaussians.csv', 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['phase', 'latitude', 'longitude', 'depth_km', 'width_km', 'change_km_s', 'std_km_s'])
        for phase in total:
            for k in np.argsort(-np.abs(total[phase]) / std[phase]):
                out.writerow([phase, f'{latitude[k]:.4f}', f'{longitude[k]:.4f}', f'{gaussians.centres[k, 2]:.1f}',
                              f'{gaussians.widths[k]:g}', f'{total[phase][k]:.4f}', f'{std[phase][k]:.4f}'])
    latitude, longitude = data.geographic(positions[:, 0], positions[:, 1])
    moved = np.linalg.norm(positions - data.events, axis=1)
    with open(run / 'events.csv', 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['event', 'latitude', 'longitude', 'depth_km', 'moved_km'])
        for e, name in enumerate(data.event_names):
            out.writerow([name, f'{latitude[e]:.5f}', f'{longitude[e]:.5f}', f'{positions[e, 2]:.2f}', f'{moved[e]:.2f}'])
    with open(run / 'residuals.csv', 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['event', 'station', 'phase', 'residual_s'])
        for phase, (picks, r) in residual.items():
            for i, value in zip(picks, r):
                out.writerow([data.event_names[data.event[i]], data.station_names[data.station[i]], phase, f'{value:.4f}'])
    final = {p: float(np.sqrt(np.mean(r ** 2))) for p, (_, r) in residual.items()}
    (run / 'summary.json').write_text(json.dumps(dict(gaussians=len(gaussians), grid=list(grid.shape),
                                                      iterations=report, final_residual_s=final), indent=1))


if __name__ == '__main__':
    c = config.load()
    data = Data(c['data']['folder'])
    grid = Grid(data, **c['grid'])
    gaussians = Gaussians(grid, c['invert']['widths'])
    start = {p: grid.layered(c['start']['depth'], c['start'][f'v{p.lower()}']) for p in 'PS'}
    print(f'{len(data.stations)} stations, {len(data.events)} events, {len(data.time)} picks; '
          f'grid {grid.shape} at {grid.spacing:g} km; {len(gaussians)} Gaussians')
    results = invert(data, grid, gaussians, start, c['invert']['iterations'], c['invert']['relocate'], c['invert']['seed'])
    save(c['run'], data, grid, gaussians, start, *results, c['path'].read_text())
    print(f'-> {c["run"]}/')
