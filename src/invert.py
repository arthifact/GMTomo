"""Invert P and S arrival times for 3-D velocity: damped least squares over Gaussians, self-calibrated.

Each iteration, for each phase: rays through the current model give every pick's sensitivity to every
Gaussian; origin times are removed by subtracting each event's mean; the damping is chosen on fake earths
whose data are one matrix product away; one solve updates the model.

    uv run src/invert.py      -> outputs/<name>/
"""
import csv
import json
import shutil
import time

import numpy as np

import config
from data import Data
from eikonal import linearise
from model import Gaussians, Grid

DAMPINGS = 10. ** np.arange(-6, 1.01, .25)            # relative to the mean eigenvalue of GᵀG
FAKES, ROUNDS = 100, 4                                 # fake earths per round; rounds of noise and damping


def centre(values, events):
    """Values minus their event's mean: the part no origin time can explain. Works on vectors and on rows."""
    counts = np.bincount(events)
    sums = np.zeros((len(counts),) + values.shape[1:])
    np.add.at(sums, events, values)
    return values - (sums / np.maximum(counts, 1).reshape((-1,) + (1,) * (values.ndim - 1)))[events]


def calibrate(G, r, events, gram, rng):
    """Solve G c ≈ r with the damping that best recovers fake earths. Each fake earth is a few random Gaussians
    as strong as the data suggest, with noise as large as the data show; both are re-estimated each round.
    Returns c, its standard deviation, the noise (s) and the damping."""
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
        noise = float(np.linalg.norm(r - G @ c)) / max(n - dof - len(np.unique(events)), 1) ** .5
        signal = max(float(r @ r) - n * noise ** 2, 1e-12)
        probes = np.zeros((K, 20))
        for j in range(20):
            probes[rng.choice(K, few, replace=False), j] = rng.normal(size=few)
        size = (signal / float(np.mean((G @ probes) ** 2) * n)) ** .5
        errors = np.zeros(len(DAMPINGS))
        for _ in range(FAKES):
            truth = np.zeros(K)
            truth[rng.choice(K, few, replace=False)] = size * rng.normal(size=few)
            d = G @ truth + noise * centre(rng.normal(size=n), events)
            e = solve(d, DAMPINGS) - truth[:, None]
            errors += (e * (gram @ e)).sum(0)
        damping = float(DAMPINGS[np.argmin(errors)])
    c = solve(r, [damping])[:, 0]
    std = noise * np.sqrt((U ** 2 / (w + damping * scale)).sum(1))
    return c, std, noise, damping


def invert(data, grid, gaussians, start, iterations=2, seed=0, log=print):
    """Velocity models {phase: [nx, ny, nz]} (km/s) from `start`, the Gaussian coefficients with their
    standard deviations, the ray coverage, and a report per iteration."""
    rng = np.random.default_rng(seed)
    gram = gaussians.gram()
    phases = [p for p in ('P', 'S') if (data.phase == p).any()]
    speed = {p: start[p].copy() for p in phases}
    total = {p: np.zeros(len(gaussians)) for p in phases}
    std, report = {}, []
    for iteration in range(1, iterations + 1):
        for phase in phases:
            t = time.time()
            picks, predicted, G, _ = linearise(speed[phase], grid, gaussians, data, phase)
            events = data.event[picks]
            r = centre(data.time[picks] - predicted, events)
            c, std[phase], noise, damping = calibrate(centre(G, events), r, events, gram, rng)
            speed[phase] = np.maximum(speed[phase] + gaussians.render(c), .1)
            total[phase] += c
            report.append(dict(iteration=iteration, phase=phase, picks=len(picks), residual_s=float(np.sqrt(np.mean(r ** 2))),
                               noise_s=noise, damping=damping, seconds=round(time.time() - t, 1)))
            log(f'iteration {iteration} {phase}: {len(picks)} picks, residual {report[-1]["residual_s"]:.3f} s, '
                f'noise {noise:.3f} s, damping {damping:.0e}, {report[-1]["seconds"]} s')
    coverage, residual = {}, {}
    for phase in phases:                                                 # the final fit
        picks, predicted, _, coverage[phase] = linearise(speed[phase], grid, gaussians, data, phase)
        residual[phase] = (picks, centre(data.time[picks] - predicted, data.event[picks]))
        log(f'final {phase}: residual {np.sqrt(np.mean(residual[phase][1] ** 2)):.3f} s')
    return speed, total, std, coverage, residual, report


def save(run, data, grid, gaussians, start, speed, total, std, coverage, residual, report, config_text):
    """outputs/<name>/: model.npz, gaussians.csv, residuals.csv, summary.json and the config used."""
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
    results = invert(data, grid, gaussians, start, c['invert']['iterations'], c['invert']['seed'])
    save(c['run'], data, grid, gaussians, start, *results, c['path'].read_text())
    print(f'-> {c["run"]}/')
