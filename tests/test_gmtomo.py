"""Checks of the travel times, the sensitivities, the separation of origin times and locations, the
calibration, relocation, the CSV reading and a whole run on a small synthetic subduction zone.

    uv run tests/test_gmtomo.py
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Data  # noqa: E402
from eikonal import field, linearise, march  # noqa: E402
import example  # noqa: E402
from invert import Separation, calibrate, invert, relocate  # noqa: E402
from model import Gaussians, Grid  # noqa: E402

RNG = np.random.default_rng(0)
NETWORK = SimpleNamespace(stations=np.c_[RNG.uniform(-40, 40, (6, 2)), np.zeros(6)],
                          events=np.c_[RNG.uniform(-40, 40, (25, 2)), RNG.uniform(5, 35, 25)])


def gradient_model(grid, v0=5., g=.03):
    """v = v0 + g z, whose exact times are known: cosh(g t) = 1 + g² |r - s|² / (2 v(s) v(r))."""
    return v0 + g * grid.layered([grid.axes[2][0], grid.axes[2][-1]], grid.axes[2][[0, -1]])


class TravelTimeTests(unittest.TestCase):
    def test_times_match_the_exact_answer(self):
        grid = Grid(NETWORK, spacing=1., depth=40., margin=10.)
        v0, g, source = 5., .03, np.array([.3, .7, 0.])
        times = field(gradient_model(grid, v0, g), grid, source)
        far = NETWORK.events
        exact = np.arccosh(1 + g ** 2 * np.sum((far - source) ** 2, 1) / (2 * v0 * (v0 + g * far[:, 2]))) / g
        error = np.abs(grid.sample(times, far) - exact)
        self.assertLess(error.max(), .03)
        self.assertLess(error.mean(), .01)

    def test_the_correction_divides_the_error_by_three(self):
        grid = Grid(NETWORK, spacing=2., depth=40., margin=10.)
        v, source = np.full(grid.shape, 6.), np.array([.3, .7, 0.])
        plain, r = march(v, grid, source)
        corrected = field(v, grid, source)
        far = r > 10
        self.assertLess(np.abs(corrected - r / 6)[far].mean(), np.abs(plain - r / 6)[far].mean() / 3)

    def test_sensitivity_matches_finite_differences(self):
        grid = Grid(NETWORK, spacing=2., depth=40., margin=10.)
        data = SimpleNamespace(**vars(NETWORK), event=np.repeat(np.arange(25), 6), station=np.tile(np.arange(6), 25),
                               phase=np.full(150, 'P'))
        gaussians, v = Gaussians(grid, [10., 20.]), gradient_model(grid)
        c = np.zeros(len(gaussians))
        c[RNG.choice(len(gaussians), 5, replace=False)] = RNG.normal(0, .02, 5)
        _, before, _, G, _ = linearise(v, grid, gaussians, data, 'P')
        _, after, _, _, _ = linearise(v + gaussians.render(c), grid, gaussians, data, 'P')
        change = after - before
        self.assertLess(np.linalg.norm(change - G @ c) / np.linalg.norm(change), .05)


class InversionTests(unittest.TestCase):
    def test_origin_times_and_small_moves_do_not_matter(self):
        events, moving = np.repeat(np.arange(10), 8), RNG.normal(size=(80, 3))
        times, shifts, origins = RNG.normal(size=80), RNG.normal(size=(10, 3)), 100 * RNG.normal(size=10)
        separate = Separation(events, moving)
        moved = times + origins[events] + np.sum(moving * shifts[events], 1)
        np.testing.assert_allclose(separate(times), separate(moved), atol=1e-9)
        np.testing.assert_allclose(Separation(events)(times), Separation(events)(times + origins[events]), atol=1e-9)

    def test_calibration_finds_the_noise(self):
        events = np.repeat(np.arange(200), 10)
        separate = Separation(events)
        G = separate(RNG.normal(size=(2000, 60)))
        truth = np.zeros(60)
        truth[:5] = 1
        r = G @ truth + .3 * separate(RNG.normal(size=2000))
        c, std, noise, damping = calibrate(G, r, separate, np.eye(60), np.random.default_rng(1))
        self.assertAlmostEqual(noise, .3, delta=.03)
        self.assertLess(np.linalg.norm(c - truth), np.linalg.norm(truth) / 2)
        self.assertTrue((std > 0).all())


class RelocationTests(unittest.TestCase):
    def test_relocation_moves_events_towards_the_truth(self):
        grid = Grid(NETWORK, spacing=2., depth=50., margin=10.)
        v = gradient_model(grid)
        stations = np.c_[RNG.uniform(-40, 40, (12, 2)), np.zeros(12)]
        data = SimpleNamespace(stations=stations, events=NETWORK.events, event=np.repeat(np.arange(25), 12),
                               station=np.tile(np.arange(12), 25), phase=np.full(300, 'P'))
        data.time = np.concatenate([grid.sample(field(v, grid, stations[s]), NETWORK.events[[e]])
                                    for e, s in zip(data.event, data.station)]) + 50
        wrong = NETWORK.events + RNG.normal(0, [2, 2, 4], NETWORK.events.shape)
        positions = wrong.copy()
        for _ in range(3):
            picks, predicted, moving, _, _ = linearise(v, grid, Gaussians(grid, [20.]), data, 'P', positions)
            r = Separation(data.event[picks])(data.time[picks] - predicted)
            shifts, _ = relocate({'P': (picks, r, moving, .05)}, data.event, positions, grid)
            positions = positions + shifts
        before = np.linalg.norm(wrong - NETWORK.events, axis=1).mean()
        self.assertLess(np.linalg.norm(positions - NETWORK.events, axis=1).mean(), before / 3)


class RunTests(unittest.TestCase):
    def test_reading_iso_times_and_unknown_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'stations.csv').write_text('station,latitude,longitude,elevation_m\nA,0,0,0\nB,0.1,0.1,500\n')
            (tmp / 'events.csv').write_text('event,latitude,longitude,depth_km\nE1,0.05,0.05,10\n')
            (tmp / 'picks.csv').write_text('event,station,phase,time\nE1,A,Pg,2026-01-01T00:00:01.5Z\n'
                                           'E1,B,Sn,2026-01-01T00:00:03Z\nE1,A,Lg,0\n')
            data = Data(tmp)
            self.assertEqual(list(data.phase), ['P', 'S'])
            self.assertAlmostEqual(data.time[1] - data.time[0], 1.5)
            self.assertAlmostEqual(data.stations[1, 2], -.5)
            (tmp / 'picks.csv').write_text('event,station,phase,time\nE1,C,P,0\n')
            with self.assertRaises(ValueError):
                Data(tmp)

    def test_a_small_subduction_zone_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = example.make(Path(tmp) / 'example', stations=16, events=150, spacing=1.5)
            data = Data(folder)
            grid = Grid(data, spacing=2., depth=60., margin=20.)
            start = {p: grid.layered(example.DEPTH, v) for p, v in (('P', example.VP), ('S', example.VS))}
            speed, *_ = invert(data, grid, Gaussians(grid, [10., 20., 40.]), start, iterations=2, log=lambda *_: None)
            true, _ = example.truth(grid)
            under = (np.abs(grid.axes[0])[:, None, None] < 40) & (np.abs(grid.axes[1])[None, :, None] < 40) \
                & (grid.axes[2][None, None] < 50)
            before = np.sqrt(np.mean((start['P'] - true)[under] ** 2))
            after = np.sqrt(np.mean((speed['P'] - true)[under] ** 2))
            self.assertLess(after, .8 * before)


if __name__ == '__main__':
    unittest.main()
