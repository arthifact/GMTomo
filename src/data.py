"""The three CSV files, in local kilometres: x east, y north, z down from sea level.

    stations.csv   station, latitude, longitude, elevation_m
    events.csv     event, latitude, longitude, depth_km
    picks.csv      event, station, phase, time

A phase starting with P or S (P, Pg, Pn, S, Sg, ...) is a P or S pick; others are skipped. Times are
seconds or ISO 8601, on any clock: origin times are unknown, so only the differences between the picks
of one event are used.
"""
import csv
import datetime
import math
from pathlib import Path

import numpy as np

EARTH_KM = 6371.


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def seconds(text):
    try:
        return float(text)
    except ValueError:
        return datetime.datetime.fromisoformat(text.strip().replace('Z', '+00:00')).timestamp()


class Data:
    """Stations [S, 3] and events [E, 3] in km, and picks: event and station indices, phase, time (s)."""

    def __init__(self, folder):
        folder = Path(folder)
        stations, events, picks = (rows(folder / f'{name}.csv') for name in ('stations', 'events', 'picks'))
        self.latitude = float(np.mean([float(s['latitude']) for s in stations]))     # the local origin
        self.longitude = float(np.mean([float(s['longitude']) for s in stations]))
        self.station_names = [s['station'] for s in stations]
        self.event_names = [e['event'] for e in events]
        self.stations = np.array([[*self.local(float(s['latitude']), float(s['longitude'])),
                                   -float(s['elevation_m']) / 1000] for s in stations])
        self.events = np.array([[*self.local(float(e['latitude']), float(e['longitude'])),
                                 float(e['depth_km'])] for e in events])
        station_index = {name: i for i, name in enumerate(self.station_names)}
        event_index = {name: i for i, name in enumerate(self.event_names)}
        kept = []
        for line, p in enumerate(picks, start=2):
            phase = p['phase'].strip().upper()[:1]
            if phase not in ('P', 'S'):
                continue
            for name, index, kind in ((p['station'], station_index, 'station'), (p['event'], event_index, 'event')):
                if name not in index:
                    raise ValueError(f'picks.csv line {line}: unknown {kind} {name!r}')
            kept.append((event_index[p['event']], station_index[p['station']], phase, seconds(p['time'])))
        self.event, self.station, self.phase, self.time = (np.array(column) for column in zip(*kept))

    def local(self, latitude, longitude):
        """Latitude and longitude (degrees) to x east, y north (km) from the stations' centre."""
        x = EARTH_KM * math.radians(longitude - self.longitude) * math.cos(math.radians(self.latitude))
        return x, EARTH_KM * math.radians(latitude - self.latitude)

    def geographic(self, x, y):
        """x east, y north (km) back to latitude and longitude (degrees)."""
        latitude = self.latitude + np.degrees(np.asarray(y) / EARTH_KM)
        longitude = self.longitude + np.degrees(np.asarray(x) / (EARTH_KM * math.cos(math.radians(self.latitude))))
        return latitude, longitude
