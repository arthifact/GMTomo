# Data

Put three CSV files in one folder, for example `data/my_region/`, and set `folder` in `config.toml` to it.

`stations.csv`

| station | latitude | longitude | elevation_m |
|---|---|---|---|
| ST01 | -33.4521 | -70.6611 | 520 |

`events.csv`

| event | latitude | longitude | depth_km |
|---|---|---|---|
| EQ0001 | -33.1020 | -71.2045 | 34.2 |

`picks.csv`

| event | station | phase | time |
|---|---|---|---|
| EQ0001 | ST01 | P | 2026-03-01T04:12:09.31Z |

- **phase**: anything starting with P or S counts (P, Pg, Pn, S, Sg, ...); other phases are skipped.
- **time**: seconds or ISO 8601, on any clock. Origin times are not needed.
- **events**: the catalogue locations, where relocation starts (`relocate = false` in `config.toml` keeps them).

`uv run src/example.py` writes a synthetic example to `data/example/`. Nothing here except this file is
tracked by git.
