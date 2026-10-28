# Outputs

One folder per run, named by `name` in its config:

```
outputs/example/
  config.toml     the config that made this run
  model.npz       invert: the 3-D model
  gaussians.csv   invert: every Gaussian's velocity change, most significant first
  residuals.csv   invert: every pick's final residual
  summary.json    invert: per iteration, the residual, the estimated noise, the damping and the time
  figures/        plot:   depth slices and a section
```

- `model.npz`: `x_km`, `y_km`, `z_km` (east and north of `origin` [latitude, longitude], depth below sea
  level); `vp`, `vs` and their 1-D starts `vp_start`, `vs_start` (km/s); `coverage_p_km`, `coverage_s_km`,
  the ray length through each node.
- `gaussians.csv`: centre, width, change (km/s) and its standard deviation. The standard deviation covers
  picking noise only, not trade-offs between neighbouring Gaussians: in the example, a fast Gaussian beside
  the slow body is 45 standard deviations from zero and still an artifact. Judge structure on the model.
- `residuals.csv`: observed minus predicted time (s), after removing each event's origin time.

Running again with the same `name` overwrites the folder. Nothing here except this file is tracked by git.
