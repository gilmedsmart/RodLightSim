# RodLightSim

Monte-Carlo optical ray tracer for a **PMMA cylindrical rod with a shaped
radially-symmetric end face**, plus an **inverse designer** that solves for the
end-face profile `f(r)` producing a prescribed target irradiance.

Dependencies: `numpy`, `scipy`, `matplotlib` (no mesh/CAD engine required — the
geometry is analytic).

## Geometry
- Flat entrance face at `z = 0`, radius `R_rod`, where light is injected.
- Cylindrical PMMA rod, `z ∈ [0, L]` (TIR at the wall).
- End "lens" surface `z = L + f(r)`, `r ∈ [0, R_rod]`, with `f(R_rod) = 0`.
  A sphere of radius `Rs` corresponds to `f(r) = sqrt(Rs² − r²) − sqrt(Rs² − R_rod²)`.
- Flat target plane a set distance beyond the lens apex.

Physics: Snell refraction + unpolarized Fresnel reflectance + total internal
reflection at every glass/air boundary, evaluated by Monte-Carlo.

## Scripts
- **`rod_lens_tracer.py`** — forward tracer. Run directly for a demo (default
  spherical end, `Rs = 4 mm`); produces `target_distribution.png`. Importable:
  `simulate(f, fprime, params)` returns the `(x, y)` hits on the target.
  Change the end face by editing `f(r)` / `fprime(r)`.
- **`inverse_design.py`** — set `E_target(rho)` (flat-top, ring, Gaussian, …),
  and it solves for `f(r)` via energy conservation + Snell inversion, then
  verifies with the forward tracer. Produces `inverse_design.png` and
  `designed_profile.csv` (`r, f, f'`).

## Usage
```bash
python rod_lens_tracer.py      # forward simulation + plots
python inverse_design.py       # inverse design + verification
```

## Design assumptions (inverse mode)
Collimated, uniform input; single thin refracting surface; monotonic `r → ρ`
mapping; Fresnel losses ignored in the design step (included in the MC
verification). A single surface can only bend rays up to the TIR limit.
