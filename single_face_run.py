"""
Single run: hemispherical end face (sphere Rs = rod radius = 3 mm), real LED,
99% mirror side walls, N=10 jumps.  Produces ONLY the full-system ray graph
(z=0..screen, radius +/-30 mm) plus a config.txt record with the metrics.
"""
import numpy as np
from rod_lens_tracer import (Params, simulate, radial_profile,
                             make_sphere_profile, plot_system_rays, new_run_dir)

RS        = 3.0          # sphere radius of curvature [mm] (= rod radius -> hemisphere)
N_RAYS    = 80000
SEED      = 1
GRAPH_RAYS = 90          # meridional rays drawn in the ray graph (readability)
TARGET_R  = 15.0
SCREEN_R  = 30.0

P = Params(
    R_ROD=3.0, L=50.0, N_PMMA=1.49, TARGET_GAP=80.0, MAX_BOUNCES=30,
    WALL_MIRROR=True, WALL_REFLECTIVITY=0.99,
    SOURCE="led", LED_DIRECT_COUPLE=True, LED_VIEW_DEG=120.0, EMITTER_R=0.7,
)


def main():
    f, fprime = make_sphere_profile(RS, P.R_ROD)
    run_dir, stamp = new_run_dir("LED_sphere_Rs%gmm_mirror_N%d" % (RS, P.MAX_BOUNCES))
    print("run folder:", run_dir)

    hits, frac = simulate(f, fprime, P, n_rays=N_RAYS, seed=SEED,
                          progress=True, desc="sphere Rs%g" % RS)
    r = np.hypot(hits[:, 0], hits[:, 1]) if len(hits) else np.array([0.0])
    within15 = float(np.mean(r <= TARGET_R) * frac)          # of ALL emitted rays
    sub = hits[r <= TARGET_R]
    if len(sub) > 50:
        rc, irr = radial_profile(sub, nbin=30, rmax=TARGET_R, smooth=True, savgol_win=7)
        uni = float(np.std(irr) / np.mean(irr))
    else:
        uni = float("nan")
    rms = float(np.sqrt(np.mean(r**2)))
    r95 = float(np.percentile(r, 95))
    print(f"\nreaches screen {frac*100:.1f}% | within 15mm {within15*100:.2f}% of emitted "
          f"| uniformity {uni:.3f} | RMS {rms:.1f}mm | r95 {r95:.1f}mm")

    graph = run_dir / ("system_rays__sphere_Rs%gmm.png" % RS)
    plot_system_rays(f, fprime, P, n_rays=GRAPH_RAYS, target_r=TARGET_R,
                     screen_r=SCREEN_R, fname=str(graph),
                     profile_label="convex sphere Rs=%g mm (hemisphere) [99%% mirror, N=%d]"
                     % (RS, P.MAX_BOUNCES))

    lines = [
        "SINGLE-FACE RUN RECORD",
        "=" * 60,
        f"timestamp: {stamp}",
        "",
        "CONFIGURATION",
        "-" * 60,
        f"  rod material / index   : PMMA, n = {P.N_PMMA}",
        f"  rod radius             : {P.R_ROD} mm (diameter {2*P.R_ROD} mm)",
        f"  rod length L           : {P.L} mm",
        f"  target distance (apex) : {P.TARGET_GAP} mm",
        f"  end face               : convex sphere Rs = {RS} mm  "
        f"(= rod radius -> hemisphere; apex bulge {f(0.0):.3f} mm)",
        f"  source                 : OSLON Pure 1414, Lambertian, direct-coupled "
        f"(no air gap), emitter dia {2*P.EMITTER_R} mm centered",
        f"  internal angle range   : up to 90 deg inside PMMA",
        f"  side wall              : MIRROR {P.WALL_REFLECTIVITY*100:.0f}% reflective "
        f"({(1-P.WALL_REFLECTIVITY)*100:.0f}% absorbed/hit), no side transmission",
        f"  ray lifetime           : vanishes after {P.MAX_BOUNCES} jumps",
        f"  success metric         : uniformity over {TARGET_R} mm-radius disc",
        f"  screen shown to        : {SCREEN_R} mm radius",
        "",
        "IMPLEMENTATION",
        "-" * 60,
        f"  rays (statistics)      : {N_RAYS}",
        f"  rays drawn in graph    : {GRAPH_RAYS} meridional",
        f"  RNG seed               : {SEED}",
        f"  tracer engine          : vectorized (Snell + Fresnel + TIR + mirror wall)",
        "",
        "RESULTS",
        "-" * 60,
        f"  reaches screen         : {frac*100:.2f} % of emitted",
        f"  within 15 mm disc      : {within15*100:.2f} % of emitted",
        f"  uniformity (std/mean)  : {uni:.4f}",
        f"  RMS radius             : {rms:.2f} mm",
        f"  r95 (95% within)       : {r95:.2f} mm",
        "",
    ]
    (run_dir / "config.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", run_dir / "config.txt")


if __name__ == "__main__":
    main()
