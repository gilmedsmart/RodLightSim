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
PROFILE_R = 60.0         # radial intensity profile shown r = 0 .. this [mm]

P = Params(
    R_ROD=3.0, L=50.0, N_PMMA=1.49, TARGET_GAP=80.0, MAX_BOUNCES=30,
    WALL_MIRROR=True, WALL_REFLECTIVITY=0.99,
    SOURCE="led", LED_DIRECT_COUPLE=True, LED_VIEW_DEG=120.0, EMITTER_R=0.7,
)


def main():
    f, fprime = make_sphere_profile(RS, P.R_ROD)
    run_dir, stamp = new_run_dir("LED_sphere_Rs%gmm_mirror_N%d" % (RS, P.MAX_BOUNCES))
    print("run folder:", run_dir)

    hits, frac, fates = simulate(f, fprime, P, n_rays=N_RAYS, seed=SEED,
                                 progress=True, desc="sphere Rs%g" % RS,
                                 return_fates=True)
    r = np.hypot(hits[:, 0], hits[:, 1]) if len(hits) else np.array([0.0])

    # ---- ray-fate breakdown (percent of ALL emitted rays; sums to 100) ----
    E = fates["emitted"]
    n_hit = int(np.sum(r <= TARGET_R))                       # reached screen AND in disc
    n_reach = fates["reached_screen"]
    n_miss = n_reach - n_hit                                 # reached screen, outside disc
    n_absorbed = fates["absorbed_wall"]                     # died on the 99% coating
    n_maxb = fates["decayed_maxbounce"]                    # died at the jump limit
    n_decay_rod = n_absorbed + n_maxb                       # total "decayed inside the rod"
    n_backair = fates["exited_air_missed"]                 # left rod (e.g. backward), missed screen
    pct = lambda k: 100.0 * k / E if E else 0.0

    sub = hits[r <= TARGET_R]
    if len(sub) > 50:
        rc, irr = radial_profile(sub, nbin=30, rmax=TARGET_R, smooth=True, savgol_win=7)
        uni = float(np.std(irr) / np.mean(irr))
    else:
        uni = float("nan")
    rms = float(np.sqrt(np.mean(r**2)))
    r95 = float(np.percentile(r, 95))
    within15 = pct(n_hit) / 100.0

    print(f"\nRAY FATES (% of {E} emitted):")
    print(f"  hit target (<= {TARGET_R:g} mm)       : {pct(n_hit):6.2f}%  ({n_hit})")
    print(f"  exit, miss target (> {TARGET_R:g} mm) : {pct(n_miss):6.2f}%  ({n_miss})")
    print(f"  decay inside rod            : {pct(n_decay_rod):6.2f}%  ({n_decay_rod})")
    print(f"     - absorbed by wall coat  : {pct(n_absorbed):6.2f}%  ({n_absorbed})")
    print(f"     - decayed at jump limit  : {pct(n_maxb):6.2f}%  ({n_maxb})")
    print(f"  exit backward, miss screen  : {pct(n_backair):6.2f}%  ({n_backair})")
    print(f"  uniformity {uni:.3f} | RMS {rms:.1f}mm | r95 {r95:.1f}mm")

    graph = run_dir / ("system_rays__sphere_Rs%gmm.png" % RS)
    plot_system_rays(f, fprime, P, n_rays=GRAPH_RAYS, target_r=TARGET_R,
                     screen_r=SCREEN_R, fname=str(graph),
                     profile_label="convex sphere Rs=%g mm (hemisphere) [99%% mirror, N=%d]"
                     % (RS, P.MAX_BOUNCES))

    # ---- radial intensity profile + encircled energy, r = 0 .. PROFILE_R ----
    import matplotlib.pyplot as plt
    rc_raw, irr_raw = radial_profile(hits, nbin=120, rmax=PROFILE_R, smooth=False)
    rc, irr = radial_profile(hits, nbin=120, rmax=PROFILE_R, smooth=True, savgol_win=11)
    # encircled energy: fraction of ALL emitted rays landing within radius r
    rsort = np.sort(r)
    rr = np.linspace(0, PROFILE_R, 240)
    enc = np.searchsorted(rsort, rr, side="right") / E * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rc_raw, irr_raw, color="0.8", lw=1, label="irradiance (raw)")
    ax.plot(rc, irr, color="C1", lw=2, label="irradiance (smoothed)")
    ax.axvspan(0, TARGET_R, color="c", alpha=0.08, label="%g mm target disc" % TARGET_R)
    ax.axvline(TARGET_R, color="c", ls="--", lw=1.2)
    ax.set_xlim(0, PROFILE_R); ax.set_ylim(bottom=0)
    ax.set_xlabel("r on screen [mm]")
    ax.set_ylabel("irradiance [a.u.]  (counts / annulus area)", color="C1")

    ax2 = ax.twinx()                              # encircled energy on the right axis
    ax2.plot(rr, enc, color="C0", lw=2.2, label="encircled energy")
    ax2.set_ylabel("encircled energy  [% of emitted]", color="C0")
    ax2.set_ylim(0, max(1.0, enc[-1]*1.15))
    for rmark in (TARGET_R, 30.0):
        if rmark <= PROFILE_R:
            val = np.interp(rmark, rr, enc)
            ax2.plot([rmark], [val], "o", color="C0", ms=5)
            ax2.annotate("%.1f%% @ %gmm" % (val, rmark), (rmark, val),
                         (rmark+2, val+1), color="C0", fontsize=8)

    ax.set_title("Radial intensity + encircled energy  —  sphere Rs=%g mm, "
                 "LED direct, 99%% mirror, N=%d" % (RS, P.MAX_BOUNCES))
    l1, la = ax.get_legend_handles_labels()
    l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la+lb, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    prof = run_dir / ("radial_intensity__sphere_Rs%gmm.png" % RS)
    fig.savefig(prof, dpi=130)
    print("saved", prof)

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
        "RESULTS — ray fates (%% of %d emitted; sum = 100%%)" % E,
        "-" * 60,
        f"  hit target (<= {TARGET_R:g} mm)        : {pct(n_hit):6.2f} %   ({n_hit})",
        f"  exit, miss target (> {TARGET_R:g} mm)  : {pct(n_miss):6.2f} %   ({n_miss})",
        f"  decay inside rod             : {pct(n_decay_rod):6.2f} %   ({n_decay_rod})",
        f"      - absorbed by wall coat  : {pct(n_absorbed):6.2f} %   ({n_absorbed})",
        f"      - decayed at jump limit  : {pct(n_maxb):6.2f} %   ({n_maxb})",
        f"  exit backward, miss screen   : {pct(n_backair):6.2f} %   ({n_backair})",
        "",
        f"  reaches screen (hit+miss)    : {frac*100:.2f} % of emitted",
        f"  uniformity (std/mean, 0-15mm): {uni:.4f}",
        f"  RMS radius                   : {rms:.2f} mm",
        f"  r95 (95% within)             : {r95:.2f} mm",
        "",
    ]
    (run_dir / "config.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", run_dir / "config.txt")


if __name__ == "__main__":
    main()
