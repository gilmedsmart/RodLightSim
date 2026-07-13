"""
Tapered mirror-wall collimator: compare exit-face shapes (flat vs convex).

Fixed: PMMA rod L=50 mm, LED (OSLON 1414) direct-coupled at the small end,
99% mirror TAPERED wall (entrance dia 2 mm -> exit dia 8 mm, the 8 mm max),
N=30 jumps, screen 80 mm past the apex.  Success disc = 15 mm radius.

Compares end faces f(r): flat, and convex spheres of a few radii.  Saves a
comparison figure (2D maps + encircled energy) + per-face ray graphs + config.txt.
"""
import numpy as np
import matplotlib.pyplot as plt
from rod_lens_tracer import (Params, simulate, radial_profile, smoothed_image,
                             make_flat_profile, make_sphere_profile,
                             plot_system_rays, new_run_dir, wall_params)

R_ENTRANCE = 1.0         # entrance radius [mm]  (2 mm dia)
R_EXIT     = 4.0         # exit radius   [mm]  (8 mm dia, the max allowed)
N_RAYS     = 80000
SEED       = 1
TARGET_R   = 15.0
SCREEN_R   = 30.0
PROFILE_R  = 60.0

P = Params(R_ROD=R_EXIT, L=50.0, N_PMMA=1.49, TARGET_GAP=80.0, MAX_BOUNCES=30,
           WALL_MIRROR=True, WALL_REFLECTIVITY=0.99,
           WALL_TAPER=True, R_ENTRANCE=R_ENTRANCE,
           SOURCE="led", LED_DIRECT_COUPLE=True, LED_VIEW_DEG=120.0, EMITTER_R=0.7)

FACES = [("flat",       make_flat_profile()),
         ("convex_Rs8", make_sphere_profile(8.0, R_EXIT)),
         ("convex_Rs6", make_sphere_profile(6.0, R_EXIT)),
         ("convex_Rs5", make_sphere_profile(5.0, R_EXIT))]


def main():
    run_dir, stamp = new_run_dir("TAPER_2to8mm_exitface_compare")
    print("run folder:", run_dir)
    print("wall (R0,k,Rexit):", wall_params(P))

    results = {}
    fig, axes = plt.subplots(2, len(FACES), figsize=(4.6*len(FACES), 8.6))
    for j, (label, (f, fprime)) in enumerate(FACES):
        print(f"\n--- {label} (apex f0={f(0.0):.3f} mm)")
        hits, frac, fates = simulate(f, fprime, P, n_rays=N_RAYS, seed=SEED,
                                     progress=True, desc=label, return_fates=True)
        r = np.hypot(hits[:, 0], hits[:, 1]); E = fates["emitted"]
        in15 = 100*np.mean(r <= TARGET_R); in30 = 100*np.mean(r <= 30.0)
        sub = hits[r <= TARGET_R]
        if len(sub) > 50:
            rc, irr = radial_profile(sub, nbin=30, rmax=TARGET_R, smooth=True, savgol_win=7)
            uni = float(np.std(irr)/np.mean(irr))
        else:
            uni = float("nan")
        results[label] = dict(apex_mm=round(float(f(0.0)), 3), reach_pct=round(frac*100, 1),
                              in15_pct=round(in15, 1), in30_pct=round(in30, 1),
                              uniformity=round(uni, 3), median_r_mm=round(float(np.median(r)), 1))
        print(f"    reach {frac*100:.1f}% | in15 {in15:.1f}% | in30 {in30:.1f}% | "
              f"uni {uni:.3f} | medR {np.median(r):.0f}mm")

        ax = axes[0, j]
        Himg, xe, ye = smoothed_image(hits, SCREEN_R, bins=200, sigma_bins=1.5)
        ax.imshow(Himg, origin="lower", cmap="inferno", extent=[xe[0], xe[-1], ye[0], ye[-1]])
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(TARGET_R*np.cos(th), TARGET_R*np.sin(th), "c--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title("%s\nin15=%.0f%%  in30=%.0f%%  uni=%.2f" % (label, in15, in30, uni))
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

        ax2 = axes[1, j]
        rsort = np.sort(r); rr = np.linspace(0, PROFILE_R, 240)
        enc = np.searchsorted(rsort, rr, side="right")/E*100.0
        ax2.plot(rr, enc, "C0", lw=2)
        ax2.axvspan(0, TARGET_R, color="c", alpha=0.08); ax2.axvline(TARGET_R, color="c", ls="--", lw=1)
        ax2.set_xlim(0, PROFILE_R); ax2.set_ylim(0, 100)
        ax2.set_xlabel("r on screen [mm]"); ax2.set_ylabel("encircled energy [% emitted]")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("%.0f%% @15mm   %.0f%% @30mm" % (in15, in30))

        plot_system_rays(f, fprime, P, n_rays=60, target_r=TARGET_R, screen_r=SCREEN_R,
                         fname=str(run_dir / ("system_rays__%s.png" % label)),
                         profile_label="taper 2->8mm, exit=%s" % label)

    fig.suptitle("Tapered collimator (2->8 mm wall, 99%% mirror, N=30) — exit-face comparison  |  %s"
                 % stamp, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cmp_png = run_dir / "taper_exitface_compare.png"
    fig.savefig(cmp_png, dpi=130); print("\nsaved", cmp_png)

    lines = ["TAPERED COLLIMATOR — EXIT-FACE COMPARISON", "="*60, f"timestamp: {stamp}", "",
             "CONFIG", "-"*60,
             f"  rod: PMMA n={P.N_PMMA}, L={P.L} mm, TAPERED wall",
             f"  entrance dia {2*R_ENTRANCE} mm -> exit dia {2*R_EXIT} mm (max 8)",
             f"  wall: 99% mirror, N={P.MAX_BOUNCES} jumps",
             f"  source: OSLON 1414 Lambertian, direct-coupled, emitter dia {2*P.EMITTER_R} mm",
             f"  target screen {P.TARGET_GAP} mm past apex; success disc {TARGET_R} mm radius",
             f"  rays {N_RAYS}, seed {SEED}", "",
             "RESULTS (uniformity over 0-15mm; % of emitted)", "-"*60,
             f"  {'exit face':<14}{'apex':>7}{'reach%':>8}{'in15%':>8}{'in30%':>8}{'uni':>7}{'medR':>7}"]
    for k, v in results.items():
        lines.append(f"  {k:<14}{v['apex_mm']:>7}{v['reach_pct']:>8}{v['in15_pct']:>8}"
                     f"{v['in30_pct']:>8}{v['uniformity']:>7}{v['median_r_mm']:>7}")
    lines.append("")
    (run_dir / "config.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", run_dir / "config.txt")


if __name__ == "__main__":
    main()
