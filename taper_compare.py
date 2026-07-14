"""
Tapered mirror-wall collimator: compare exit-face shapes (flat vs convex).

Fixed: PMMA rod L=50 mm, LED (OSLON 1414) direct-coupled at the small end,
99% mirror TAPERED wall (entrance dia 2 mm -> exit dia 8 mm, the 8 mm max),
N=30 jumps, screen 80 mm past the apex.  Success disc = 15 mm radius.

Compares end faces f(r): flat, and convex spheres of a few radii.  Saves a
comparison figure (2D maps + encircled energy) + per-face ray graphs + config.txt.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")            # file-only rendering; avoids GUI-backend flakiness
import matplotlib.pyplot as plt
from rod_lens_tracer import (Params, simulate, radial_profile, smoothed_image,
                             make_flat_profile, make_sphere_profile,
                             plot_system_rays, new_run_dir, wall_params)

R_ENTRANCE = 1.0         # entrance radius [mm]  (2 mm dia)
R_EXIT     = 4.0         # exit radius   [mm]  (8 mm dia, the max allowed)
N_RAYS     = 80000
SEED       = 1
TARGET_R   = 45.0        # inspected output radius (uniformity reported over this)
SCREEN_R   = 60.0        # 2D map half-width [mm]
PROFILE_R  = 90.0        # encircled-energy curve extent [mm]

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
    all_r = []                                     # (label, radii) for the dP/dr overlay
    # 3 stacked graphs per simulation: 2D map / encircled power / intensity-vs-r
    fig, axes = plt.subplots(3, len(FACES), figsize=(4.6*len(FACES), 12.6))
    for j, (label, (f, fprime)) in enumerate(FACES):
        print(f"\n--- {label} (apex f0={f(0.0):.3f} mm)")
        hits, frac, fates = simulate(f, fprime, P, n_rays=N_RAYS, seed=SEED,
                                     progress=True, desc=label, return_fates=True)
        r = np.hypot(hits[:, 0], hits[:, 1]); E = fates["emitted"]
        all_r.append((label, r))
        inT = 100*np.mean(r <= TARGET_R)      # within the inspected radius (45 mm)
        in30 = 100*np.mean(r <= 30.0); in15 = 100*np.mean(r <= 15.0)
        sub = hits[r <= TARGET_R]             # uniformity over 0..TARGET_R
        if len(sub) > 50:
            rc, irr = radial_profile(sub, nbin=45, rmax=TARGET_R, smooth=True, savgol_win=9)
            uni = float(np.std(irr)/np.mean(irr))
        else:
            uni = float("nan")
        results[label] = dict(apex_mm=round(float(f(0.0)), 3), reach_pct=round(frac*100, 1),
                              in15_pct=round(in15, 1), in30_pct=round(in30, 1),
                              in45_pct=round(inT, 1),
                              uniformity45=round(uni, 3), median_r_mm=round(float(np.median(r)), 1))
        print(f"    reach {frac*100:.1f}% | in15 {in15:.1f}% | in30 {in30:.1f}% | "
              f"in45 {inT:.1f}% | uni(0-45) {uni:.3f} | medR {np.median(r):.0f}mm")

        ax = axes[0, j]
        Himg, xe, ye = smoothed_image(hits, SCREEN_R, bins=200, sigma_bins=1.5)
        ax.imshow(Himg, origin="lower", cmap="inferno", extent=[xe[0], xe[-1], ye[0], ye[-1]])
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(TARGET_R*np.cos(th), TARGET_R*np.sin(th), "c--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title("%s\nin45=%.0f%%  in30=%.0f%%  uni(0-45)=%.2f" % (label, inT, in30, uni))
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

        ax2 = axes[1, j]
        rsort = np.sort(r); rr = np.linspace(0, PROFILE_R, 240)
        enc = np.searchsorted(rsort, rr, side="right")/E*100.0
        ax2.plot(rr, enc, "C0", lw=2)
        ax2.axvspan(0, TARGET_R, color="c", alpha=0.08); ax2.axvline(TARGET_R, color="c", ls="--", lw=1)
        ax2.set_xlim(0, PROFILE_R); ax2.set_ylim(0, 100)
        ax2.set_xlabel("r on screen [mm]"); ax2.set_ylabel("encircled energy [% emitted]")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("encircled power: %.0f%% @30mm  %.0f%% @45mm" % (in30, inT))

        # intensity / irradiance vs r  (power per AREA -> what the screen "looks like")
        ax3 = axes[2, j]
        rc, irr = radial_profile(hits, nbin=60, rmax=SCREEN_R, smooth=True, savgol_win=9)
        ax3.plot(rc, irr, color="C3", lw=2)
        ax3.axvspan(0, TARGET_R, color="c", alpha=0.08); ax3.axvline(TARGET_R, color="c", ls="--", lw=1)
        ax3.set_xlim(0, SCREEN_R); ax3.set_ylim(bottom=0)
        ax3.set_xlabel("r on screen [mm]"); ax3.set_ylabel("intensity [a.u.]  (per area)")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("intensity vs r  (uniformity 0-45 = %.2f)" % uni)

        plot_system_rays(f, fprime, P, n_rays=60, target_r=TARGET_R, screen_r=SCREEN_R,
                         fname=str(run_dir / ("system_rays__%s.png" % label)),
                         profile_label="taper 2->8mm, exit=%s" % label)

    fig.suptitle("Tapered collimator (2->8 mm wall, 99%% mirror, N=30) — exit-face comparison  |  %s"
                 % stamp, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cmp_png = run_dir / "taper_exitface_compare.png"
    fig.savefig(cmp_png, dpi=130); plt.close(fig); print("\nsaved", cmp_png)

    # ---- radial power distribution dP/dr (power landing per mm of radius) ----
    figP, axP = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, PROFILE_R, int(PROFILE_R) + 1)     # 1 mm rings
    centers = 0.5*(bins[:-1] + bins[1:]); width = bins[1] - bins[0]
    for label, rr in all_r:
        counts, _ = np.histogram(rr, bins=bins)
        dpdr = counts / N_RAYS * 100.0 / width              # % of emitted per mm
        axP.plot(centers, dpdr, lw=2, label=label)
    for rm, c in [(15, "0.7"), (30, "0.7"), (TARGET_R, "c")]:
        axP.axvline(rm, color=c, ls="--" if c == "c" else ":", lw=1.2)
    axP.set_xlim(0, PROFILE_R); axP.set_ylim(bottom=0)
    axP.set_xlabel("r from center [mm]")
    axP.set_ylabel("power per mm of radius  [% of emitted / mm]")
    axP.set_title("Radial power distribution  dP/dr  —  tapered collimator exit faces\n"
                  "(area under each curve = % of light reaching the screen)")
    axP.grid(True, alpha=0.3); axP.legend()
    figP.tight_layout()
    pow_png = run_dir / "power_vs_radius.png"
    figP.savefig(pow_png, dpi=130); plt.close(figP); print("saved", pow_png)

    lines = ["TAPERED COLLIMATOR — EXIT-FACE COMPARISON", "="*60, f"timestamp: {stamp}", "",
             "CONFIG", "-"*60,
             f"  rod: PMMA n={P.N_PMMA}, L={P.L} mm, TAPERED wall",
             f"  entrance dia {2*R_ENTRANCE} mm -> exit dia {2*R_EXIT} mm (max 8)",
             f"  wall: 99% mirror, N={P.MAX_BOUNCES} jumps",
             f"  source: OSLON 1414 Lambertian, direct-coupled, emitter dia {2*P.EMITTER_R} mm",
             f"  target screen {P.TARGET_GAP} mm past apex; inspected radius {TARGET_R} mm",
             f"  rays {N_RAYS}, seed {SEED}", "",
             "RESULTS (uniformity = std/mean over 0-%gmm; %% of emitted)" % TARGET_R, "-"*66,
             f"  {'exit face':<14}{'apex':>7}{'reach%':>8}{'in15%':>7}{'in30%':>7}"
             f"{'in45%':>7}{'uni45':>7}{'medR':>7}"]
    for k, v in results.items():
        lines.append(f"  {k:<14}{v['apex_mm']:>7}{v['reach_pct']:>8}{v['in15_pct']:>7}"
                     f"{v['in30_pct']:>7}{v['in45_pct']:>7}{v['uniformity45']:>7}{v['median_r_mm']:>7}")
    lines.append("")
    (run_dir / "config.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", run_dir / "config.txt")


if __name__ == "__main__":
    main()
