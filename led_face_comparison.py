"""
LED end-face comparison (the real system).

Fixed setup (locked with the user):
    - Rod: PMMA n=1.49, radius 3 mm (6 mm dia), length L=50 mm.
    - Target: flat screen 80 mm past the end-face apex.
    - Source: OSLON Pure 1414, Lambertian (I ~ cos theta), DIRECT-COUPLED to the
      PMMA at z=0 (no air gap) -> emitted straight into the glass, up to 90 deg
      inside.  Radially symmetric emitter disk of diameter 1.4 mm, centered.
    - Success metric: uniformity over the 15 mm-RADIUS disc (primary merit);
      efficiency into 15 mm reported too.  Screen shown out to 30 mm radius.

Three end faces compared (only f(r) changes):
    1. flat        f(r) = 0
    2. sphere      convex cap, Rs = 5 mm
    3. designed    inverse-designed 15 mm flat-top (built on a COLLIMATED
                   assumption; included for comparison, expected to underperform)

Writes a timestamped runs/ folder with: compare_led_faces.png, per-face
ray_paths__*.png, and config.txt (full parameter + metric record).
"""

import numpy as np
import matplotlib.pyplot as plt

from rod_lens_tracer import (Params, simulate, radial_profile, smoothed_image,
                             make_flat_profile, make_sphere_profile,
                             plot_ray_paths, new_run_dir)
import inverse_design as inv

# --------------------------------------------------------------------------- #
# Locked configuration
# --------------------------------------------------------------------------- #
RS_SPHERE   = 5.0        # convex sphere radius [mm]
LED_DIA_MM  = 1.4        # LED emitter diameter [mm]
TARGET_R    = 15.0       # success disc radius [mm]
SCREEN_R    = 30.0       # how far out to show results [mm]
N_RAYS      = 40000      # first pass; bump to 150000+ for cleaner maps
SEED        = 1
MAX_BOUNCES = 10         # a ray vanishes after this many "jumps" (may change later)
WALL_REFLECT = 0.99      # reflective side coating: 99% reflect, 1% absorbed

P = Params(
    R_ROD=3.0, L=50.0, N_PMMA=1.49, TARGET_GAP=80.0,
    MAX_BOUNCES=MAX_BOUNCES,
    WALL_MIRROR=True, WALL_REFLECTIVITY=WALL_REFLECT,
    SOURCE="led", LED_DIRECT_COUPLE=True, LED_VIEW_DEG=120.0,
    EMITTER_R=LED_DIA_MM/2.0,
)


def metrics(hits):
    if not len(hits):
        return dict(eff=0.0, uni=float("nan"), rms=float("nan"), r95=float("nan"))
    r = np.hypot(hits[:, 0], hits[:, 1])
    eff = float(np.mean(r <= TARGET_R))                      # of on-target rays
    sub = hits[r <= TARGET_R]
    if len(sub) > 50:
        rc, irr = radial_profile(sub, nbin=30, rmax=TARGET_R, smooth=True, savgol_win=7)
        uni = float(np.std(irr) / np.mean(irr)) if np.mean(irr) > 0 else float("nan")
    else:
        uni = float("nan")
    return dict(eff=eff, uni=uni, rms=float(np.sqrt(np.mean(r**2))),
                r95=float(np.percentile(r, 95)))


def build_faces():
    faces = []
    f0, fp0 = make_flat_profile()
    faces.append(("flat", "flat end face, f(r)=0", f0, fp0))

    fs, fps = make_sphere_profile(RS_SPHERE, P.R_ROD)
    faces.append(("sphere_Rs%gmm" % RS_SPHERE,
                  "convex sphere Rs=%g mm (apex %.3f mm)" % (RS_SPHERE, fs(0.0)),
                  fs, fps))

    r, fv, fp = inv.design_profile(P)                        # collimated-assumption design
    fd, fpd = inv.make_callables(r, fv, fp)
    faces.append(("designed_flattop_%gmm" % inv.RHO_MAX,
                  "inverse-designed %g mm flat-top (collimated assumption; apex %.3f mm)"
                  % (inv.RHO_MAX, fv[0]), fd, fpd))
    return faces


def main():
    run_dir, stamp = new_run_dir("LED_flat_sphere_designed_%gmm" % TARGET_R)
    print("run folder:", run_dir)
    faces = build_faces()

    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, (label, descr, f, fprime) in enumerate(faces):
        print(f"\n--- {label}: {descr}")
        hits, frac = simulate(f, fprime, P, n_rays=N_RAYS, seed=SEED,
                              progress=True, desc=label)
        m = metrics(hits)
        # efficiency into the disc, as a fraction of ALL emitted rays
        eff_total = m["eff"] * frac
        print(f"    on-target {frac*100:.1f}% | within {TARGET_R:g}mm "
              f"{eff_total*100:.1f}% of emitted | uniformity {m['uni']:.3f} | "
              f"RMS {m['rms']:.1f}mm | r95 {m['r95']:.1f}mm")
        results[label] = dict(descr=descr,
                              on_target_pct=round(frac*100, 2),
                              within15_pct_of_emitted=round(eff_total*100, 2),
                              uniformity_std_over_mean=round(m["uni"], 4),
                              rms_r_mm=round(m["rms"], 2),
                              r95_mm=round(m["r95"], 2),
                              apex_f0_mm=round(float(f(0.0)), 4))

        ax = axes[0, j]
        if len(hits):
            Himg, xe, ye = smoothed_image(hits, SCREEN_R, bins=220, sigma_bins=1.5)
            im = ax.imshow(Himg, origin="lower", cmap="inferno",
                           extent=[xe[0], xe[-1], ye[0], ye[-1]])
            fig.colorbar(im, ax=ax, fraction=0.046)
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(TARGET_R*np.cos(th), TARGET_R*np.sin(th), "c--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nwithin15: {eff_total*100:.0f}% emitted | "
                     f"unif {m['uni']:.2f}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

        ax2 = axes[1, j]
        if len(hits):
            rc, irr = radial_profile(hits, nbin=90, rmax=SCREEN_R, smooth=True, savgol_win=11)
            ax2.plot(rc, irr/irr.max(), color="C1", lw=2, label="achieved")
        ax2.axvspan(0, TARGET_R, color="c", alpha=0.08, label="15 mm disc")
        ax2.axvline(TARGET_R, color="c", ls="--", lw=1)
        ax2.set_xlim(0, SCREEN_R); ax2.set_ylim(0, 1.25)
        ax2.set_xlabel("r on target [mm]"); ax2.set_ylabel("irradiance [a.u.]")
        ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"LED end-face comparison  |  OSLON Pure 1414 Lambertian, "
                 f"direct-coupled  |  L=50 mm, gap=80 mm  |  {stamp}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cmp_png = run_dir / "compare_led_faces.png"
    fig.savefig(cmp_png, dpi=130)
    print("\nsaved", cmp_png)

    for label, descr, f, fprime in faces:
        rp = run_dir / f"ray_paths__{label}.png"
        plot_ray_paths(f, fprime, P, n_rays=60, fname=str(rp), profile_label=descr)
        print("saved", rp)

    write_config(run_dir / "config.txt", stamp, faces, results)
    print("saved", run_dir / "config.txt")


def write_config(path, stamp, faces, results):
    L = []
    W = L.append
    W("LED END-FACE COMPARISON — RUN RECORD")
    W("=" * 60)
    W(f"timestamp: {stamp}")
    W("")
    W("LOCKED CONFIGURATION")
    W("-" * 60)
    W(f"  rod material / index      : PMMA, n = {P.N_PMMA}")
    W(f"  rod radius                : {P.R_ROD} mm  (diameter {2*P.R_ROD} mm)")
    W(f"  rod length L              : {P.L} mm")
    W(f"  target distance (apex)    : {P.TARGET_GAP} mm")
    W(f"  source                    : OSLON Pure 1414, Lambertian (I ~ cos theta)")
    W(f"  LED coupling              : DIRECT to PMMA, no air gap "
      f"(LED_DIRECT_COUPLE={P.LED_DIRECT_COUPLE})")
    W(f"  LED viewing angle         : {P.LED_VIEW_DEG} deg (full)")
    W(f"  LED emitter               : radially-symmetric disk, "
      f"diameter {2*P.EMITTER_R} mm, centered")
    W(f"  internal angle range      : full hemisphere, up to 90 deg inside PMMA")
    W(f"  side wall                 : MIRROR, {P.WALL_REFLECTIVITY*100:.0f}% reflective "
      f"({(1-P.WALL_REFLECTIVITY)*100:.0f}% absorbed/hit), no side transmission")
    W(f"  ray lifetime              : vanishes after {P.MAX_BOUNCES} jumps (bounces)")
    W(f"  success metric            : uniformity over {TARGET_R} mm-RADIUS disc")
    W(f"  screen shown out to       : {SCREEN_R} mm radius")
    W("")
    W("IMPLEMENTATION")
    W("-" * 60)
    W(f"  rays per face             : {N_RAYS}")
    W(f"  RNG seed                  : {SEED}")
    W(f"  max bounces               : {P.MAX_BOUNCES}")
    W(f"  tracer engine             : vectorized (Snell + unpolarized Fresnel + TIR)")
    W(f"  efficiency denominator    : total emitted rays")
    W(f"  uniformity                : std/mean of radial irradiance over 0..{TARGET_R} mm")
    W("")
    W("END FACES")
    W("-" * 60)
    for i, (label, descr, _f, _fp) in enumerate(faces, 1):
        W(f"  {i}. {label}: {descr}")
    W("")
    W("RESULTS  (uniformity = primary merit, lower is flatter)")
    W("-" * 60)
    W(f"  {'face':<26}{'within15%':>10}{'unif15':>9}{'RMS mm':>9}{'r95 mm':>9}")
    for label, r in results.items():
        W(f"  {label:<26}{r['within15_pct_of_emitted']:>10}"
          f"{r['uniformity_std_over_mean']:>9}{r['rms_r_mm']:>9}{r['r95_mm']:>9}")
    W("")
    W("PER-FACE DETAIL")
    W("-" * 60)
    for label, r in results.items():
        W(f"  {label}:")
        for k, v in r.items():
            W(f"       {k:<26} = {v}")
    W("")
    from pathlib import Path
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
