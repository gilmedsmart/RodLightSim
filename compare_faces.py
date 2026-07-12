"""
Compare three rod end-faces on the SAME source and target:
    1. flat            f(r) = 0
    2. sphere          spherical cap, Rs = 4 mm
    3. designed        inverse-designed for a flat-top over a disk of radius
                       RHO_MAX (see inverse_design.py) -- here 20 mm.

Runs the forward Monte-Carlo tracer for each face, then plots the achieved 2D
irradiance and the radial profiles together, and writes a parameter manifest so
the run is traceable.  Saves everything into a fresh runs/<timestamp>__... folder.

The inverse design assumes a COLLIMATED input, so this comparison is run
collimated (DIV_HALF_DEG = 0) -- an apples-to-apples test of what each face does
to the beam.  (A divergent source would blur all three; ask to redo with div>0.)
"""

import numpy as np
import matplotlib.pyplot as plt

from rod_lens_tracer import (Params, simulate, radial_profile, smoothed_image,
                             make_flat_profile, make_sphere_profile,
                             plot_ray_paths, new_run_dir, write_params_file)
import inverse_design as inv


R_SPHERE = 4.0
N_RAYS = 120000          # cheap now that the tracer is vectorized
SEED = 1
SOURCE_DIV_DEG = 8.0     # source half-angle inside PMMA. 0 = collimated.
                         # >0 = realistic diverging source (rays fan out, beam
                         # widens with distance).  NOTE: the inverse design still
                         # assumes collimated, so the designed face blurs here.
SPAN = inv.RHO_MAX * 1.3          # plot half-width [mm]
TARGET_R = inv.RHO_MAX            # the disk we care about [mm]


def build_faces(p):
    """Return list of (label, descr, f, fprime)."""
    faces = []

    f0, fp0 = make_flat_profile()
    faces.append(("flat", "flat end face, f(r)=0", f0, fp0))

    fs, fps = make_sphere_profile(R_SPHERE, p.R_ROD)
    faces.append(("sphere_Rs%gmm" % R_SPHERE,
                  "spherical cap Rs=%g mm (apex %.3f mm)" % (R_SPHERE, fs(0.0)),
                  fs, fps))

    # inverse-designed flat-top over a disk of radius inv.RHO_MAX
    r, f_vals, fp = inv.design_profile(p)
    fd, fpd = inv.make_callables(r, f_vals, fp)
    faces.append(("designed_flattop_%gmm" % inv.RHO_MAX,
                  "inverse-designed flat-top over r<=%g mm (apex %.3f mm)"
                  % (inv.RHO_MAX, f_vals[0]),
                  fd, fpd))
    return faces


def metrics(hits, target_r):
    """Simple quality numbers for the illuminated disk we care about."""
    if not len(hits):
        return dict(frac_in_disk=0.0, rms_mm=float("nan"), uniformity=float("nan"))
    r = np.hypot(hits[:, 0], hits[:, 1])
    frac_in = float(np.mean(r <= target_r))
    rms = float(np.sqrt(np.mean(r**2)))
    # uniformity inside the disk: std/mean of the radial irradiance over [0, target_r]
    rc, irr = radial_profile(hits[r <= target_r], nbin=40, rmax=target_r,
                             smooth=True, savgol_win=9)
    uni = float(np.std(irr) / np.mean(irr)) if np.mean(irr) > 0 else float("nan")
    return dict(frac_in_disk=frac_in, rms_mm=rms, uniformity=uni)


def main():
    # collimated to match the inverse-design assumption
    p = Params(DIV_HALF_DEG=SOURCE_DIV_DEG)
    run_dir, stamp = new_run_dir("compare_flat_sphere_designed_%gmm" % inv.RHO_MAX)
    print("run folder:", run_dir)

    faces = build_faces(p)

    results_all = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, (label, descr, f, fprime) in enumerate(faces):
        print(f"\n--- {label}: {descr}")
        hits, frac = simulate(f, fprime, p, n_rays=N_RAYS, seed=SEED,
                              progress=True, desc=label)
        m = metrics(hits, TARGET_R)
        print(f"    on target {frac*100:.1f}%  |  within {TARGET_R:g} mm: "
              f"{m['frac_in_disk']*100:.1f}%  |  RMS r {m['rms_mm']:.2f} mm  |  "
              f"uniformity(std/mean) {m['uniformity']:.3f}")
        results_all[label] = dict(on_target_pct=round(frac*100, 2),
                                  within_disk_pct=round(m['frac_in_disk']*100, 2),
                                  rms_r_mm=round(m['rms_mm'], 3),
                                  uniformity_std_over_mean=round(m['uniformity'], 4),
                                  apex_f0_mm=round(float(f(0.0)), 4))

        # top row: 2D image
        ax = axes[0, j]
        if len(hits):
            Himg, xe, ye = smoothed_image(hits, SPAN, bins=200, sigma_bins=1.5)
            im = ax.imshow(Himg, origin="lower", cmap="inferno",
                           extent=[xe[0], xe[-1], ye[0], ye[-1]])
            fig.colorbar(im, ax=ax, fraction=0.046)
        # mark the target disk
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(TARGET_R*np.cos(th), TARGET_R*np.sin(th), "c--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n(within {TARGET_R:g} mm: {m['frac_in_disk']*100:.0f}%)")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

        # bottom row: radial profile
        ax2 = axes[1, j]
        if len(hits):
            rc, irr = radial_profile(hits, nbin=100, rmax=SPAN,
                                     smooth=True, savgol_win=11)
            ax2.plot(rc, irr/irr.max(), color="C1", lw=2, label="achieved")
        ax2.axvspan(0, TARGET_R, color="c", alpha=0.08, label="target disk")
        ax2.axvline(TARGET_R, color="c", ls="--", lw=1)
        ax2.set_ylim(0, 1.25); ax2.set_xlim(0, SPAN)
        ax2.set_xlabel("r on target [mm]"); ax2.set_ylabel("irradiance [a.u.]")
        ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right", fontsize=8)

    src = "collimated" if SOURCE_DIV_DEG == 0 else f"diverging {SOURCE_DIV_DEG:g}deg"
    fig.suptitle(f"End-face comparison  ({src} source)  |  "
                 f"even light over r<={TARGET_R:g} mm  |  {stamp}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cmp_png = run_dir / "compare_faces.png"
    fig.savefig(cmp_png, dpi=130)
    print("\nsaved", cmp_png)

    # per-face ray-path diagrams, using the SAME source divergence as the sim
    for label, descr, f, fprime in faces:
        rp = run_dir / f"ray_paths__{label}.png"
        plot_ray_paths(f, fprime, p, n_rays=63, div_half_deg=SOURCE_DIV_DEG,
                       fname=str(rp), profile_label=descr)
        print("saved", rp)

    # manifest
    lines = [f"{k}: {v}" for k, v in results_all.items()]
    write_params_file(run_dir / "params.txt", p,
                      "3-face comparison; target = flat-top over r<=%g mm" % TARGET_R,
                      dict(timestamp=stamp, n_rays=N_RAYS, seed=SEED,
                           source="collimated (DIV_HALF_DEG=0)",
                           target_disk_mm=TARGET_R,
                           **{f"face[{i}]": faces[i][0] for i in range(len(faces))},
                           **{f"result[{lbl}]": v for lbl, v in results_all.items()}))
    print("saved", run_dir / "params.txt")


if __name__ == "__main__":
    main()
