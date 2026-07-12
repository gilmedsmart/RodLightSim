"""
INVERSE design: find the end-face profile f(r) that turns the rod's output into a
PRESCRIBED target irradiance E_t(rho).

Method (analytic first-order design, then Monte-Carlo verification):

  Assumptions for the design step
    - Input light is COLLIMATED along the axis inside the PMMA and has uniform
      irradiance over the entrance disk (radius R_rod).
    - Single refracting surface, thin (a ray at input radius r meets the surface
      near radius r), radially symmetric, monotonic r -> rho mapping.
    - Fresnel losses ignored in the design (the MC verification includes them).

  1. Ray mapping by energy conservation.
     Uniform input: power fraction inside radius r is  u(r) = (r/R)^2.
     Target:        C(rho) = normalized cumulative power of E_t inside rho.
     Solve  C(rho(r)) = u(r)  ->  rho(r)   (a light ray from r must land at rho).

  2. Required deflection.
     Over axial gap D = z_target - (L + f), the ray must go from radius r to rho:
         theta(r) = atan2(rho(r) - r, D).

  3. Snell inversion for the local surface slope.
     A collimated ray (0,1) hitting a surface whose outward normal is tilted by
     beta from the axis refracts (n_pmma -> n_air) to some output angle. Solve for
     beta giving theta(r); the surface slope is  f'(r) = -tan(beta).

  4. Integrate f'(r) inward from the edge with f(R_rod)=0.

Then the forward Monte-Carlo tracer is run with this f(r) to check the achieved
distribution against the requested one.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from rod_lens_tracer import Params, simulate, radial_profile, interact


# --------------------------------------------------------------------------- #
# 1. TARGET irradiance you want on the plane.  Edit E_target(rho).
# --------------------------------------------------------------------------- #
RHO_MAX = 15.0   # outer radius of the desired pattern on the target [mm]

def E_target(rho):
    """Desired irradiance vs radius (un-normalized). Examples below."""
    rho = np.asarray(rho, float)
    # ---- flat-top disk of radius RHO_MAX ----
    return np.where(rho <= RHO_MAX, 1.0, 0.0)
    # ---- annulus (ring) between 4 and 6 mm ----
    # return np.where((rho >= 4.0) & (rho <= RHO_MAX), 1.0, 0.0)
    # ---- Gaussian of sigma 2 mm ----
    # return np.exp(-0.5*(rho/2.0)**2)
    # ---- linear ramp (bright edge) ----
    # return np.where(rho <= RHO_MAX, rho, 0.0)


# --------------------------------------------------------------------------- #
# 2. Ray mapping r -> rho by energy conservation
# --------------------------------------------------------------------------- #
def build_mapping(p, nrho=4000):
    rho = np.linspace(0, RHO_MAX, nrho)
    e = np.clip(E_target(rho), 0, None)
    C = np.concatenate([[0], np.cumsum(0.5*(e[1:]+e[:-1]) *
                        np.pi*(rho[1:]**2 - rho[:-1]**2))])   # cumulative power
    if C[-1] <= 0:
        raise ValueError("E_target integrates to zero.")
    C /= C[-1]
    # invert C(rho)=u  ->  rho(u)
    Cu, idx = np.unique(C, return_index=True)
    rho_of_u = interp1d(Cu, rho[idx], bounds_error=False,
                        fill_value=(0.0, RHO_MAX))

    def rho_of_r(r):
        u = (np.asarray(r, float) / p.R_ROD)**2           # uniform input cdf
        return rho_of_u(np.clip(u, 0, 1))
    return rho_of_r


# --------------------------------------------------------------------------- #
# 3. Snell inversion: output angle of a collimated axial ray vs normal tilt beta
# --------------------------------------------------------------------------- #
def output_angle(beta, n1, n2):
    """Refract d=(0,1) through a surface with outward normal at tilt beta; return
    the output ray angle from the axis (signed, in the r-z meridian)."""
    d = np.array([0.0, 1.0, 0.0])                 # +z, using (r, z, dummy)
    N = np.array([np.sin(beta), np.cos(beta), 0.0])
    dout, transmitted = interact(d, N, n1, n2, _NOREFLECT)
    if not transmitted:                            # TIR
        return np.nan
    return np.arctan2(dout[0], dout[1])


class _NoReflect:
    """A fake rng that never triggers Fresnel reflection (always transmit)."""
    def random(self):
        return 1.0
_NOREFLECT = _NoReflect()


def slope_for_angle(theta, n1, n2):
    """Find surface slope f' = -tan(beta) giving output angle theta."""
    beta_max = np.arcsin(min(1.0, n2/n1)) - 1e-4   # TIR limit on incidence
    lo, hi = -beta_max, beta_max
    g = lambda b: output_angle(b, n1, n2) - theta
    glo, ghi = g(lo), g(hi)
    if np.isnan(glo) or np.isnan(ghi) or glo*ghi > 0:
        # requested angle not reachable with one surface -> clamp to nearest
        betas = np.linspace(lo, hi, 201)
        outs = np.array([output_angle(b, n1, n2) for b in betas])
        k = np.nanargmin(np.abs(outs - theta))
        return -np.tan(betas[k]), True
    beta = brentq(g, lo, hi, xtol=1e-10)
    return -np.tan(beta), False


# --------------------------------------------------------------------------- #
# 4. Integrate f'(r) inward from the edge, f(R)=0
# --------------------------------------------------------------------------- #
def design_profile(p, nr=400, n_outer_iters=3):
    rho_of_r = build_mapping(p)
    r = np.linspace(0, p.R_ROD, nr)

    D0 = p.TARGET_GAP                     # axial gap surface->target (updated below)
    f_vals = np.zeros(nr)                 # start with flat surface
    clamped = False

    for _ in range(n_outer_iters):
        # axial gap for each r: target is at z = L + f(0) + TARGET_GAP
        z_target = p.L + f_vals[np.argmin(r)] + p.TARGET_GAP  # ~ L + f(0) + gap
        D = z_target - (p.L + f_vals)
        D = np.maximum(D, 1.0)

        fp = np.zeros(nr)
        for i, ri in enumerate(r):
            rho = float(rho_of_r(ri))
            theta = np.arctan2(rho - ri, D[i])
            s, cl = slope_for_angle(theta, p.N_PMMA, p.N_AIR)
            fp[i] = s
            clamped = clamped or cl
        fp[0] = 0.0                       # symmetry: flat at the axis

        # integrate inward from edge: f(R)=0, f(r) = -∫_r^R f'(s) ds
        f_new = np.zeros(nr)
        for i in range(nr-2, -1, -1):
            f_new[i] = f_new[i+1] - 0.5*(fp[i]+fp[i+1])*(r[i+1]-r[i])
        f_vals = f_new

    if clamped:
        print("  [warn] some rays needed more deflection than one surface can give;"
              " slope was clamped (single surface can't reach that angle).")

    return r, f_vals, fp


# --------------------------------------------------------------------------- #
# Assemble callable f, fprime from the samples and verify with the MC tracer
# --------------------------------------------------------------------------- #
def make_callables(r, f_vals, fp):
    f_i  = interp1d(r, f_vals, kind="cubic", bounds_error=False,
                    fill_value=(f_vals[0], 0.0))
    fp_i = interp1d(r, fp,     kind="cubic", bounds_error=False,
                    fill_value=(0.0, fp[-1]))
    return (lambda rr: np.asarray(f_i(np.clip(np.asarray(rr, float), r[0], r[-1])), float),
            lambda rr: np.asarray(fp_i(np.clip(np.asarray(rr, float), r[0], r[-1])), float))


def main():
    # design assumes collimated input -> verify with a collimated source
    p = Params(DIV_HALF_DEG=0.0)

    print("Designing f(r) for the requested target distribution...")
    r, f_vals, fp = design_profile(p)
    f, fprime = make_callables(r, f_vals, fp)
    print(f"  apex bulge f(0) = {f_vals[0]:.3f} mm, edge f(R) = {f_vals[-1]:.3f} mm")

    print("Verifying with the forward Monte-Carlo tracer...")
    hits, frac = simulate(f, fprime, p, n_rays=40000, seed=1,
                          progress=True, desc="verify")
    print(f"  rays on target: {frac*100:.1f} %")

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].plot(r, f_vals, lw=2)
    ax[0].set_title("Designed end-face profile f(r)")
    ax[0].set_xlabel("r [mm]"); ax[0].set_ylabel("bulge height f [mm]")
    ax[0].grid(True, alpha=0.3); ax[0].set_aspect("auto")

    # requested vs achieved radial irradiance (both normalized to unit peak)
    rho = np.linspace(0, RHO_MAX*1.3, 300)
    want = E_target(rho); want = want/want.max() if want.max() > 0 else want
    ax[1].plot(rho, want, "k--", lw=2, label="requested")
    if len(hits):
        rc, irr = radial_profile(hits, nbin=90, rmax=RHO_MAX*1.3,
                                 smooth=True, savgol_win=11)
        ax[1].plot(rc, irr/irr.max(), color="C1", lw=2, label="achieved (MC)")
    ax[1].set_title("Radial irradiance: requested vs achieved")
    ax[1].set_xlabel("rho on target [mm]"); ax[1].set_ylabel("irradiance [a.u.]")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

    if len(hits):
        span = RHO_MAX*1.3
        from rod_lens_tracer import smoothed_image
        Himg, xe, ye = smoothed_image(hits, span, bins=180, sigma_bins=1.5)
        im = ax[2].imshow(Himg, origin="lower", cmap="inferno",
                          extent=[xe[0], xe[-1], ye[0], ye[-1]])
        fig.colorbar(im, ax=ax[2], label="irradiance [a.u.]")
    ax[2].set_aspect("equal")
    ax[2].set_title("Achieved 2D distribution")
    ax[2].set_xlabel("x [mm]"); ax[2].set_ylabel("y [mm]")

    fig.tight_layout()
    fig.savefig("inverse_design.png", dpi=130)
    print("saved inverse_design.png")

    # also dump the profile so it can be reused/manufactured
    np.savetxt("designed_profile.csv",
               np.column_stack([r, f_vals, fp]),
               delimiter=",", header="r_mm,f_mm,fprime", comments="")
    print("saved designed_profile.csv")


if __name__ == "__main__":
    main()
