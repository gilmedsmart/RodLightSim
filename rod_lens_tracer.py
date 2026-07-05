"""
Monte-Carlo ray tracer for a PMMA rod with a shaped (radially-symmetric) end face.

Geometry (axis = z):
    - Flat entrance face at z = 0, radius R_rod. Light is injected here.
    - Cylindrical rod body, z in [0, L], radius R_rod (PMMA, TIR at the wall).
    - End "lens" surface at z = L + f(r), r in [0, R_rod], f(R_rod)=0 to meet the
      cylinder edge.  For a sphere of radius Rs:
          f(r) = sqrt(Rs^2 - r^2) - sqrt(Rs^2 - R_rod^2).
    - Flat target plane, TARGET_GAP beyond the lens apex.

Physics: Snell refraction + unpolarized Fresnel reflectance at every glass/air
boundary; total internal reflection handled automatically; Monte-Carlo reflect/
transmit choice.  Depends only on numpy / scipy / matplotlib.

This module is importable:  `simulate(f, fprime, params)` returns the (x,y) hits
on the target so other scripts (e.g. inverse design) can reuse the forward model.
"""

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    R_ROD:   float = 2.5     # rod radius [mm]  (5 mm diameter)
    L:       float = 50.0    # rod length [mm]  (5 cm)
    N_PMMA:  float = 1.49    # PMMA index (~589 nm)
    N_AIR:   float = 1.0
    TARGET_GAP: float = 50.0 # target distance from lens apex [mm]  (5 cm)
    DIV_HALF_DEG: float = 5.0  # source cone half-angle INSIDE the PMMA (0 = collimated)
    MAX_BOUNCES: int = 30

    def z_target(self, f):
        return self.L + float(f(0.0)) + self.TARGET_GAP


EPS = 1e-7


# --------------------------------------------------------------------------- #
# Sphere convenience
# --------------------------------------------------------------------------- #
def make_sphere_profile(Rs, R_rod):
    """(f, fprime) for a convex spherical cap of sphere-radius Rs."""
    edge = np.sqrt(Rs**2 - R_rod**2)
    def f(r):
        r = np.asarray(r, float)
        return np.sqrt(np.maximum(Rs**2 - r**2, 0.0)) - edge
    def fprime(r):
        r = np.asarray(r, float)
        return -r / np.sqrt(np.maximum(Rs**2 - r**2, 1e-12))
    return f, fprime


# --------------------------------------------------------------------------- #
# Optics
# --------------------------------------------------------------------------- #
def fresnel_R(cos_i, cos_t, n1, n2):
    rs = ((n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t))**2
    rp = ((n1*cos_t - n2*cos_i) / (n1*cos_t + n2*cos_i))**2
    return 0.5 * (rs + rp)


def interact(d, n_out, n1, n2, rng):
    """Monte-Carlo Fresnel/TIR. Returns (new_dir, transmitted?)."""
    n = n_out if np.dot(d, n_out) < 0 else -n_out
    cos_i = -np.dot(d, n)
    ratio = n1 / n2
    sin2_t = ratio**2 * (1.0 - cos_i**2)
    if sin2_t > 1.0:                                   # TIR
        return d + 2*cos_i*n, False
    cos_t = np.sqrt(1.0 - sin2_t)
    if rng.random() < fresnel_R(cos_i, cos_t, n1, n2):  # reflect
        return d + 2*cos_i*n, False
    t = ratio*d + (ratio*cos_i - cos_t)*n               # refract
    return t / np.linalg.norm(t), True


# --------------------------------------------------------------------------- #
# Intersections (single ray, meridional-safe 3D)
# --------------------------------------------------------------------------- #
def _hit_cylinder(o, d, R, L):
    a = d[0]**2 + d[1]**2
    if a < 1e-15:
        return np.inf
    b = 2*(o[0]*d[0] + o[1]*d[1])
    c = o[0]**2 + o[1]**2 - R**2
    disc = b*b - 4*a*c
    if disc < 0:
        return np.inf
    sq = np.sqrt(disc)
    for t in sorted(((-b-sq)/(2*a), (-b+sq)/(2*a))):
        if t > EPS and 0.0 <= o[2]+t*d[2] <= L:
            return t
    return np.inf


def _hit_entrance(o, d, R):
    if abs(d[2]) < 1e-15:
        return np.inf
    t = -o[2] / d[2]
    if t > EPS and (o[0]+t*d[0])**2 + (o[1]+t*d[1])**2 <= R**2:
        return t
    return np.inf


def _hit_cap(o, d, f, R, L):
    """First intersection with z = L + f(r), r<=R. Scan for sign change + bisect."""
    hmax = float(f(0.0))
    if d[2] > 1e-9:
        t_end = (L + hmax + 1.0 - o[2]) / d[2]
    else:
        t_end = 3*(L + hmax)
    t_end = max(t_end, EPS*10)

    ts = np.linspace(EPS, t_end, 400)
    P = o[None, :] + ts[:, None]*d[None, :]
    r = np.hypot(P[:, 0], P[:, 1])
    g = P[:, 2] - (L + f(np.minimum(r, R)))
    g[r > R] = np.nan

    for i in range(len(ts)-1):
        g0, g1 = g[i], g[i+1]
        if np.isnan(g0) or np.isnan(g1):
            continue
        if g0 == 0.0:
            return ts[i]
        if g0*g1 < 0.0:
            lo, hi = ts[i], ts[i+1]
            for _ in range(60):
                mid = 0.5*(lo+hi)
                Pm = o + mid*d
                gm = Pm[2] - (L + float(f(min(np.hypot(Pm[0], Pm[1]), R))))
                if g0*gm <= 0:
                    hi = mid
                else:
                    lo, g0 = mid, gm
            t = 0.5*(lo+hi)
            Pt = o + t*d
            if np.hypot(Pt[0], Pt[1]) <= R + 1e-6:
                return t
    return np.inf


def _cap_normal(P, fprime, R):
    r = np.hypot(P[0], P[1])
    if r < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    fp = float(fprime(min(r, R)))
    n = np.array([-fp*P[0]/r, -fp*P[1]/r, 1.0])
    return n / np.linalg.norm(n)


# --------------------------------------------------------------------------- #
# Single-ray trace
# --------------------------------------------------------------------------- #
def _trace_ray(o, d, f, fprime, p, z_target, rng):
    inside = True
    R, L = p.R_ROD, p.L
    for _ in range(p.MAX_BOUNCES):
        if inside:
            tc = _hit_cylinder(o, d, R, L)
            tp = _hit_cap(o, d, f, R, L)
            te = _hit_entrance(o, d, R)
            t = min(tc, tp, te)
            if not np.isfinite(t):
                return None
            P = o + t*d
            if t == tc:
                n_out = np.array([P[0], P[1], 0.0]) / R
            elif t == te:
                n_out = np.array([0.0, 0.0, -1.0])
            else:
                n_out = _cap_normal(P, fprime, R)
            d, transmitted = interact(d, n_out, p.N_PMMA, p.N_AIR, rng)
            o = P
            if transmitted:
                inside = False
        else:
            tp = _hit_cap(o, d, f, R, L)
            tt = (z_target - o[2]) / d[2] if abs(d[2]) > 1e-15 else np.inf
            if EPS < tt < tp:
                P = o + tt*d
                return P[0], P[1]
            if np.isfinite(tp):                        # re-entered glass
                P = o + tp*d
                d, transmitted = interact(d, _cap_normal(P, fprime, R),
                                          p.N_AIR, p.N_PMMA, rng)
                o = P
                if transmitted:
                    inside = True
            else:
                return None
    return None


# --------------------------------------------------------------------------- #
# Source + driver
# --------------------------------------------------------------------------- #
def make_rays(n, p, rng):
    r = p.R_ROD * np.sqrt(rng.random(n))
    phi = 2*np.pi*rng.random(n)
    o = np.column_stack([r*np.cos(phi), r*np.sin(phi), np.full(n, EPS)])
    half = np.radians(p.DIV_HALF_DEG)
    if half <= 0:
        d = np.tile([0.0, 0.0, 1.0], (n, 1))
    else:
        cos_max = np.cos(half)
        cz = 1 - rng.random(n)*(1 - cos_max)
        s = np.sqrt(1 - cz**2)
        az = 2*np.pi*rng.random(n)
        d = np.column_stack([s*np.cos(az), s*np.sin(az), cz])
    return o, d


def simulate(f, fprime, p=Params(), n_rays=40000, seed=0):
    """Trace n_rays; return (hits Nx2 on target plane, fraction_on_target)."""
    rng = np.random.default_rng(seed)
    o, d = make_rays(n_rays, p, rng)
    zt = p.z_target(f)
    hits = []
    for i in range(n_rays):
        res = _trace_ray(o[i].copy(), d[i].copy(), f, fprime, p, zt, rng)
        if res is not None:
            hits.append(res)
    hits = np.array(hits) if hits else np.empty((0, 2))
    return hits, (len(hits)/n_rays if n_rays else 0.0)


# --------------------------------------------------------------------------- #
# Radial irradiance profile, with optional smoothing
# --------------------------------------------------------------------------- #
def radial_profile(hits, nbin=120, rmax=None, smooth=True, savgol_win=11):
    """
    Convert target hits -> (r_centers, irradiance[a.u.]).
    irradiance = counts / annulus_area.  Optional Savitzky-Golay smoothing to
    suppress Monte-Carlo shot noise (worst at small r where the annulus is tiny).
    """
    r = np.hypot(hits[:, 0], hits[:, 1])
    if rmax is None:
        rmax = np.percentile(r, 99.5) if len(r) else 1.0
    edges = np.linspace(0, rmax, nbin+1)
    counts, _ = np.histogram(r, bins=edges)
    area = np.pi*(edges[1:]**2 - edges[:-1]**2)
    irr = counts / area
    rc = 0.5*(edges[1:] + edges[:-1])
    if smooth and nbin >= savgol_win:
        from scipy.signal import savgol_filter
        win = savgol_win if savgol_win % 2 == 1 else savgol_win+1
        irr = np.clip(savgol_filter(irr, win, 3), 0, None)
    return rc, irr


def smoothed_image(hits, span, bins=200, sigma_bins=1.5):
    """2D histogram of target hits, Gaussian-smoothed to reduce shot noise."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(hits[:, 0], hits[:, 1], bins=bins,
                               range=[[-span, span], [-span, span]])
    return gaussian_filter(H.T, sigma_bins), xe, ye


# --------------------------------------------------------------------------- #
# Standalone demo
# --------------------------------------------------------------------------- #
def main():
    import matplotlib.pyplot as plt

    p = Params()
    R_SPHERE = 4.0
    f, fprime = make_sphere_profile(R_SPHERE, p.R_ROD)

    hits, frac = simulate(f, fprime, p, n_rays=80000, seed=0)
    print(f"end face:       spherical Rs={R_SPHERE} mm (apex bulge {f(0.0):.3f} mm)")
    print(f"rays on target: {frac*100:.1f} %")
    if len(hits):
        rr = np.hypot(hits[:, 0], hits[:, 1])
        print(f"spot RMS r:     {np.sqrt(np.mean(rr**2)):.3f} mm")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    if len(hits):
        span = max(1.0, np.percentile(np.abs(hits), 99))
        Himg, xe, ye = smoothed_image(hits, span, bins=200, sigma_bins=1.5)
        im = ax[0].imshow(Himg, origin="lower", cmap="inferno",
                          extent=[xe[0], xe[-1], ye[0], ye[-1]])
        fig.colorbar(im, ax=ax[0], label="irradiance [a.u.] (smoothed)")

        rc, irr_raw = radial_profile(hits, nbin=120, smooth=False)
        rc, irr_sm = radial_profile(hits, nbin=120, smooth=True, savgol_win=11)
        ax[1].plot(rc, irr_raw/irr_sm.max(), color="0.7", lw=1, label="raw (shot noise)")
        ax[1].plot(rc, irr_sm/irr_sm.max(), color="C0", lw=2, label="Savitzky-Golay")
        ax[1].legend(); ax[1].grid(True, alpha=0.3)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"Irradiance on target (z = {p.z_target(f):.1f} mm)")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("y [mm]")
    ax[1].set_title("Radial irradiance profile")
    ax[1].set_xlabel("r on target [mm]"); ax[1].set_ylabel("irradiance [a.u.]")

    fig.tight_layout()
    fig.savefig("target_distribution.png", dpi=130)
    print("saved target_distribution.png")


if __name__ == "__main__":
    main()
