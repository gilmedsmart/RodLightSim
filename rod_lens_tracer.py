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

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    R_ROD:   float = 3.0     # rod radius [mm]  (6 mm diameter)
    L:       float = 50.0    # rod length [mm]  (5 cm)
    N_PMMA:  float = 1.49    # PMMA index (~589 nm)
    N_AIR:   float = 1.0
    TARGET_GAP: float = 80.0 # target distance from lens apex [mm]  (8 cm)
    MAX_BOUNCES: int = 30    # a ray vanishes after this many interactions ("jumps")
    # ---- side wall ----
    WALL_MIRROR: bool = False       # True = reflective coating on the rod side wall
                             #  (specular reflect, never transmits out the side)
    WALL_REFLECTIVITY: float = 0.99  # [WALL_MIRROR] reflect prob/hit; rest absorbed
    WALL_TAPER: bool = False        # True = conical wall: radius goes R_ENTRANCE at
                             #  z=0 -> R_ROD at z=L (R_ROD is then the EXIT radius)
    R_ENTRANCE: float = 1.0         # [WALL_TAPER] wall radius at the LED end [mm]
    # ---- source model ----
    SOURCE:  str = "led"     # "led" = centered Lambertian LED; "cone" = uniform
                             #  disk + uniform cone (legacy / idealized)
    DIV_HALF_DEG: float = 5.0  # ["cone"] half-angle INSIDE the PMMA (0=collimated)
    EMITTER_R: float = 0.5   # ["led"] emitter radius [mm]; OSLON Pure 1414 ~1 mm sq
    LED_VIEW_DEG: float = 120.0  # ["led"] full viewing angle (120 = Lambertian)
    LED_DIRECT_COUPLE: bool = True  # ["led"] True = LED bonded to PMMA, no air gap
                             #  (Lambertian emitted directly into glass, up to 90
                             #  deg inside); False = air gap (Snell caps at ~42 deg)

    def z_target(self, f):
        return self.L + float(f(0.0)) + self.TARGET_GAP


EPS = 1e-7


# --------------------------------------------------------------------------- #
# End-face profile conveniences
# --------------------------------------------------------------------------- #
def make_flat_profile():
    """(f, fprime) for a plain flat end face (f(r) = 0): the rod is just a
    cylinder cut square.  Rays exit through the flat glass/air boundary."""
    def f(r):
        return np.zeros_like(np.asarray(r, float))
    def fprime(r):
        return np.zeros_like(np.asarray(r, float))
    return f, fprime


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
def _trace_ray(o, d, f, fprime, p, z_target, rng, path=None):
    inside = True
    L = p.L
    R0, kwall, Rexit = wall_params(p)
    if path is not None:
        path.append(o.copy())
    for _ in range(p.MAX_BOUNCES):
        if inside:
            tc = _hit_cone(o, d, R0, kwall, L)
            tp = _hit_cap(o, d, f, Rexit, L)
            te = _hit_entrance(o, d, R0)
            t = min(tc, tp, te)
            if not np.isfinite(t):
                return None
            P = o + t*d
            if t == tc and getattr(p, "WALL_MIRROR", False):
                # reflective side wall: specular reflect, absorb (1-R) fraction
                n = _cone_normal(P, kwall)
                d = d - 2*np.dot(d, n)*n
                o = P
                if path is not None:
                    path.append(P.copy())
                if rng.random() >= p.WALL_REFLECTIVITY:
                    return None                    # absorbed by the coating
                continue                            # stays inside the glass
            if t == tc:
                n_out = _cone_normal(P, kwall)
            elif t == te:
                n_out = np.array([0.0, 0.0, -1.0])
            else:
                n_out = _cap_normal(P, fprime, Rexit)
            d, transmitted = interact(d, n_out, p.N_PMMA, p.N_AIR, rng)
            o = P
            if path is not None:
                path.append(P.copy())
            if transmitted:
                inside = False
        else:
            tp = _hit_cap(o, d, f, Rexit, L)
            tt = (z_target - o[2]) / d[2] if abs(d[2]) > 1e-15 else np.inf
            if EPS < tt < tp:
                P = o + tt*d
                if path is not None:
                    path.append(P.copy())
                return P[0], P[1]
            if np.isfinite(tp):                        # re-entered glass
                P = o + tp*d
                d, transmitted = interact(d, _cap_normal(P, fprime, Rexit),
                                          p.N_AIR, p.N_PMMA, rng)
                o = P
                if path is not None:
                    path.append(P.copy())
                if transmitted:
                    inside = True
            else:
                return None
    return None


# --------------------------------------------------------------------------- #
# Source + driver
# --------------------------------------------------------------------------- #
def make_rays(n, p, rng):
    if getattr(p, "SOURCE", "cone") == "led":
        return _make_led_rays(n, p, rng)
    # ---- legacy: uniform disk over the whole face + uniform cone ----
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


def _make_led_rays(n, p, rng):
    """A single small LED centered on the entrance face (uniform disk of radius
    EMITTER_R), emitting a Lambertian-ish profile (intensity ~ cos^m theta; m from
    LED_VIEW_DEG; m=1 is a true Lambertian / 120 deg viewing angle).

    LED_DIRECT_COUPLE=True: the LED is bonded to the PMMA (no air gap), so the
    cos^m distribution is emitted DIRECTLY into the glass, spanning up to 90 deg
    inside.  =False: emitted into air then refracted at the flat entrance (Snell
    caps the internal cone near the 42 deg critical angle, Fresnel reflection at
    the entrance handled by rejection).  Returns rays launched at z=EPS inside."""
    # cos^m exponent so intensity halves at half the viewing angle
    half_view = np.radians(p.LED_VIEW_DEG) / 2.0
    m = 1.0 if p.LED_VIEW_DEG >= 119.5 else np.log(0.5) / np.log(max(np.cos(half_view), 1e-6))

    def emit_positions(k):
        re = p.EMITTER_R * np.sqrt(rng.random(k))
        pe = 2*np.pi*rng.random(k)
        return re*np.cos(pe), re*np.sin(pe)

    if getattr(p, "LED_DIRECT_COUPLE", True):
        # Lambertian emitted straight into the glass; no interface, exactly n rays.
        x, y = emit_positions(n)
        cos_p = (1.0 - rng.random(n)) ** (1.0 / (m + 1.0))   # pdf ~ cos^m * sin
        sin_p = np.sqrt(np.clip(1.0 - cos_p**2, 0.0, None))
        phi = 2*np.pi*rng.random(n)
        o = np.column_stack([x, y, np.full(n, EPS)])
        d = np.column_stack([sin_p*np.cos(phi), sin_p*np.sin(phi), cos_p])
        return o, d

    # air gap: emit in air, refract air->glass, reject the Fresnel-reflected part
    n_glass = p.N_PMMA
    O, D = [], []
    got = 0
    while got < n:
        k = int((n - got) * 1.5) + 32
        x, y = emit_positions(k)
        cos_air = np.clip((1.0 - rng.random(k)) ** (1.0 / (m + 1.0)), -1.0, 1.0)
        sin_air = np.sqrt(1.0 - cos_air**2)
        phi = 2*np.pi*rng.random(k)
        sin_t = sin_air / n_glass
        cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, None))
        rs = ((cos_air - n_glass*cos_t) / (cos_air + n_glass*cos_t + 1e-12))**2
        rp = ((cos_t - n_glass*cos_air) / (cos_t + n_glass*cos_air + 1e-12))**2
        T = 1.0 - 0.5*(rs + rp)
        acc = rng.random(k) < T
        dr = sin_t
        O.append(np.column_stack([x[acc], y[acc], np.full(int(acc.sum()), EPS)]))
        D.append(np.column_stack([dr[acc]*np.cos(phi[acc]), dr[acc]*np.sin(phi[acc]), cos_t[acc]]))
        got += int(acc.sum())
    return np.vstack(O)[:n], np.vstack(D)[:n]


# --------------------------------------------------------------------------- #
# Vectorized batch optics (all rays advance together — the fast path)
# --------------------------------------------------------------------------- #
def _fresnel_R_v(cos_i, cos_t, n1, n2):
    rs = ((n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t))**2
    rp = ((n1*cos_t - n2*cos_i) / (n1*cos_t + n2*cos_i))**2
    return 0.5*(rs + rp)


def _interact_v(d, n_out, n1, n2, rng):
    """Vectorized Fresnel/TIR for M rays.  Returns (new_dir MxN, transmitted?)."""
    n = n_out.copy()
    flip = np.einsum('ij,ij->i', d, n) > 0        # make n oppose d
    n[flip] = -n[flip]
    cos_i = -np.einsum('ij,ij->i', d, n)
    ratio = n1 / n2
    sin2_t = ratio*ratio * (1.0 - cos_i*cos_i)
    tir = sin2_t > 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    Rref = _fresnel_R_v(cos_i, cos_t, n1, n2)
    reflect = tir | (rng.random(len(d)) < Rref)
    d_ref = d + 2*cos_i[:, None]*n
    t = ratio*d + (ratio*cos_i - cos_t)[:, None]*n
    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    d_tr = t / nrm
    new_d = np.where(reflect[:, None], d_ref, d_tr)
    return new_d, ~reflect


def _hit_cyl_v(o, d, R, L):
    a = d[:, 0]**2 + d[:, 1]**2
    b = 2*(o[:, 0]*d[:, 0] + o[:, 1]*d[:, 1])
    c = o[:, 0]**2 + o[:, 1]**2 - R*R
    disc = b*b - 4*a*c
    ok = (disc >= 0) & (a > 1e-15)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t = np.full(len(o), np.inf)
    for sgn in (-1.0, 1.0):
        tt = (-b + sgn*sq) / np.where(a > 1e-15, 2*a, 1.0)
        z = o[:, 2] + tt*d[:, 2]
        good = ok & (tt > EPS) & (z >= 0.0) & (z <= L) & (tt < t)
        t = np.where(good, tt, t)
    return t


def wall_params(p):
    """(R0, k, R_exit): the wall radius is r(z) = R0 + k*z.  Straight cylinder is
    the special case k=0 (R0 = R_exit = R_ROD)."""
    R_exit = p.R_ROD
    R0 = p.R_ENTRANCE if getattr(p, "WALL_TAPER", False) else R_exit
    k = (R_exit - R0) / p.L
    return R0, k, R_exit


def _hit_cone_v(o, d, R0, k, L):
    """First intersection with the wall of revolution r(z)=R0+k*z, 0<=z<=L."""
    ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    rw0 = R0 + k*oz
    A = dx*dx + dy*dy - (k*dz)**2
    B = 2*(ox*dx + oy*dy) - 2*k*dz*rw0
    C = ox*ox + oy*oy - rw0*rw0
    t = np.full(len(o), np.inf)
    quad = np.abs(A) > 1e-12
    disc = B*B - 4*A*C
    okq = quad & (disc >= 0)
    sq = np.sqrt(np.where(okq, disc, 0.0))
    for sgn in (-1.0, 1.0):
        tt = np.where(quad, (-B + sgn*sq) / np.where(quad, 2*A, 1.0), np.inf)
        z = oz + tt*dz
        good = okq & (tt > EPS) & (z >= 0.0) & (z <= L) & (tt < t)
        t = np.where(good, tt, t)
    lin = ~quad & (np.abs(B) > 1e-15)              # A~0: linear B t + C = 0
    tl = np.where(lin, -C / np.where(lin, B, 1.0), np.inf)
    zl = oz + tl*dz
    goodl = lin & (tl > EPS) & (zl >= 0.0) & (zl <= L) & (tl < t)
    t = np.where(goodl, tl, t)
    return t


def _cone_normal_v(P, k):
    rw = np.hypot(P[:, 0], P[:, 1])                # = R0 + k z on the wall
    n = np.column_stack([P[:, 0], P[:, 1], -k*rw])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _hit_cone(o, d, R0, k, L):
    """Scalar cone-wall intersection (for the single-ray path tracer)."""
    ox, oy, oz = o
    dx, dy, dz = d
    rw0 = R0 + k*oz
    A = dx*dx + dy*dy - (k*dz)**2
    B = 2*(ox*dx + oy*dy) - 2*k*dz*rw0
    C = ox*ox + oy*oy - rw0*rw0
    cands = []
    if abs(A) > 1e-12:
        disc = B*B - 4*A*C
        if disc >= 0:
            sq = np.sqrt(disc)
            cands = [(-B - sq)/(2*A), (-B + sq)/(2*A)]
    elif abs(B) > 1e-15:
        cands = [-C/B]
    best = np.inf
    for tt in sorted(cands):
        if tt > EPS and 0.0 <= oz + tt*dz <= L and tt < best:
            best = tt
    return best


def _cone_normal(P, k):
    rw = np.hypot(P[0], P[1])
    n = np.array([P[0], P[1], -k*rw])
    return n / np.linalg.norm(n)


def _hit_entrance_v(o, d, R):
    okz = np.abs(d[:, 2]) > 1e-15
    tt = np.where(okz, -o[:, 2] / np.where(okz, d[:, 2], 1.0), np.inf)
    x = o[:, 0] + tt*d[:, 0]
    y = o[:, 1] + tt*d[:, 1]
    good = okz & (tt > EPS) & (x*x + y*y <= R*R)
    return np.where(good, tt, np.inf)


def _cap_normal_v(P, fprime, R):
    r = np.hypot(P[:, 0], P[:, 1])
    small = r < 1e-9
    rs = np.where(small, 1.0, r)
    fp = np.asarray(fprime(np.minimum(r, R)), float)
    nx = np.where(small, 0.0, -fp*P[:, 0]/rs)
    ny = np.where(small, 0.0, -fp*P[:, 1]/rs)
    n = np.stack([nx, ny, np.ones_like(r)], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _hit_cap_v(o, d, f, R, L, hmax, K=160, nbis=40):
    """Vectorized first intersection with z = L + f(r), r<=R (scan + bisect)."""
    M = len(o)
    if M == 0:
        return np.empty(0)
    dz = d[:, 2]
    t_end = np.where(dz > 1e-9, (L + hmax + 1.0 - o[:, 2]) / np.where(dz > 1e-9, dz, 1.0),
                     3.0*(L + hmax))
    t_end = np.maximum(t_end, EPS*10)

    def gvals(t):                                  # t: (M,) or (M,K)
        t3 = t[..., None]
        P = o[:, None, :] + t3*d[:, None, :] if t.ndim == 2 else o + t3*d
        if t.ndim == 2:
            r = np.hypot(P[:, :, 0], P[:, :, 1])
            g = P[:, :, 2] - (L + np.asarray(f(np.minimum(r, R)), float))
            return g, r
        r = np.hypot(P[:, 0], P[:, 1])
        return P[:, 2] - (L + np.asarray(f(np.minimum(r, R)), float)), r

    s = np.linspace(0.0, 1.0, K)
    t = EPS + s[None, :]*(t_end[:, None] - EPS)     # (M,K)
    g, r = gvals(t)
    g = np.where(r > R, np.nan, g)
    g0, g1 = g[:, :-1], g[:, 1:]
    valid = np.isfinite(g0) & np.isfinite(g1)
    cross = valid & ((g0*g1 < 0.0) | (g0 == 0.0))
    has = cross.any(axis=1)
    first = np.argmax(cross, axis=1)                # 0 if no crossing
    rows = np.arange(M)
    lo = t[rows, first]
    hi = t[rows, first + 1]
    glo, _ = gvals(lo)
    for _ in range(nbis):
        mid = 0.5*(lo + hi)
        gm, _ = gvals(mid)
        take_hi = glo*gm <= 0.0                     # root in [lo, mid]
        hi = np.where(take_hi, mid, hi)
        lo = np.where(take_hi, lo, mid)
        glo = np.where(take_hi, glo, gm)
    t_hit = 0.5*(lo + hi)
    _, rr = gvals(t_hit)
    return np.where(has & (rr <= R + 1e-6), t_hit, np.inf)


def _simulate_vec(f, fprime, p, n_rays, seed, progress, desc, return_fates=False):
    import time
    rng = np.random.default_rng(seed)
    o, d = make_rays(n_rays, p, rng)
    o = np.ascontiguousarray(o, float); d = np.ascontiguousarray(d, float)
    zt = p.z_target(f)
    L, hmax = p.L, float(f(0.0))
    R0, kwall, Rexit = wall_params(p)              # wall r(z) = R0 + kwall*z

    inside = np.ones(n_rays, bool)
    alive = np.ones(n_rays, bool)
    landed = np.zeros(n_rays, bool)
    hit = np.full((n_rays, 2), np.nan)
    # per-ray fate: 0=unresolved, 1=reached screen, 2=absorbed by wall coating,
    # 3=exited into air missing the screen (e.g. backward out the entrance)
    reason = np.zeros(n_rays, np.int8)
    t0 = time.time()

    for bounce in range(p.MAX_BOUNCES):
        act = alive & ~landed
        if not act.any():
            break
        ins = np.where(act & inside)[0]
        out = np.where(act & ~inside)[0]

        # ---- rays currently inside the glass ----
        if len(ins):
            oi, di = o[ins], d[ins]
            tc = _hit_cone_v(oi, di, R0, kwall, L)
            te = _hit_entrance_v(oi, di, R0)
            tp = _hit_cap_v(oi, di, f, Rexit, L, hmax)
            t = np.minimum(np.minimum(tc, tp), te)
            lost = ~np.isfinite(t)
            alive[ins[lost]] = False
            reason[ins[lost]] = 3
            gd = ~lost
            gi = ins[gd]
            if len(gi):
                tg = t[gd]
                P = o[gi] + tg[:, None]*d[gi]
                o[gi] = P
                is_c = tg == tc[gd]
                is_e = (tg == te[gd]) & ~is_c
                is_p = ~is_c & ~is_e

                # cap + entrance: normal glass/air Snell+Fresnel
                refr = is_e | is_p
                if refr.any():
                    idx = gi[refr]
                    nrm = np.empty((int(refr.sum()), 3))
                    ie, ip = is_e[refr], is_p[refr]
                    if ie.any():
                        nrm[ie] = np.array([0.0, 0.0, -1.0])
                    if ip.any():
                        nrm[ip] = _cap_normal_v(P[refr][ip], fprime, Rexit)
                    new_d, transmitted = _interact_v(d[idx], nrm, p.N_PMMA, p.N_AIR, rng)
                    d[idx] = new_d
                    inside[idx[transmitted]] = False

                # side wall
                if is_c.any():
                    idx = gi[is_c]
                    Pc = P[is_c]
                    n = _cone_normal_v(Pc, kwall)
                    if getattr(p, "WALL_MIRROR", False):
                        # reflective coating: specular reflect, absorb (1-R)
                        dot = np.einsum('ij,ij->i', d[idx], n)
                        d[idx] = d[idx] - 2*dot[:, None]*n
                        absorbed = rng.random(len(idx)) >= p.WALL_REFLECTIVITY
                        alive[idx[absorbed]] = False
                        reason[idx[absorbed]] = 2
                    else:
                        new_d, transmitted = _interact_v(d[idx], n, p.N_PMMA, p.N_AIR, rng)
                        d[idx] = new_d
                        inside[idx[transmitted]] = False

        # ---- rays currently outside, in air past the cap ----
        if len(out):
            oo, do = o[out], d[out]
            tp = _hit_cap_v(oo, do, f, Rexit, L, hmax)
            dz = do[:, 2]
            okz = np.abs(dz) > 1e-15
            tt = np.where(okz, (zt - oo[:, 2]) / np.where(okz, dz, 1.0), np.inf)
            reach = (tt > EPS) & (tt < tp)
            ri = out[reach]
            if len(ri):
                P = o[ri] + tt[reach][:, None]*d[ri]
                hit[ri] = P[:, :2]
                landed[ri] = True
                reason[ri] = 1
            rest = ~reach
            reenter = rest & np.isfinite(tp)
            ei = out[reenter]
            if len(ei):
                P = o[ei] + tp[reenter][:, None]*d[ei]
                new_d, transmitted = _interact_v(d[ei], _cap_normal_v(P, fprime, Rexit),
                                                 p.N_AIR, p.N_PMMA, rng)
                o[ei] = P
                d[ei] = new_d
                inside[ei[transmitted]] = True
            gone = out[rest & ~np.isfinite(tp)]
            alive[gone] = False
            reason[gone] = 3

        if progress:
            resolved = float((landed | ~alive).mean())
            print(f"    {desc}: {100*resolved:5.1f}% resolved  (bounce {bounce+1}/"
                  f"{p.MAX_BOUNCES}, active {int((alive & ~landed).sum())}, "
                  f"{time.time()-t0:4.1f}s)", flush=True)

    reason[reason == 0] = 4                    # still bouncing at the jump limit
    hits = hit[landed]
    frac = (int(landed.sum())/n_rays if n_rays else 0.0)
    if return_fates:
        fates = dict(emitted=int(n_rays),
                     reached_screen=int((reason == 1).sum()),
                     absorbed_wall=int((reason == 2).sum()),
                     exited_air_missed=int((reason == 3).sum()),
                     decayed_maxbounce=int((reason == 4).sum()))
        return hits, frac, fates
    return hits, frac


def _simulate_scalar(f, fprime, p, n_rays, seed, progress, progress_step, desc):
    """Original single-ray-at-a-time engine.  Kept for cross-checking the fast
    vectorized path; ~50-100x slower."""
    import time
    rng = np.random.default_rng(seed)
    o, d = make_rays(n_rays, p, rng)
    zt = p.z_target(f)
    hits = []
    step = max(1, int(n_rays * progress_step)) if progress else 0
    t0 = time.time()
    for i in range(n_rays):
        res = _trace_ray(o[i].copy(), d[i].copy(), f, fprime, p, zt, rng)
        if res is not None:
            hits.append(res)
        if progress and ((i + 1) % step == 0 or i + 1 == n_rays):
            done = i + 1
            el = time.time() - t0
            print(f"    {desc}: {100*done/n_rays:5.1f}%  ({done}/{n_rays} rays)"
                  f"  elapsed {el:4.1f}s  eta {el*(n_rays-done)/done:4.1f}s", flush=True)
    hits = np.array(hits) if hits else np.empty((0, 2))
    return hits, (len(hits)/n_rays if n_rays else 0.0)


def simulate(f, fprime, p=Params(), n_rays=40000, seed=0,
             progress=False, progress_step=0.05, desc="tracing", engine="vec",
             return_fates=False):
    """Trace n_rays; return (hits Nx2 on target plane, fraction_on_target).

    engine="vec" (default) advances all rays together with numpy — ~50-100x
    faster than engine="scalar" (the reference per-ray loop).  Both use the same
    physics (Snell + unpolarized Fresnel + TIR).  progress=True prints progress
    (per-bounce for vec, every `progress_step` fraction for scalar), flushed so
    it shows live even when piped to a log.  return_fates=True (vec only) also
    returns a dict counting each ray's outcome (reached screen / absorbed by the
    wall / exited into air missing the screen / decayed at the jump limit)."""
    if engine == "scalar":
        return _simulate_scalar(f, fprime, p, n_rays, seed, progress, progress_step, desc)
    return _simulate_vec(f, fprime, p, n_rays, seed, progress, desc, return_fates)


# --------------------------------------------------------------------------- #
# Ray-path capture + cross-section plot (visual inspection of refraction)
# --------------------------------------------------------------------------- #
def make_meridional_rays(n, p, div_half_deg=0.0, n_pos=7):
    """Rays that live in the y=0 plane so a 2D (z, x) cross-section shows their
    true paths (a y=0, d_y=0 ray stays in the plane through every cylinder / cap /
    entrance interaction).

    div_half_deg == 0 : a collimated fan of `n` rays across the aperture.
    div_half_deg  > 0 : `n_pos` point sources spread across the aperture, each
                        emitting a fan of directions within +/- div_half_deg, so
                        the source's divergence (and its widening in air) shows.
    """
    R = p.R_ROD
    if getattr(p, "SOURCE", "cone") == "led":
        # a few emitter points at the center, each fanning across the internal
        # angular range — shows the light-pipe / TIR mixing.  Direct-coupled LED
        # reaches ~90 deg inside; air-gap LED is capped at the critical angle.
        th_max = np.radians(85.0) if getattr(p, "LED_DIRECT_COUPLE", True) \
            else np.arcsin(1.0 / p.N_PMMA)
        n_pos = 3
        n_ang = max(3, n // n_pos)
        x0 = np.linspace(-p.EMITTER_R, p.EMITTER_R, n_pos)
        ang = np.linspace(-th_max, th_max, n_ang)
        X, A = np.meshgrid(x0, ang, indexing="ij")
        X, A = X.ravel(), A.ravel()
        o = np.column_stack([X, np.zeros(X.size), np.full(X.size, EPS)])
        d = np.column_stack([np.sin(A), np.zeros(A.size), np.cos(A)])
        return o, d

    if div_half_deg <= 0:
        x0 = np.linspace(-R*0.985, R*0.985, n)
        o = np.column_stack([x0, np.zeros(n), np.full(n, EPS)])
        d = np.tile([0.0, 0.0, 1.0], (n, 1))
        return o, d

    half = np.radians(div_half_deg)
    n_ang = max(2, n // n_pos)
    x0 = np.linspace(-R*0.9, R*0.9, n_pos)
    ang = np.linspace(-half, half, n_ang)
    X, A = np.meshgrid(x0, ang, indexing="ij")
    X, A = X.ravel(), A.ravel()
    o = np.column_stack([X, np.zeros(X.size), np.full(X.size, EPS)])
    d = np.column_stack([np.sin(A), np.zeros(A.size), np.cos(A)])
    return o, d


def trace_paths(o, d, f, fprime, p=Params(), seed=0):
    """Trace the given rays and return a list of vertex arrays (each Nx3), one per
    ray, recording every point where the ray meets a surface (plus start/target)."""
    rng = np.random.default_rng(seed)
    zt = p.z_target(f)
    paths = []
    for i in range(len(o)):
        pth = []
        _trace_ray(o[i].copy(), d[i].copy(), f, fprime, p, zt, rng, path=pth)
        paths.append(np.array(pth))
    return paths, zt


def plot_system_rays(f, fprime, p=Params(), n_rays=45, target_r=15.0,
                     screen_r=30.0, fname="system_rays.png", profile_label=""):
    """One clear cross-section of the WHOLE system: rod (z=0..L) + air gap, out to
    the screen at z = L + f(0) + TARGET_GAP.  Y axis = radius, shown to +/-screen_r.
    Rays are colored by outcome (this is the physical meaning of the color):
        green  = lands inside the target disc (|r| <= target_r) on the screen
        grey   = reaches the screen but outside the target disc
        red    = leaks out the side of the rod / lost (never reaches the screen)
    A thick green mark on the screen shows the target disc."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    o, d = make_meridional_rays(n_rays, p, div_half_deg=getattr(p, "DIV_HALF_DEG", 0.0))
    paths, zt = trace_paths(o, d, f, fprime, p)
    L = p.L
    R0, kwall, Rexit = wall_params(p)

    fig, ax = plt.subplots(figsize=(13, 6))
    # --- geometry (tapered wall R0 -> Rexit) ---
    ax.plot([0, 0], [-R0, R0], color="0.3", lw=1.5)              # LED / entrance
    ax.plot([0, L], [R0, Rexit], color="0.3", lw=1.5)            # tapered walls
    ax.plot([0, L], [-R0, -Rexit], color="0.3", lw=1.5)
    xr = np.linspace(-Rexit, Rexit, 160)
    ax.plot(L + f(np.abs(xr)), xr, color="0.3", lw=1.5)          # shaped end face
    # fill the rod interior (tapered wall + end face) lightly
    zw = np.linspace(0, L, 60)
    yface = np.linspace(Rexit, -Rexit, 90)
    poly_z = np.concatenate([zw, L + f(np.abs(yface)), zw[::-1]])
    poly_y = np.concatenate([R0 + kwall*zw, yface, -(R0 + kwall*zw[::-1])])
    ax.fill(poly_z, poly_y, color="C0", alpha=0.06, lw=0)
    ax.plot([zt, zt], [-screen_r, screen_r], color="0.45", lw=2)  # screen
    ax.plot([zt, zt], [-target_r, target_r], color="limegreen", lw=5, alpha=0.85)
    ax.annotate("LED", (0, 0), (-6, 0), ha="right", va="center", fontsize=9, color="0.3")
    ax.annotate("screen", (zt, screen_r), (zt, screen_r+1.5), ha="center", fontsize=9, color="0.3")

    # --- rays, colored by outcome ---
    n_hit = n_miss = n_lost = 0
    for pp in paths:
        if len(pp) < 2:
            n_lost += 1
            continue
        reached = abs(pp[-1, 2] - zt) < 1e-3
        xf = pp[-1, 0]
        if reached and abs(xf) <= target_r:
            c, z = "limegreen", 3; n_hit += 1
        elif reached:
            c, z = "0.55", 2; n_miss += 1
        else:
            c, z = "indianred", 2; n_lost += 1
        ax.plot(pp[:, 2], pp[:, 0], color=c, lw=0.8, alpha=0.85, zorder=z)

    ax.set_xlim(-8, zt + 4)
    ax.set_ylim(-screen_r, screen_r)
    ax.set_xlabel("z [mm]   (LED at z=0; PMMA rod 0–%g mm; then %g mm air to screen)"
                  % (L, p.TARGET_GAP))
    ax.set_ylabel("radius [mm]   (screen shown to ±%g mm)" % screen_r)
    ntot = len(paths)
    ax.set_title(("System ray paths — %s\n" % profile_label if profile_label else "")
                 + "%d rays: %d land in %g mm disc (green), %d hit screen outside (grey), "
                   "%d leak/lost (red)" % (ntot, n_hit, target_r, n_miss, n_lost))
    handles = [Line2D([0], [0], color="limegreen", lw=2, label="lands in %g mm disc" % target_r),
               Line2D([0], [0], color="0.55", lw=2, label="screen, outside disc"),
               Line2D([0], [0], color="indianred", lw=2, label="side-leak / lost")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"saved {fname}  ({ntot} rays: {n_hit} hit / {n_miss} miss / {n_lost} lost)")
    return fname


def plot_ray_paths(f, fprime, p=Params(), n_rays=25, div_half_deg=0.0,
                   seed=0, fname="ray_paths.png", profile_label=""):
    """Draw the rod cross-section (entrance, cylinder walls, shaped end face,
    target plane) and overlay a fan of meridional rays so the refraction at the
    PMMA/air interface can be inspected by eye.  Top: whole rod; bottom: zoom on
    the end-face interface."""
    import matplotlib.pyplot as plt

    o, d = make_meridional_rays(n_rays, p, div_half_deg)
    paths, zt = trace_paths(o, d, f, fprime, p, seed)
    R, L = p.R_ROD, p.L
    apex = float(f(0.0))
    cmap = plt.cm.viridis

    def draw_geometry(ax):
        ax.plot([0, 0], [-R, R], color="0.25", lw=1.6)            # entrance face
        ax.plot([0, L], [ R,  R], color="0.25", lw=1.6)           # cylinder walls
        ax.plot([0, L], [-R, -R], color="0.25", lw=1.6)
        rr = np.linspace(-R, R, 240)
        zc = L + f(np.abs(rr))
        ax.plot(zc, rr, color="0.25", lw=1.6)                     # shaped end face
        # shade the PMMA body between entrance and cap
        ax.fill_betweenx(rr, np.zeros_like(zc), zc, color="C0", alpha=0.06)
        ax.plot([zt, zt], [-1.4*R, 1.4*R], color="C3", lw=1.5, ls="--")

    def draw_rays(ax):
        for k, pp in enumerate(paths):
            if len(pp) < 2:
                continue
            ax.plot(pp[:, 2], pp[:, 0], lw=0.9, alpha=0.85,
                    color=cmap(k / max(1, len(paths) - 1)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7))

    # --- full rod ---
    draw_geometry(ax0); draw_rays(ax0)
    head = f"Ray paths — {profile_label}" if profile_label else "Ray paths through the PMMA rod"
    ax0.set_title(f"{head}\n(n_PMMA={p.N_PMMA}, div={div_half_deg:g}° in-plane)")
    ax0.set_xlim(-3, zt + 3)
    ax0.set_aspect("equal"); ax0.grid(True, alpha=0.2)
    ax0.set_ylabel("x [mm]")

    # --- zoom on the interface ---
    draw_geometry(ax1); draw_rays(ax1)
    ax1.set_title("Zoom: refraction at the PMMA → air end face")
    ax1.set_xlim(L - 8, L + apex + 18)
    ax1.set_ylim(-1.5*R, 1.5*R)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.2)
    ax1.set_xlabel("z [mm]  (optical axis)"); ax1.set_ylabel("x [mm]")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"saved {fname}")
    return fig


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
# Run bookkeeping: unique output folder + parameter manifest per run
# --------------------------------------------------------------------------- #
def new_run_dir(label, base="runs"):
    """Create and return a unique per-run directory  runs/<timestamp>__<label>/ ."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in label)
    run_dir = Path(base) / f"{stamp}__{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, stamp


def write_params_file(path, p, profile_descr, results):
    """Dump a human-readable manifest of every parameter + result for this run,
    so a saved picture can always be traced back to the exact simulation."""
    lines = ["RodLightSim run manifest",
             "=" * 40,
             "",
             "End face f(r):",
             f"    {profile_descr}",
             "",
             "Geometry / physics (Params):",
             f"    rod radius   R_ROD        = {p.R_ROD} mm   (diameter {2*p.R_ROD} mm)",
             f"    rod length   L            = {p.L} mm",
             f"    PMMA index   N_PMMA       = {p.N_PMMA}",
             f"    air index    N_AIR        = {p.N_AIR}",
             f"    target gap   TARGET_GAP   = {p.TARGET_GAP} mm  (past lens apex)",
             f"    source cone  DIV_HALF_DEG = {p.DIV_HALF_DEG} deg (half-angle inside PMMA)",
             f"    max bounces  MAX_BOUNCES  = {p.MAX_BOUNCES}",
             "",
             "Results:"]
    for k, v in results.items():
        lines.append(f"    {k:20s} = {v}")
    lines += ["", "Raw Params dataclass:", f"    {dataclasses.asdict(p)}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Standalone demo
# --------------------------------------------------------------------------- #
def main():
    import matplotlib.pyplot as plt

    p = Params()

    # ---- choose the end face here ----
    END_FACE = "flat"          # "flat" or "sphere"
    R_SPHERE = 4.0             # sphere radius [mm] (only used for "sphere")

    if END_FACE == "flat":
        f, fprime = make_flat_profile()
        label = "flat"
        descr = "flat end face,  f(r) = 0"
    else:
        f, fprime = make_sphere_profile(R_SPHERE, p.R_ROD)
        label = f"sphere_Rs{R_SPHERE:g}mm"
        descr = (f"spherical cap  Rs={R_SPHERE:g} mm,  "
                 f"f(r)=sqrt(Rs^2-r^2)-sqrt(Rs^2-R^2)  (apex bulge {f(0.0):.3f} mm)")

    n_rays, seed = 80000, 0
    run_dir, stamp = new_run_dir(label)
    print(f"run folder:     {run_dir}")
    print(f"end face:       {descr}")

    hits, frac = simulate(f, fprime, p, n_rays=n_rays, seed=seed,
                          progress=True, desc="target sim")
    rms = float(np.sqrt(np.mean(np.hypot(hits[:, 0], hits[:, 1])**2))) if len(hits) else float("nan")
    z_target = p.z_target(f)
    print(f"rays on target: {frac*100:.1f} %")
    if len(hits):
        print(f"spot RMS r:     {rms:.3f} mm")

    # ---- target-distribution figure ----
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
    ax[0].set_title(f"Irradiance on target (z = {z_target:.1f} mm)")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("y [mm]")
    ax[1].set_title("Radial irradiance profile")
    ax[1].set_xlabel("r on target [mm]"); ax[1].set_ylabel("irradiance [a.u.]")
    fig.suptitle(f"Target distribution  —  {descr}\n{stamp}", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    tgt_png = run_dir / "target_distribution.png"
    fig.savefig(tgt_png, dpi=130)
    print(f"saved {tgt_png}")

    # ---- ray-path cross-section (with the source's real divergence) ----
    ray_png = run_dir / "ray_paths.png"
    plot_ray_paths(f, fprime, p, n_rays=63, div_half_deg=p.DIV_HALF_DEG,
                   fname=str(ray_png), profile_label=descr)

    # ---- parameter manifest so this run is fully traceable ----
    write_params_file(run_dir / "params.txt", p, descr, {
        "timestamp": stamp,
        "end_face_label": label,
        "n_rays": n_rays,
        "seed": seed,
        "z_target_mm": f"{z_target:.3f}",
        "rays_on_target_pct": f"{frac*100:.2f}",
        "spot_rms_r_mm": f"{rms:.3f}",
    })
    print(f"saved {run_dir / 'params.txt'}")


if __name__ == "__main__":
    main()
