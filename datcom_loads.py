"""
Shear force and bending moment from Missile DATCOM output + AoA + pitch rate.
Requires rocket_loads.py in the same folder.

Pitch-plane only (sideslip = 0). Station x is measured from the nose tip, positive aft.
Positive normal force = direction of positive CN (positive CN at positive alpha).

What it does
------------
1. Converts DATCOM coefficients to per-radian and corrects them from the DATCOM
   case's ALPHA to your actual AoA, then adds pitch-rate / alpha-dot increments:
       CN_tot = CN + CNA*(alpha - ALPHA_datcom) + CNQ*qhat + CNAD*ahat
       CM_tot = CM + CMA*(alpha - ALPHA_datcom) + CMQ*qhat + CMAD*ahat
       qhat = q*LREF/(2V),  ahat = alpha_dot*LREF/(2V)
2. N = CN_tot*Q*SREF.   Moment about the DATCOM reference point = -CM_tot*Q*SREF*LREF
   (DATCOM CM is nose-up positive; the shear code uses tail-toward-+N positive.)
3. Effective line of action of N:  x_cp_eff = x_ref + M_tail_ref / N
4. Hands N (at x_cp_eff, or split into fin + body loads) to shear_and_moment().
"""
import math
from rocket_loads import shear_and_moment


def speed_of_sound(alt_m):
    """US Standard Atmosphere 1976 speed of sound (m/s) at geometric altitude alt_m (m).
    Valid to ~84 km (your table tops out at 175,000 ft = 53 km)."""
    if alt_m < 0 or alt_m > 84000:
        raise ValueError(f"Altitude {alt_m:.0f} m outside 0-84 km standard atmosphere")
    R_E, GAMMA, R_AIR = 6356766.0, 1.4, 287.053
    H = R_E * alt_m / (R_E + alt_m) / 1000.0           # geopotential altitude, km
    # (base H km, base T K, lapse K/km)
    layers = [(0.0, 288.15, -6.5), (11.0, 216.65, 0.0), (20.0, 216.65, 1.0),
              (32.0, 228.65, 2.8), (47.0, 270.65, 0.0), (51.0, 270.65, -2.8),
              (71.0, 214.65, -2.0)]
    Hb, Tb, L = layers[0]
    for lay in layers:
        if H >= lay[0]:
            Hb, Tb, L = lay
    T = Tb + L * (H - Hb)
    return math.sqrt(GAMMA * R_AIR * T)


def _get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    if default is not None:
        return default
    raise KeyError(f"DATCOM dict is missing one of {keys}")


def loads_from_datcom(
    x,
    datcom,
    mass,
    xcg,
    Iyy,
    length,
    alpha_deg,
    pitch_rate,
    mach=None,
    alt=None,
    velocity=None,
    alpha_dot=0.0,
    len_to_m=0.0254,
    q_to_pa=1.0,
    alt_to_m=0.3048,
    x_ref_m=None,
    dyn_deriv_per_rad=None,
    fin_fraction=None,
    fin_x_range=None,
):
    """
    Parameters
    ----------
    x           : station to evaluate (m from nose tip)
    datcom      : dict of DATCOM values for ONE case, keyed by the dictionary names
                  (needs: RAD?, Q, SREF, LREF, XCG, ALPHA, CN, CM, CNA, CMA,
                   CNQ, CMQ, CNAD, CMAD, CA, X-C.P.). Values exactly as in the file;
                   get one from datcom_parser.DatcomTable.get(mach, alt, alpha).
    mass, xcg, Iyy, length : physical rocket state at this flight time (SI; xcg, length in m)
    alpha_deg   : actual angle of attack (deg)
    pitch_rate  : pitch rate q (rad/s), positive nose-up
    mach, alt   : flight Mach number and altitude (in the file's altitude units, feet by
                  default). Airspeed is Mach * speed of sound from the 1976 standard
                  atmosphere. If omitted, they default to the MACH and ALT in `datcom`.
    velocity    : optional airspeed override (m/s), e.g. for a non-standard day.
                  If given, mach and alt are not used.
    alpha_dot   : rate of change of AoA (rad/s), default 0
    len_to_m    : factor converting DATCOM length units to m (in: 0.0254, ft: 0.3048, m: 1)
                  (SREF is converted with len_to_m**2)
    q_to_pa     : factor converting DATCOM Q units to Pa (psf: 47.8803, psi: 6894.76, Pa: 1)
    alt_to_m    : factor converting altitude units to m (ft: 0.3048, default; m: 1)
    x_ref_m     : moment reference point (m from nose). Default = DATCOM XCG * len_to_m.
                  Set this if your DATCOM moment reference is not the XCG in the file.
    dyn_deriv_per_rad : units of CNQ/CMQ/CNAD/CMAD. Default None = follow the RAD? flag
                  (RAD?=1 -> per radian of nondimensional rate, consistent with your file).
    fin_fraction: optional fraction (0-1) of total N carried by the fins
    fin_x_range : (x_start_m, x_end_m) of the fin root chord. Needed with fin_fraction.
                  Fin load is spread uniformly there; the remainder goes at the body
                  point that keeps the total moment correct.

    Returns dict with V (N), M (N*m) at x plus intermediate values and warnings.
    """
    d = datcom
    warnings = []

    # ---- unit handling ----
    rad_flag = int(_get(d, "RAD?", "RAD"))
    to_per_rad = 1.0 if rad_flag == 1 else 180.0 / math.pi   # per-degree -> per-radian
    alpha_d_rad = math.radians(_get(d, "ALPHA"))     # ALPHA is in degrees even when RAD? = 1

    Q = _get(d, "Q") * q_to_pa
    S = _get(d, "SREF") * len_to_m ** 2
    Lref = _get(d, "LREF") * len_to_m
    x_ref = x_ref_m if x_ref_m is not None else _get(d, "XCG") * len_to_m

    CN, CM = _get(d, "CN"), _get(d, "CM")
    CNA, CMA = _get(d, "CNA") * to_per_rad, _get(d, "CMA") * to_per_rad
    CNQ, CMQ = _get(d, "CNQ"), _get(d, "CMQ")
    CNAD, CMAD = _get(d, "CNAD"), _get(d, "CMAD")
    if dyn_deriv_per_rad is None:
        dyn_deriv_per_rad = (rad_flag == 1)
    if not dyn_deriv_per_rad:
        CNQ, CMQ, CNAD, CMAD = (v * 180.0 / math.pi for v in (CNQ, CMQ, CNAD, CMAD))

    # ---- AoA correction and rate increments ----
    alpha = math.radians(alpha_deg)
    d_alpha = alpha - alpha_d_rad
    if abs(math.degrees(d_alpha)) > 2.0:
        warnings.append(
            f"AoA differs from DATCOM case by {math.degrees(d_alpha):.1f} deg; "
            "linear correction may be inaccurate. Rerun DATCOM at this AoA.")
    if velocity is None:
        mach = _get(d, "MACH") if mach is None else mach
        alt = _get(d, "ALT") if alt is None else alt
        a_sound = speed_of_sound(alt * alt_to_m)
        velocity = mach * a_sound
    else:
        a_sound = float("nan")
    qhat = pitch_rate * Lref / (2.0 * velocity)
    ahat = alpha_dot * Lref / (2.0 * velocity)

    CN_tot = CN + CNA * d_alpha + CNQ * qhat + CNAD * ahat
    CM_tot = CM + CMA * d_alpha + CMQ * qhat + CMAD * ahat

    # ---- forces and moment ----
    N = CN_tot * Q * S
    A = _get(d, "CA") * Q * S                       # axial force (info only)
    M_tail_ref = -CM_tot * Q * S * Lref             # tail-toward-+N positive, about x_ref

    if abs(N) < 1e-6:
        raise ValueError("Net normal force ~0: cannot place an equivalent point load. "
                         "Aero load is a pure couple.")
    x_cp_eff = x_ref + M_tail_ref / N

    # In the FOR042 file X-C.P. = CM/CN in LREF units, relative to the moment reference
    # (negative = aft of it), so CP from the nose is x_ref - X-C.P.*LREF.
    cp_outside = not (0.0 <= x_cp_eff <= length)
    if cp_outside:
        warnings.append(f"Effective CP at {x_cp_eff:.2f} m is outside the rocket (net N is small "
                        "relative to the moment, nearly a pure couple); a single point load is "
                        "not a valid model here.")
    x_cp_datcom = x_ref - _get(d, "X-C.P.", "XCP") * Lref
    cp_static = x_ref - CM / CN * Lref if CN != 0 else float("nan")

    # ---- build aero load list ----
    point_loads, dist_loads = [], []
    x_body = None
    if fin_fraction is None:
        point_loads.append((N, x_cp_eff))
    else:
        if fin_x_range is None:
            raise ValueError("fin_x_range is required with fin_fraction")
        xa, xb = fin_x_range
        N_fin = fin_fraction * N
        N_body = N - N_fin
        x_fin = 0.5 * (xa + xb)
        dist_loads.append((N_fin, xa, xb))
        if abs(N_body) > 1e-9:
            x_body = (N * x_cp_eff - N_fin * x_fin) / N_body
            point_loads.append((N_body, x_body))
            if not (0.0 <= x_body <= length):
                warnings.append(f"Body load lands at x={x_body:.2f} m, outside the rocket: "
                                "fin_fraction is inconsistent with the DATCOM moment.")

    r = shear_and_moment(x, mass, xcg, Iyy, length, point_loads, dist_loads)
    if r["min_mass_per_length"] < 0:
        warnings.append("Quadratic mass fit has negative mass per length; results unreliable.")

    r.update({
        "CN_total": CN_tot, "CM_total": CM_tot, "Q_Pa": Q, "SREF_m2": S,
        "normal_force_N": N, "axial_force_N": A,
        "x_cp_effective_m": x_cp_eff,
        "cp_outside_rocket": cp_outside,
        "x_cp_datcom_m": x_cp_datcom,           # compare to x_cp_effective_m
        "x_cp_static_from_CM_m": cp_static,
        "x_body_load_m": x_body,
        "velocity_ms": velocity, "speed_of_sound_ms": a_sound,
        "warnings": warnings,
    })
    return r


def sweep_loads(table, mach, alt, mass, xcg, Iyy, length, alphas_deg, pitch_rates,
                n_x=40, q=None, **kw):
    """
    Shear V and bending moment M on a grid of (station x, alpha, pitch rate) at one
    Mach/altitude. For each alpha the DATCOM row at that alpha is pulled from `table`,
    so no linear alpha correction is used. Points where the single-point-load model is
    invalid (effective CP outside the rocket) are NaN.
    kw is passed to loads_from_datcom (len_to_m, q_to_pa, alt_to_m, fin_fraction, ...).
    Returns X, A, R, V, M arrays of shape (n_x, n_alpha, n_rate).
    """
    import numpy as np
    xs = np.linspace(0.0, length, n_x)
    alphas = np.asarray(alphas_deg, dtype=float)
    rates = np.asarray(pitch_rates, dtype=float)
    X, A, R = np.meshgrid(xs, alphas, rates, indexing="ij")
    V = np.full(X.shape, np.nan)
    M = np.full(X.shape, np.nan)
    for j, a in enumerate(alphas):
        d = table.get(mach, alt, a, q=q)
        for k, qr in enumerate(rates):
            if a == 0.0 and qr == 0.0:               # no aero load at all
                V[:, j, k] = 0.0
                M[:, j, k] = 0.0
                continue
            for i, x in enumerate(xs):
                r = loads_from_datcom(x=x, datcom=d, mass=mass, xcg=xcg, Iyy=Iyy,
                                      length=length, alpha_deg=a, pitch_rate=qr,
                                      mach=mach, alt=alt, **kw)
                if r["cp_outside_rocket"]:
                    break                            # rest of this column stays NaN
                V[i, j, k] = r["V"]
                M[i, j, k] = r["M"]
    return X, A, R, V, M


def sweep_shear(*args, **kw):
    """Same as sweep_loads but returns only X, A, R, V (kept for older scripts)."""
    X, A, R, V, _ = sweep_loads(*args, **kw)
    return X, A, R, V


def plot_heatmaps(X, A, R, Z, label="Shear force V (N)", title="", max_panels=6,
                  save_path=None):
    """
    Small multiples: one heatmap per pitch rate, x = station, y = alpha, color = Z
    (shear or moment). All panels share one symmetric color scale.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    xs, alphas, rates = X[:, 0, 0], A[0, :, 0], R[0, 0, :]
    n = min(max_panels, len(rates))
    idx = sorted(set(np.round(np.linspace(0, len(rates) - 1, n)).astype(int)))
    ncols = min(3, len(idx))
    nrows = math.ceil(len(idx) / ncols)
    zmax = np.nanmax(np.abs(Z))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows),
                             sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)
    pcm = None
    for ax, k in zip(axes.ravel(), idx):
        pcm = ax.pcolormesh(xs, alphas, np.ma.masked_invalid(Z[:, :, k]).T,
                            cmap="coolwarm", vmin=-zmax, vmax=zmax, shading="nearest")
        ax.set_title(f"pitch rate = {rates[k]:+.2f} rad/s", fontsize=10)
        ax.set_facecolor("0.85")                       # grey = invalid / omitted
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Station x from nose (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Angle of attack (deg)")
    fig.colorbar(pcm, ax=axes, shrink=0.85, label=label)
    fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_envelope(X, A, R, Z, label="Shear force V (N)", short="V", unit="N",
                  title="", save_path=None):
    """
    Left : peak |Z| anywhere along the rocket for every (pitch rate, alpha) combination.
    Right: the Z(x) diagram for the worst-case combination.
    Returns (fig, (alpha, pitch_rate, x_of_peak, Z_at_peak)).
    """
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    xs, alphas, rates = X[:, 0, 0], A[0, :, 0], R[0, 0, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(np.abs(Z), axis=0)                    # (n_alpha, n_rate)
    ja, kr = np.unravel_index(np.nanargmax(peak), peak.shape)
    ix = int(np.nanargmax(np.abs(Z[:, ja, kr])))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    pcm = a1.pcolormesh(rates, alphas, np.ma.masked_invalid(peak), cmap="magma_r",
                        shading="nearest")
    if np.isfinite(peak).sum() > 4:
        cs = a1.contour(rates, alphas, np.ma.masked_invalid(peak), colors="k",
                        linewidths=0.6, levels=8)
        a1.clabel(cs, fmt="%.0f", fontsize=8)
    a1.plot(rates[kr], alphas[ja], "c*", ms=16, mec="k")
    a1.set_facecolor("0.85")
    a1.set_xlabel("Pitch rate (rad/s)")
    a1.set_ylabel("Angle of attack (deg)")
    a1.set_title(f"Peak |{short}| along the rocket")
    fig.colorbar(pcm, ax=a1, label=f"Peak |{short}| ({unit})")

    z = Z[:, ja, kr]
    a2.fill_between(xs, 0, z, where=np.isfinite(z), color="tab:red", alpha=0.25)
    a2.plot(xs, z, color="tab:red", lw=2)
    a2.axhline(0, color="k", lw=0.8)
    a2.plot(xs[ix], z[ix], "k*", ms=14)
    a2.annotate(f"{z[ix]:,.0f} {unit} at x = {xs[ix]:.2f} m", (xs[ix], z[ix]),
                textcoords="offset points", xytext=(10, 10 if z[ix] >= 0 else -18))
    a2.set_xlabel("Station x from nose (m)")
    a2.set_ylabel(label)
    a2.set_title(f"Worst case: alpha = {alphas[ja]:.1f} deg, pitch rate = {rates[kr]:+.2f} rad/s")
    a2.margins(y=0.18)
    a2.grid(alpha=0.3)
    fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, (alphas[ja], rates[kr], xs[ix], z[ix])


def plot_loads_vs_alpha(X, A, R, V, M, k=0, line_alphas=None, title="", save_path=None):
    """
    One figure, shear on the left and bending moment on the right (pitch rate fixed at
    the k-th entry of the sweep's rate axis).
      top row    : heatmap, x = station, y = alpha, color = V or M
      bottom row : V(x) / M(x) curves, one line per alpha
    Returns (fig, peaks) where peaks maps "V"/"M" to (value, station_m, alpha_deg).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    xs, alphas, rate = X[:, 0, 0], A[0, :, 0], R[0, 0, k]
    if line_alphas is None:
        line_alphas = alphas[::max(1, len(alphas) // 5)][1:] if len(alphas) > 2 else alphas
    cmap = plt.get_cmap("plasma")
    lo, hi = float(np.min(line_alphas)), float(np.max(line_alphas))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    peaks = {}
    for col, (Z, name, short, unit) in enumerate(
            ((V, "Shear force", "V", "N"), (M, "Bending moment", "M", "N*m"))):
        z = Z[:, :, k]
        zmax = np.nanmax(np.abs(z))
        ax = axes[0, col]
        pcm = ax.pcolormesh(xs, alphas, np.ma.masked_invalid(z).T, cmap="coolwarm",
                            vmin=-zmax, vmax=zmax, shading="nearest")
        ax.set_facecolor("0.85")
        ax.set_xlabel("Station x from nose (m)")
        ax.set_ylabel("Angle of attack (deg)")
        ax.set_title(f"{name} {short} ({unit})")
        fig.colorbar(pcm, ax=ax, label=f"{short} ({unit})")

        ax = axes[1, col]
        for a in line_alphas:
            j = int(np.argmin(np.abs(alphas - a)))
            ax.plot(xs, z[:, j], color=cmap((a - lo) / (hi - lo + 1e-12) * 0.85),
                    lw=2, label=f"{alphas[j]:.1f}")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Station x from nose (m)")
        ax.set_ylabel(f"{short} ({unit})")
        ax.set_title(f"{name} along the rocket, by angle of attack")
        ax.legend(title="alpha (deg)", fontsize=8, ncol=2)

        i, j = np.unravel_index(np.nanargmax(np.abs(z)), z.shape)
        ax.plot(xs[i], z[i, j], "k*", ms=14)
        ax.annotate(f"{z[i, j]:,.0f} {unit}\nx = {xs[i]:.2f} m, alpha = {alphas[j]:.1f}",
                    (xs[i], z[i, j]), textcoords="offset points",
                    xytext=(12, 8 if z[i, j] >= 0 else -34), fontsize=9)
        ax.margins(y=0.2)
        peaks[short] = (z[i, j], xs[i], alphas[j])
    fig.suptitle(f"{title}   (pitch rate = {rate:+.2f} rad/s)")
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, peaks


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from datcom_parser import parse_datcom_csv

    table = parse_datcom_csv("for042_SMALL.csv")

    # ---- flight condition ----
    MACH, ALT_FT = 4.0, 30000
    PITCH_RATE = 0.0                              # rad/s, held constant (+ = nose up)

    # ---- rocket state: convert your inch/lb numbers to SI ----
    LB, IN = 0.45359237, 0.0254
    mass   = 209.0 * LB                      # kg      (209 lbm)
    xcg    = 167.0 * IN                      # m       (167 in)
    Iyy    = 1434057.908 * LB * IN**2        # kg*m^2  (lbm*in^2 -> kg*m^2)
    length = 287.0 * IN                      # m

    # ---- sweep: station x angle of attack ----
    alphas = np.linspace(0.0, 10.0, 21)          # deg (0.5 deg = native DATCOM spacing)

    X, A, R, V, M = sweep_loads(table, MACH, ALT_FT, mass, xcg, Iyy, length,
                                alphas, [PITCH_RATE], n_x=100,
                                len_to_m=IN, q_to_pa=47.8803)   # file units: in, psf

    _, pk = plot_loads_vs_alpha(X, A, R, V, M, line_alphas=[2, 4, 6, 8, 10],
                                title=f"Mach {MACH}, {ALT_FT} ft",
                                save_path="loads_vs_alpha.png")
    print("Peak shear : %.0f N at x=%.2f m (alpha=%.1f deg)" % pk["V"])
    print("Peak moment: %.0f N*m (%.0f lbf*in) at x=%.2f m (alpha=%.1f deg)"
          % (pk["M"][0], abs(pk["M"][0]) * 8.8507, pk["M"][1], pk["M"][2]))
    plt.show()