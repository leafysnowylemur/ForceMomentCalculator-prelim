"""
Shear force and bending moment in a free-flying rocket (rigid-body inertial relief).

Units: SI only (m, kg, N, N*m).
Station x is measured from the nose tip, positive aft.
Positive lateral direction = direction of the aerodynamic normal force.

Method
------
1. Total aero force N and moment about the CG come from the aero loads you supply.
2. a     = N / mass                 (CG lateral acceleration)
   alpha = M_cg / Iyy               (angular acceleration about CG)
3. Mass per length m'(s) is NOT given directly, so it is modeled as a quadratic
       m'(s) = c0 + c1*u + c2*u^2,   u = s - xcg,   0 <= s <= length
   with c0, c1, c2 solved so that it reproduces EXACTLY your mass, xcg and Iyy:
       integral m' ds           = mass
       integral u m' ds         = 0
       integral u^2 m' ds       = Iyy
4. w_inertial(s) = -m'(s) * [a + alpha*u]
5. V(x)   = integral_0^x [w_aero + w_inertial] ds
   M_b(x) = integral_0^x V(s) ds
All integrals are exact (polynomials), not numerical.
"""


def _solve3(A, b):
    """Solve 3x3 linear system by Cramer's rule (no numpy needed)."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det(A)
    out = []
    for j in range(3):
        Aj = [row[:] for row in A]
        for i in range(3):
            Aj[i][j] = b[i]
        out.append(det(Aj) / D)
    return out


def shear_and_moment(
    x,
    mass,
    xcg,
    Iyy,
    length,
    aero_point_loads=(),
    aero_dist_loads=(),
):
    """
    Parameters
    ----------
    x : float            Station (m from nose tip) to evaluate.
    mass : float         Total mass at this flight condition (kg).
    xcg : float          CG location from nose tip (m).
    Iyy : float          Pitch moment of inertia about the CG (kg*m^2).
    length : float       Overall rocket length (m). Mass is distributed over [0, length].
    aero_point_loads : list of (force_N, x_m)
    aero_dist_loads  : list of (total_force_N, x_start_m, x_end_m)  (uniform over range)

    Returns dict: V (N), M (N*m), a, alpha, N, M_cg, coeffs (c0,c1,c2),
                  min_mass_per_length (kg/m; negative means the fit is unphysical)
    """
    # ---------- aero totals ----------
    N = sum(F for F, _ in aero_point_loads) + sum(F for F, _, _ in aero_dist_loads)
    M_cg = sum(F * (xp - xcg) for F, xp in aero_point_loads) + sum(
        F * ((xa + xb) / 2.0 - xcg) for F, xa, xb in aero_dist_loads
    )
    a = N / mass
    alpha = M_cg / Iyy

    # ---------- quadratic mass-per-length fit ----------
    A_ = -xcg                # u at nose
    B_ = length - xcg        # u at tail
    def I(p):                # integral of u^p du over [A_, B_]
        return (B_ ** (p + 1) - A_ ** (p + 1)) / (p + 1)
    Mat = [[I(i + j) for i in range(3)] for j in range(3)]
    c = _solve3(Mat, [mass, 0.0, Iyy])

    # check for negative mass density (quadratic: test ends and vertex)
    def mp(u):
        return c[0] + c[1] * u + c[2] * u * u
    test_u = [A_, B_]
    if abs(c[2]) > 1e-12:
        uv = -c[1] / (2 * c[2])
        if A_ < uv < B_:
            test_u.append(uv)
    min_mp = min(mp(u) for u in test_u)

    # ---------- inertial load polynomial in u: w = -m'(u)*(a + alpha*u) ----------
    d = [0.0] * 4
    for k in range(3):
        d[k] += -c[k] * a
        d[k + 1] += -c[k] * alpha

    # integrate from nose (u = A_) to station (U = x - xcg)
    U = min(max(x, 0.0), length) - xcg
    F_in = sum(d[k] * (U ** (k + 1) - A_ ** (k + 1)) / (k + 1) for k in range(4))
    first = sum(d[k] * (U ** (k + 2) - A_ ** (k + 2)) / (k + 2) for k in range(4))
    V = F_in
    M = (x - xcg) * F_in - first        # moment of forward loads about x

    # ---------- aero loads ----------
    for F, xp in aero_point_loads:
        if x >= xp:
            V += F
            M += F * (x - xp)
    for F, xa, xb in aero_dist_loads:
        if x > xa:
            u_end = min(x, xb)
            w = F / (xb - xa)
            Fd = w * (u_end - xa)
            V += Fd
            M += w * (u_end - xa) * (x - (xa + u_end) / 2.0)

    return {"V": V, "M": M, "a": a, "alpha": alpha, "N": N, "M_cg": M_cg,
            "coeffs": tuple(c), "min_mass_per_length": min_mp}


def check_closure(mass, xcg, Iyy, length, aero_point_loads=(), aero_dist_loads=()):
    """V and M should both be ~0 at the tail."""
    r = shear_and_moment(length, mass, xcg, Iyy, length, aero_point_loads, aero_dist_loads)
    return r["V"], r["M"]


if __name__ == "__main__":
    # Worked example: 3 m, 10 kg uniform rocket (Iyy = m L^2/12 = 7.5), N = 300 N at 2.0 m
    args = dict(mass=10.0, xcg=1.5, Iyy=7.5, length=3.0, aero_point_loads=[(300.0, 2.0)])
    for xq in (1.5, 1.999999, 2.0, 3.0):
        r = shear_and_moment(xq, **args)
        print(f"x = {xq:8.6f} m   V = {r['V']:9.2f} N   M = {r['M']:9.2f} N*m")
    print("Closure at tail:", check_closure(**args))

    # Your dry numbers (hypothetical load: 2000 N at 6.9 m)
    big = dict(mass=95.1, xcg=4.249, Iyy=419.7, length=7.18,
               aero_point_loads=[(2000.0, 6.9)])
    r = shear_and_moment(6.5, **big)
    print("Your dry numbers: V=%.1f M=%.1f min m'=%.2f kg/m" % (r["V"], r["M"], r["min_mass_per_length"]))
    print("Closure:", check_closure(**big))
