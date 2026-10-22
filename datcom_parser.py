"""
Parser for Missile DATCOM FOR042-style CSV output (e.g. for042_SMALL.csv).

    table = parse_datcom_csv("for042_SMALL.csv")
    d = table.get(mach=3.1, alt=30000, alpha=2.0)      # one DATCOM dict, all 41 values
    # d feeds straight into datcom_loads.loads_from_datcom(datcom=d, ...)

File layout this handles
------------------------
* A block starts with a 'SIZES' line and one or more quoted variable-name lines
  (everything else is numeric rows, one row = one (MACH, ALT, ALPHA) point).
* The block repeats whenever Missile DATCOM hits its per-run alpha limit, so the
  CASE column is NOT a unique key. The real key is (MACH, ALT, ALPHA).
* Values are kept exactly as in the file. For your file the units are:
    lengths/SREF in inches/in^2, Q in psf, ALT in ft, ALPHA in DEGREES,
    derivatives per RADIAN (RAD? = 1).
  X-C.P. equals CM/CN (in LREF units, i.e. calibers) measured from the moment
  reference point, negative = aft. It is not measured from the nose.
"""
import bisect
import math
import re

_EPS = 1e-9
_LOG_KEYS = ("Q", "RE")          # interpolated in log space (they scale like M^2, p)
_AXES = ("MACH", "ALT", "ALPHA")


def _to_float(tok):
    tok = tok.strip().replace("D", "E")
    try:
        return float(tok)
    except ValueError:               # e.g. Fortran overflow '*******' or blanks
        return float("nan")


def parse_datcom_csv(path):
    """Read the file and return a DatcomTable."""
    names, pending, rows = None, [], []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if "'" in line:
                toks = re.findall(r"'([^']*)'", line)
                if toks and toks[0] != "SIZES":
                    pending.extend(toks)              # variable-name line(s)
                continue
            s = line.strip().rstrip(",")
            if not s:
                continue
            if pending:                               # first data row after a header
                if names is None:
                    names = pending
                elif pending != names:
                    raise ValueError(f"Variable names changed at line {ln}")
                pending = []
            if names is None:
                raise ValueError(f"Data before any header, line {ln}")
            vals = [_to_float(t) for t in s.split(",")]
            if len(vals) != len(names):
                raise ValueError(f"Line {ln}: {len(vals)} values but {len(names)} names")
            rows.append(dict(zip(names, vals)))
    if not rows:
        raise ValueError("No data rows found")
    return DatcomTable(rows, names)


class DatcomTable:
    def __init__(self, rows, names):
        self.names = list(names)
        self.rows = rows                                   # every row as a dict
        self.machs = sorted({r["MACH"] for r in rows})
        self.alts = sorted({r["ALT"] for r in rows})
        self.alphas = sorted({r["ALPHA"] for r in rows})
        self._grid = {self._key(r["MACH"], r["ALT"], r["ALPHA"]): r for r in rows}
        self.is_complete = len(self._grid) == (
            len(self.machs) * len(self.alts) * len(self.alphas))

    @staticmethod
    def _key(m, h, a):
        return (round(m, 6), round(h, 3), round(a, 6))

    def __repr__(self):
        return (f"DatcomTable({len(self.rows)} rows, {len(self.names)} variables, "
                f"MACH {self.machs[0]}-{self.machs[-1]} ({len(self.machs)}), "
                f"ALT {self.alts[0]}-{self.alts[-1]} ({len(self.alts)}), "
                f"ALPHA {self.alphas[0]}-{self.alphas[-1]} ({len(self.alphas)}))")

    @staticmethod
    def _bracket(grid, v, name):
        if v < grid[0] - _EPS or v > grid[-1] + _EPS:
            raise ValueError(f"{name}={v} is outside the table range "
                             f"[{grid[0]}, {grid[-1]}]")
        i = bisect.bisect_left(grid, v - _EPS)
        if abs(grid[i] - v) < _EPS:
            return i, i, 0.0
        return i - 1, i, (v - grid[i - 1]) / (grid[i] - grid[i - 1])

    def get(self, mach, alt, alpha, q=None):
        """
        Return one DATCOM dict (all variables) at (mach, alt in file units, alpha in deg).
        Exact grid point -> the row itself. Otherwise trilinear interpolation of every
        variable (Q and RE in log space). No extrapolation.

        q : optional true dynamic pressure in the file's units (psf). Pass it if you
            know it: interpolating Q is only approximate, and Q scales every force.
        """
        bi = [self._bracket(g, v, n) for g, v, n in
              ((self.machs, mach, "MACH"), (self.alts, alt, "ALT"),
               (self.alphas, alpha, "ALPHA"))]
        grids = (self.machs, self.alts, self.alphas)

        def corner(sel):
            idx = [bi[k][sel[k]] for k in range(3)]
            key = self._key(*(grids[k][idx[k]] for k in range(3)))
            if key not in self._grid:
                raise KeyError(f"Missing grid point {key}; table is not a full grid")
            return self._grid[key]

        if all(b[2] == 0.0 for b in bi):
            out = dict(corner((0, 0, 0)))
            if q is not None:
                out["Q"] = q
            return out

        corners = []
        for sel in ((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)):
            w = 1.0
            for k in range(3):
                t = bi[k][2]
                w *= t if sel[k] else (1.0 - t)
            if w > 0.0:
                corners.append((w, corner(sel)))

        out = {}
        for name in self.names:
            if name in _LOG_KEYS:
                out[name] = math.exp(sum(w * math.log(c[name]) for w, c in corners))
            else:
                out[name] = sum(w * c[name] for w, c in corners)
        out["MACH"], out["ALT"], out["ALPHA"] = mach, alt, alpha
        out["CASE"] = corners[0][1]["CASE"]        # meaningless once interpolated
        if q is not None:
            out["Q"] = q
        return out


if __name__ == "__main__":
    import sys
    t = parse_datcom_csv(sys.argv[1] if len(sys.argv) > 1 else "for042.csv")
    print(t)
    print("complete grid:", t.is_complete)
    d = t.get(mach=3.1, alt=30000, alpha=2.0)
    print({k: d[k] for k in ("MACH", "ALT", "ALPHA", "Q", "CN", "CM", "CNA", "X-C.P.")})
    d = t.get(mach=3.35, alt=32500, alpha=2.25)
    print("interpolated:", {k: round(d[k], 4) for k in ("MACH", "ALT", "ALPHA", "Q", "CN", "CM")})
