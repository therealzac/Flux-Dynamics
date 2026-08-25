"""
Torus + lattice-Green's-function engine for the FD void lattice.

WHY THIS EXISTS.  The PBD solver answers "is this configuration legal?" by
iterating until it gives up, so a false 'illegal' is indistinguishable from a
real one and the only remedy is to wait longer.  It also lives in a finite ball,
so every measurement carries a boundary echo.  This engine removes both:

  * PERIODIC.  The lattice is a 3-torus.  No surface, no reflections -- the
    "infinite lattice" the physics actually assumes.
  * SPECTRAL.  The base <111> network is translation-invariant, so its dynamical
    matrix is block-circulant and the DFT block-diagonalises it exactly into
    6x6 blocks, one per wavevector.  The Green's function is then algebra, not
    iteration.
  * REDUCED.  A shortcut enters as a Lagrange multiplier -- a Kanzaki force
    dipole -- so the unknowns collapse from 3N to the NUMBER OF SHORTCUTS.
    Twenty shortcuts on a million-node torus is a 20x20 solve.

GEOMETRY.  Bravais lattice 2Z^3 (simple cubic, side 2) with a two-site basis:
site 0 at (0,0,0), site 1 at (1,1,1).  That is exactly the shared-parity integer
set the sim uses -- all-even and all-odd triples.  Torus: cell index n in
{0..N-1}^3, so 2N^3 nodes.  Integer coords scale to world by S = 1/sqrt3, which
puts the eight <111> base bonds at unit length and the six <200> shortcut
candidates at 2/sqrt3.
"""
import numpy as np, itertools

S      = 1.0/np.sqrt(3.0)      # integer coordinate -> world
SHORT  = 2.0/np.sqrt(3.0)      # <200> rest length
DELTA  = SHORT - 1.0           # contraction a shortcut must achieve

# the eight base bonds, as (cell offset d, world direction).  Site 0 in cell n
# bonds to site 1 in cell n+d for d in {0,-1}^3; the separation is 2d+(1,1,1).
BOND_D   = [np.array(d) for d in itertools.product((0,-1), repeat=3)]
BOND_VEC = [2*d + np.array([1,1,1]) for d in BOND_D]
BOND_HAT = [v/np.linalg.norm(v) for v in BOND_VEC]          # unit, world-parallel

# the six <200> shortcut candidates, as (cell offset, same sublattice)
SC_D = [np.array(d) for d in
        [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]


class Torus:
    def __init__(self, N):
        self.N = N
        self.nnodes = 2*N**3

    # ---------------------------------------------------------------- indexing
    def cell_index(self, n):
        N = self.N; n = np.mod(n, N)
        return (n[0]*N + n[1])*N + n[2]

    def dof(self, n, site):
        """first of the three dof of (cell n, basis site)"""
        return 3*(2*self.cell_index(n) + site)

    def pos(self, n, site):
        """world position, unwrapped"""
        base = 2*np.asarray(n, float) + (np.array([1.,1,1]) if site else 0.0)
        return base*S

    # ------------------------------------------------------- real-space Phi
    def phi_real(self):
        """
        The full 6N^3 x 6N^3 dynamical matrix of unit-stiffness <111> springs.
        E = 1/2 sum_bonds (nhat . (u_i - u_j))^2 = 1/2 u^T Phi u.
        Dense -- for verification at small N only.
        """
        N = self.N; M = 6*N**3
        Phi = np.zeros((M, M))
        for n in itertools.product(range(N), repeat=3):
            n = np.array(n)
            for d, h in zip(BOND_D, BOND_HAT):
                i = self.dof(n, 0); j = self.dof(n+d, 1)
                P = np.outer(h, h)
                Phi[i:i+3, i:i+3] += P
                Phi[j:j+3, j:j+3] += P
                Phi[i:i+3, j:j+3] -= P
                Phi[j:j+3, i:i+3] -= P
        return Phi

    # ---------------------------------------------------------- k-space Phi
    def kpoints(self):
        N = self.N
        return [2*np.pi*np.array(m)/N for m in itertools.product(range(N), repeat=3)]

    def phi_k(self, k):
        """
        6x6 Hermitian block at wavevector k.  Derivation: with
        u_s(n) = uhat_s e^{i k.n}, the bond term |a - b e^{i k.d}|^2 gives
            [[ P,        -P e^{i k.d} ],
             [ -P e^{-i k.d},  P     ]]
        summed over the eight d.  Sum_d P_d = (8/3) I exactly, since the eight
        <111> directions are all sign combinations and the cross terms cancel.
        """
        Phi = np.zeros((6,6), dtype=complex)
        for d, h in zip(BOND_D, BOND_HAT):
            P = np.outer(h, h)
            ph = np.exp(1j*np.dot(k, d))
            Phi[0:3,0:3] += P
            Phi[3:6,3:6] += P
            Phi[0:3,3:6] -= P*ph
            Phi[3:6,0:3] -= P*np.conj(ph)
        return Phi

    # --------------------------------------------------------------- physics
    def strain_bond_change(self, F, kind="base"):
        """
        Worst fractional length change of a bond class under the homogeneous
        deformation F, computed EXACTLY (not linearised).  This is the C' test:
        <111> bonds are blind to a tetragonal <100> strain at first order.
        """
        vecs = BOND_VEC if kind == "base" else SC_D
        if kind != "base":
            vecs = [2*np.asarray(d, float) for d in SC_D]
        worst = 0.0
        for v in vecs:
            v = np.asarray(v, float)*S
            worst = max(worst, abs(np.linalg.norm(F @ v) - np.linalg.norm(v))
                               / np.linalg.norm(v))
        return worst

    def branch_scaling(self, direction, nk=14):
        """
        How each eigenvalue of Phi(k) scales as k -> 0 along `direction`.
        Acoustic branches go as k^2.  A branch that goes faster than k^2 is a
        SOFT branch -- a vanishing elastic constant.
        """
        u = np.asarray(direction, float); u /= np.linalg.norm(u)
        ks = np.logspace(-3.2, -1.6, nk)
        ev = np.array([np.sort(np.linalg.eigvalsh(self.phi_k(t*u)))[:6] for t in ks])
        out = []
        for b in range(6):
            y = ev[:, b]
            if y.max() < 1e-14:
                out.append(np.inf); continue
            m = y > 1e-16
            if m.sum() < 3:
                out.append(np.inf); continue
            out.append(np.polyfit(np.log(ks[m]), np.log(y[m]), 1)[0])
        return np.array(out)

    def has_soft_branch(self):
        sc = self.branch_scaling([1,0,0])
        finite = sc[np.isfinite(sc)]
        return bool((finite > 2.8).any())

    def soft_branch_report(self):
        out = []
        for d, nm in ([1,0,0],"<100>"), ([1,1,0],"<110>"), ([1,1,1],"<111>"):
            sc = self.branch_scaling(d)
            sc = sc[np.isfinite(sc)]
            out.append(f"{nm}: " + ",".join(f"{x:.2f}" for x in np.sort(sc)))
        return "  exponents  " + " | ".join(out)

    # ------------------------------------------------- the kernel, exactly
    def kernel_predicate(self, m):
        """
        Analytic condition for Phi(k) to be singular, k = 2 pi m / N:
        one component zero AND the other two equal or opposite (mod N).
        Derivation in observation_floppy_modes.md.
        """
        N = self.N
        for a in range(3):
            if m[a] % N: continue
            b, c = [i for i in range(3) if i != a]
            if (m[b]-m[c]) % N == 0 or (m[b]+m[c]) % N == 0:
                return True
        return False

    def kernel_census(self):
        """(total nullity, number of k-points where the predicate disagrees)"""
        N = self.N; tot = 0; wrong = 0
        for m in itertools.product(range(N), repeat=3):
            k = 2*np.pi*np.array(m)/N
            nz = int((np.linalg.eigvalsh(self.phi_k(k)) < 1e-10).sum())
            tot += nz
            if (nz > 0) != self.kernel_predicate(m): wrong += 1
        return tot, wrong

    def kernel_lines(self):
        """The six <110> directions carrying the kernel, as integer triples."""
        out = []
        for a in range(3):
            b, c = [i for i in range(3) if i != a]
            for s in (1, -1):
                v = np.zeros(3, int); v[b] = 1; v[c] = s
                out.append(tuple(v))
        return out

    def line_angles(self):
        """
        Pairwise angles between the six kernel lines, as a set of degrees.
        Lines, not vectors: an angle and its supplement are the same line pair.
        """
        L = [np.array(v, float) for v in self.kernel_lines()]
        A = set()
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                c = abs(L[i] @ L[j])/(np.linalg.norm(L[i])*np.linalg.norm(L[j]))
                A.add(round(np.degrees(np.arccos(min(1.0, c))), 6))
        return A

    # ==================================================================
    # THE CONSTRAINT LAYER
    #
    # Minimise ||B u||^2 subject to S u = c, where B is the base-edge Jacobian
    # (Phi = B^T B) and S the shortcut Jacobian.  Writing u = Phi^+ S^T lam + z
    # with z in ker Phi, the kernel part costs NOTHING (Phi z = 0), so
    #
    #     min energy = 0   <=>   c is reachable inside ker Phi
    #
    # That is the whole legality test, and it is a rank test on a
    # (#shortcuts) x (6N-3) matrix -- a fact, not the outcome of an iteration.
    # It is simultaneously the kernel/non-kernel DECOMPOSITION of the response:
    # the reachable part of c is carried for free by flat modes, the unreachable
    # remainder is what has to be paid for in base-edge strain.
    # ==================================================================
    def kernel_basis(self, tol=1e-9):
        """Orthonormal real basis for ker Phi.  Dimension must be 6N-3."""
        if getattr(self, "_Z", None) is None:
            w, V = np.linalg.eigh(self.phi_real())
            self._Z = V[:, w < tol]
        return self._Z

    def sc_row(self, cell, axis, site=0):
        """
        One row of S: the linearised length change of the <200> shortcut joining
        (cell, site) to (cell + e_axis, site).  Both ends are the same
        sublattice, which is what makes a shortcut an INTRA-sublattice bond.
        """
        row = np.zeros(6*self.N**3)
        w = np.zeros(3); w[axis] = 1.0            # <200> direction, already unit
        i = self.dof(np.asarray(cell), site)
        e = np.zeros(3, int); e[axis] = 1
        j = self.dof(np.asarray(cell)+e, site)
        row[i:i+3] += w
        row[j:j+3] -= w
        return -row      # sign: contraction is negative length change

    def solve(self, rows, contraction=DELTA):
        """
        rows : list of S-rows (from sc_row).   Returns a Result.
        Legal iff the contraction vector lies in S(ker Phi).
        """
        Z = self.kernel_basis()
        S = np.array(rows)                       # k x 3N
        c = np.full(len(rows), -contraction)     # each shortcut must shorten
        M = S @ Z                                # k x (6N-3): reach inside kernel
        alpha, *_ = np.linalg.lstsq(M, c, rcond=None)
        rem = c - M @ alpha                      # the part the kernel cannot carry
        feas = np.linalg.norm(rem)/np.linalg.norm(c)
        return Result(feas, M, rem, c, alpha, Z)


class Result:
    def __init__(self, feas, M, rem, c, alpha, Z):
        self.feasibility = feas          # 0 = fully carried by flat modes
        self.M, self.rem, self.c, self.alpha, self.Z = M, rem, c, alpha, Z
        self.rank = np.linalg.matrix_rank(M, tol=1e-9)
        self.legal = feas < 1e-9
        # base-edge strain is exactly the unreachable remainder: a field living
        # in ker Phi produces none at all.
        self.base_resid = feas
    def __repr__(self):
        return (f"<{'LEGAL' if self.legal else 'ILLEGAL'} feas={self.feasibility:.3e} "
                f"rank={self.rank}/{self.M.shape[0]}>")


def central_cell(t, base=(0,0,0)):
    """
    The five shortcut slots of one BCC bipyramid, on the torus.
    Polar runs along y from (cell base, site 0); its four equatorial vertices are
    site-1 nodes, and the four equatorial <200> edges split 2 (along z) + 2
    (along x) -- 'same axis' means the same one of those two families.
    """
    b = np.asarray(base)
    return {
        "polar": t.sc_row(b, 1, site=0),
        "eqA1":  t.sc_row(b + np.array([ 0, 0,-1]), 2, site=1),   # z-axis
        "eqA2":  t.sc_row(b + np.array([-1, 0,-1]), 2, site=1),   # z-axis
        "eqB1":  t.sc_row(b + np.array([-1, 0, 0]), 0, site=1),   # x-axis
        "eqB2":  t.sc_row(b + np.array([-1, 0,-1]), 0, site=1),   # x-axis
    }


# ======================================================================
# SECOND ORDER.
#
# The first-order test above asks whether the constraints can be met inside
# ker Phi.  That is NECESSARY but not SUFFICIENT, and the gap is real geometry,
# not numerics: a first-order flex need not be a second-order flex.  Moving
# along a mechanism at finite amplitude stretches the base bonds at O(a^2), and
# whether that can be absorbed is a question linear algebra on Phi cannot see.
#
# So the verdict comes from Gauss-Newton on the EXACT lengths, started from the
# first-order solution and preconditioned by the same Phi.  Converged base
# residual is the legality metric -- the same quantity the PBD solver reports,
# but reached by a method that terminates on a fact rather than a budget.
# ======================================================================
class Nonlinear:
    def __init__(self, t):
        self.t = t
        self.N = t.N
        self.ndof = 6*t.N**3
        self.X0 = self._positions()
        self.bonds = self._bond_list()

    def _positions(self):
        X = np.zeros((2*self.N**3, 3))
        for n in itertools.product(range(self.N), repeat=3):
            for s in (0, 1):
                X[self.t.dof(np.array(n), s)//3] = self.t.pos(n, s)
        return X

    def _bond_list(self):
        out = []
        for n in itertools.product(range(self.N), repeat=3):
            n = np.array(n)
            for d in BOND_D:
                out.append((self.t.dof(n,0)//3, self.t.dof(n+d,1)//3))
        return out

    def _min_image(self, v):
        L = 2*self.N*S
        return v - L*np.round(v/L)

    def _len(self, X, i, j):
        return np.linalg.norm(self._min_image(X[j]-X[i]))

    def residuals(self, X, sc):
        rb = np.array([self._len(X,i,j)-1.0 for i,j in self.bonds])
        rs = np.array([self._len(X,i,j)-1.0 for i,j in sc])
        return rb, rs

    def jac(self, X, pairs):
        J = np.zeros((len(pairs), self.ndof))
        for r,(i,j) in enumerate(pairs):
            d = self._min_image(X[j]-X[i]); d /= np.linalg.norm(d)
            J[r, 3*i:3*i+3] = -d
            J[r, 3*j:3*j+3] = +d
        return J

    def solve(self, sc, w=1e3, xtol=1e-15, maxit=400):
        """
        sc : list of (i,j) node pairs to pull to unit length.

        Levenberg-Marquardt on the stacked exact residual
            [ base lengths - 1 ;  sqrt(w) * (shortcut lengths - 1) ]
        with the analytic Jacobian.  The hand-rolled KKT this replaces used a
        GLOBAL-norm trust region, which throttled a 0.875-norm step over 384 dof
        down to ~0.02 per node -- so the shortcut constraint was never met, the
        second-order error accumulated, and the base residual climbed from
        8.9e-16 to 1.3e-2 instead of converging.  That was a solver bug, not a
        verdict.

        LEGALITY: converged base residual.  A legal configuration drives every
        base length to 1 exactly; an illegal one cannot, and the floor it hits
        is the same quantity the PBD solver reports.
        """
        from scipy.optimize import least_squares
        sw = np.sqrt(w)

        def fun(u):
            X = self.X0 + u.reshape(-1, 3)
            rb, rs = self.residuals(X, sc)
            return np.concatenate([rb, sw*rs])

        def jac(u):
            X = self.X0 + u.reshape(-1, 3)
            return np.vstack([self.jac(X, self.bonds), sw*self.jac(X, sc)])

        r = least_squares(fun, np.zeros(self.ndof), jac=jac, method="trf",
                          xtol=xtol, ftol=1e-15, gtol=1e-15, max_nfev=maxit)
        X = self.X0 + r.x.reshape(-1, 3)
        rb, rs = self.residuals(X, sc)
        base, scr = float(np.abs(rb).max()), float(np.abs(rs).max())
        return {"base": base, "sc": scr, "nfev": r.nfev, "X": X,
                "maxdisp": float(np.abs(r.x.reshape(-1,3)).max()),
                "legal": bool(base < 1e-8 and scr < 1e-8)}


def cell_pairs(t, base=(0,0,0)):
    """The five slots again, as node-index pairs for the nonlinear solver."""
    b = np.asarray(base)
    d = lambda cell, site: t.dof(np.asarray(cell), site)//3
    e = lambda a: np.eye(3, dtype=int)[a]
    return {
        "polar": (d(b,0),                 d(b+e(1),0)),
        "eqA1":  (d(b+np.array([0,0,-1]),1),  d(b+np.array([0,0,0]),1)),
        "eqA2":  (d(b+np.array([-1,0,-1]),1), d(b+np.array([-1,0,0]),1)),
        "eqB1":  (d(b+np.array([-1,0,0]),1),  d(b+np.array([0,0,0]),1)),
        "eqB2":  (d(b+np.array([-1,0,-1]),1), d(b+np.array([0,0,-1]),1)),
    }


# ======================================================================
# VARIABLE-CELL.  A contracted rod on a torus with a RIGID box is
# over-constrained: the defect has no "outside" to push into, so it pays an
# elastic energy against the box that a defect in an open ball never pays.
# Standard defect calculations relax the cell, and so must this one.
#
# Work in fractional coordinates: node position = A @ (s_i + du_i) with cell
# matrix A = L (I + eps), eps a symmetric 3x3.  Minimum image is taken in
# fractional space and then mapped through A, so periodicity is exact at every
# cell shape.  Unknowns: 3N displacements + 6 strains.
# ======================================================================
class VarCell(Nonlinear):
    def __init__(self, t):
        super().__init__(t)
        self.L = 2*t.N*S
        self.S0 = self.X0/self.L                       # fractional, in [0,1)
        self.nvar = self.ndof + 6
    _EPS = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]

    def _A(self, eps6):
        E = np.zeros((3,3))
        for v,(a,b) in zip(eps6, self._EPS):
            E[a,b] += v
            if a != b: E[b,a] += v
        return self.L*(np.eye(3) + E)

    def _lens(self, A, Sf, pairs):
        out = np.empty(len(pairs))
        for r,(i,j) in enumerate(pairs):
            d = Sf[j]-Sf[i]
            d = d - np.round(d)                        # min image, fractional
            out[r] = np.linalg.norm(A @ d)
        return out

    def solve(self, sc, w=1e3, maxit=800):
        from scipy.optimize import least_squares
        sw = np.sqrt(w); nd = self.ndof
        def split(v):
            return self.S0 + v[:nd].reshape(-1,3), v[nd:]
        def fun(v):
            Sf, e6 = split(v); A = self._A(e6)
            rb = self._lens(A, Sf, self.bonds) - 1.0
            rs = self._lens(A, Sf, sc) - 1.0
            return np.concatenate([rb, sw*rs])
        r = least_squares(fun, np.zeros(self.nvar), method="trf",
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=maxit)
        Sf, e6 = split(r.x); A = self._A(e6)
        rb = self._lens(A, Sf, self.bonds) - 1.0
        rs = self._lens(A, Sf, sc) - 1.0
        base, scr = float(np.abs(rb).max()), float(np.abs(rs).max())
        return {"base": base, "sc": scr, "nfev": r.nfev, "eps": e6,
                "strain": float(np.abs(e6).max()),
                "legal": bool(base < 1e-8 and scr < 1e-8)}


# ======================================================================
# SPARSE.  Each length residual touches exactly two nodes, so the Jacobian has
# 6 nonzeros per row.  Dense was fine at N=4 (384 dof) and is hopeless by N=8
# (3072 dof, a 4096x3072 dense factorisation per LM step).  With a sparse
# Jacobian and an LSMR trust-region solver the cost is linear in the bond count.
#
# FIXED BOX here, deliberately.  VarCell showed that a free box lets a single
# shortcut relax by Bain-transforming the WHOLE lattice at zero cost (measured:
# 22.5% box strain, base residual 1e-16, for every configuration including the
# illegal ones).  That is a true statement about C' = 0 -- a global Bain shear
# is free -- but it is not a local excitation, and an isolated shortcut must not
# be allowed to re-phase the entire vacuum.  So the box is held and N is scaled
# instead: as the defect becomes isolated, a legal configuration's residual must
# fall to zero while an illegal one's plateaus.  That divergence IS the verdict.
# ======================================================================
class Sparse(Nonlinear):
    def _pack(self, pairs):
        import scipy.sparse as sp
        rows, cols = [], []
        for r,(i,j) in enumerate(pairs):
            for a in range(3):
                rows += [r, r]; cols += [3*i+a, 3*j+a]
        return np.array(rows), np.array(cols)

    def jac_sparse(self, X, pairs, rows, cols, scale=1.0):
        import scipy.sparse as sp
        vals = np.empty(len(rows))
        for r,(i,j) in enumerate(pairs):
            d = self._min_image(X[j]-X[i]); d /= np.linalg.norm(d)
            for a in range(3):
                vals[6*r+2*a]   = -d[a]*scale
                vals[6*r+2*a+1] = +d[a]*scale
        return sp.csr_matrix((vals,(rows,cols)), shape=(len(pairs), self.ndof))

    def solve(self, sc, w=1e3, maxit=300):
        import scipy.sparse as sp
        from scipy.optimize import least_squares
        sw = np.sqrt(w)
        rb_r, rb_c = self._pack(self.bonds)
        rs_r, rs_c = self._pack(sc)
        def fun(u):
            X = self.X0 + u.reshape(-1,3)
            a, b = self.residuals(X, sc)
            return np.concatenate([a, sw*b])
        def jac(u):
            X = self.X0 + u.reshape(-1,3)
            return sp.vstack([self.jac_sparse(X, self.bonds, rb_r, rb_c),
                              self.jac_sparse(X, sc, rs_r, rs_c, sw)]).tocsr()
        r = least_squares(fun, np.zeros(self.ndof), jac=jac, method="trf",
                          tr_solver="lsmr", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          max_nfev=maxit)
        X = self.X0 + r.x.reshape(-1,3)
        a, b = self.residuals(X, sc)
        base, scr = float(np.abs(a).max()), float(np.abs(b).max())
        return {"base": base, "sc": scr, "nfev": r.nfev,
                "maxdisp": float(np.abs(r.x.reshape(-1,3)).max()),
                "legal": bool(base < 1e-8 and scr < 1e-8)}


# ======================================================================
# SPARSE VARIABLE-CELL -- the formulation that is actually right.
#
# Diagnosed on N=5, one shortcut, fixed box: the base residual is UNIFORM across
# the torus (mean 1.8e-3 in every shell from 0 to 4.5) with only 5% of it within
# 1.5 of the rod.  A contraction removes volume; a rigid box cannot absorb it,
# so the deficit is smeared over every bond.  That is an artefact of the box,
# not a property of the configuration.
#
# The other extreme is equally wrong: a FREE box at N=4 relaxed every
# configuration -- legal and illegal alike -- to base 1e-16 by Bain-transforming
# the whole lattice through 22.5% strain, which is free because C' = 0.
#
# The physical formulation is: free the box, and scale N.  One defect in 2N^3
# nodes has concentration ~ 1/N^3, so the box strain a single shortcut can call
# on falls away and the local answer converges to the isolated-defect one.  The
# discriminator is then the pair (base residual, box strain) as N grows:
#   legal   -> base -> 0 WITH box strain -> 0
#   illegal -> base plateaus at a finite floor
# ======================================================================
class SparseVarCell(VarCell):
    def solve(self, sc, w=1e3, maxit=400):
        import scipy.sparse as sp
        from scipy.optimize import least_squares
        sw = np.sqrt(w); nd = self.ndof; nv = self.nvar
        allp = list(self.bonds) + list(sc)
        # sparsity: 6 node columns per row, plus all 6 strain columns
        rows, cols = [], []
        for r,(i,j) in enumerate(allp):
            for a in range(3):
                rows += [r, r]; cols += [3*i+a, 3*j+a]
            for c in range(6):
                rows.append(r); cols.append(nd+c)
        Sp = sp.csr_matrix((np.ones(len(rows)),(rows,cols)), shape=(len(allp), nv))
        def fun(v):
            Sf = self.S0 + v[:nd].reshape(-1,3); A = self._A(v[nd:])
            rb = self._lens(A, Sf, self.bonds) - 1.0
            rs = self._lens(A, Sf, sc) - 1.0
            return np.concatenate([rb, sw*rs])
        r = least_squares(fun, np.zeros(nv), jac_sparsity=Sp, method="trf",
                          tr_solver="lsmr", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          max_nfev=maxit)
        Sf = self.S0 + r.x[:nd].reshape(-1,3); A = self._A(r.x[nd:])
        rb = self._lens(A, Sf, self.bonds) - 1.0
        rs = self._lens(A, Sf, sc) - 1.0
        base, scr = float(np.abs(rb).max()), float(np.abs(rs).max())
        return {"base": base, "sc": scr, "nfev": r.nfev,
                "boxstrain": float(np.abs(r.x[nd:]).max()),
                "legal": bool(base < 1e-8 and scr < 1e-8)}


# ======================================================================
# PROJECTED CELL -- forbid the free channel, allow the rest.
#
# Measured elastic tensor of the <111> network: C11 = C12 = C44 = 1/sqrt3,
# doubly degenerate.  C12 = C44 is the Cauchy relation (central forces); the
# EXTRA degeneracy C11 = C12 is what puts C' at zero.  Energy cost of each
# homogeneous strain direction, per unit amplitude:
#
#     hydrostatic  diag(1,1,1)      1.7320    = sqrt3     COSTS
#     shear        e_xy, e_xz, e_yz 1.1547    = 2/sqrt3   COSTS
#     Bain         traceless diag   ~1e-17               FREE
#
# A free box therefore lets a single shortcut ride the Bain channel and re-phase
# the whole lattice BCC -> FCC at no cost -- measured, and it does not dilute
# with N because a zero mode has no restoring force to set its amplitude.  A
# fixed box overcorrects: it also forbids the hydrostatic relaxation the defect
# legitimately needs, and smears a uniform residual over every bond.
#
# The projection keeps the four costly directions and removes the two free ones.
# The lattice may then absorb the defect's volume and shear locally, but cannot
# globally re-phase.  Legality becomes a local question again, which is what the
# PBD ball was answering all along.
# ======================================================================
_HYDRO = np.eye(3)/np.sqrt(3)
_SHEAR = []
for _a, _b in ((0,1), (0,2), (1,2)):
    _m = np.zeros((3,3)); _m[_a,_b] = _m[_b,_a] = 1/np.sqrt(2); _SHEAR.append(_m)
STRAIN_BASIS = [_HYDRO] + _SHEAR              # 4 allowed directions
BAIN_BASIS = [np.diag([1,-1,0])/np.sqrt(2), np.diag([1,1,-2])/np.sqrt(6)]


class ProjectedCell(VarCell):
    """Variable cell restricted to the four homogeneous strains that cost energy."""
    def __init__(self, t):
        super().__init__(t)
        self.nstrain = len(STRAIN_BASIS)
        self.nvar = self.ndof + self.nstrain

    def _A(self, coef):
        E = sum(c*B for c, B in zip(coef, STRAIN_BASIS))
        return self.L*(np.eye(3) + E)

    def bain_content(self, coef):
        """How much Bain is in this cell shape.  Must be ~0 by construction."""
        E = sum(c*B for c, B in zip(coef, STRAIN_BASIS))
        return float(max(abs(np.tensordot(E, B)) for B in BAIN_BASIS))

    def solve(self, sc, w=1e3, maxit=400):
        import scipy.sparse as sp
        from scipy.optimize import least_squares
        sw = np.sqrt(w); nd = self.ndof; nv = self.nvar; ns = self.nstrain
        allp = list(self.bonds) + list(sc)
        rows, cols = [], []
        for r, (i, j) in enumerate(allp):
            for a in range(3):
                rows += [r, r]; cols += [3*i+a, 3*j+a]
            for c in range(ns):
                rows.append(r); cols.append(nd+c)
        Sp = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(allp), nv))
        def fun(v):
            Sf = self.S0 + v[:nd].reshape(-1,3); A = self._A(v[nd:])
            return np.concatenate([self._lens(A, Sf, self.bonds) - 1.0,
                                   sw*(self._lens(A, Sf, sc) - 1.0)])
        r = least_squares(fun, np.zeros(nv), jac_sparsity=Sp, method="trf",
                          tr_solver="lsmr", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          max_nfev=maxit)
        Sf = self.S0 + r.x[:nd].reshape(-1,3); A = self._A(r.x[nd:])
        rb = self._lens(A, Sf, self.bonds) - 1.0
        rs = self._lens(A, Sf, sc) - 1.0
        base, scr = float(np.abs(rb).max()), float(np.abs(rs).max())
        return {"base": base, "sc": scr, "nfev": r.nfev,
                "strain": float(np.abs(r.x[nd:]).max()),
                "bain": self.bain_content(r.x[nd:]),
                "legal": bool(base < 1e-8 and scr < 1e-8)}
