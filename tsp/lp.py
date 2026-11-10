"""A small dense two-phase simplex.

The branch-and-cut solver solves the DFJ relaxation of a TSP (tens of rows,
hundreds of columns) thousands of times, so this kernel is kept intentionally
minimal: standard form, dense tableau, Dantzig pricing with a Bland's-rule
escape hatch for degenerate vertices.

numpy is used when installed (a pivot becomes a few vector operations); the
pure-Python path produces the same results and keeps the core dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # pragma: no cover - numpy is optional
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

_TOL = 1e-9
_FEAS_TOL = 1e-7
_STALL_LIMIT = 100


@dataclass
class LPResult:
    status: str  # "optimal" | "infeasible" | "unbounded"
    x: list[float] = field(default_factory=list)
    value: float = 0.0
    iterations: int = 0


def minimize(
    objective: list[float],
    constraints: list[tuple[list[float], float, str]],
    n_vars: int,
    *,
    tol: float = _TOL,
    max_iterations: int = 100_000,
) -> LPResult:
    """Minimise ``objective @ x`` subject to ``constraints`` with ``x >= 0``.

    ``constraints`` holds ``(coefficients, rhs, op)`` triples with ``op`` in
    ``{"<=", ">=", "=="}``.  Upper bounds on variables are not part of the
    standard form; ``BranchAndCutSolver`` adds them lazily as explicit rows.
    """
    normalized: list[tuple[list[float], float, str]] = []
    for coefficients, rhs, op in constraints:
        if len(coefficients) != n_vars:
            raise ValueError("constraint width does not match the number of variables")
        if op not in ("<=", ">=", "=="):
            raise ValueError(f"unsupported relation {op!r}")
        row = [float(value) for value in coefficients]
        rhs = float(rhs)
        if rhs < 0:
            row = [-value for value in row]
            rhs = -rhs
            op = {"<=": ">=", ">=": "<=", "==": "=="}[op]
        normalized.append((row, rhs, op))

    cost = [float(value) for value in objective]
    if len(normalized) == 0:
        if any(value < -tol for value in cost):
            return LPResult("unbounded")
        return LPResult("optimal", [0.0] * n_vars, 0.0)

    extra = sum(2 if op == ">=" else 1 for _, _, op in normalized)
    n_columns = n_vars + extra
    tableau: list[list[float]] = []
    b: list[float] = []
    basis: list[int] = []
    artificials: set[int] = set()
    column = n_vars
    for row, rhs, op in normalized:
        full = row + [0.0] * extra
        if op == "<=":
            full[column] = 1.0
            basis.append(column)
            column += 1
        elif op == ">=":
            full[column] = -1.0  # surplus
            column += 1
            full[column] = 1.0  # artificial
            basis.append(column)
            artificials.add(column)
            column += 1
        else:
            full[column] = 1.0  # artificial
            basis.append(column)
            artificials.add(column)
            column += 1
        tableau.append(full)
        b.append(rhs)

    A, bv = _as_matrix(tableau), _as_vector(b)

    phase_one_cost = [0.0] * n_columns
    for index in artificials:
        phase_one_cost[index] = 1.0
    rc = _reduced_costs(phase_one_cost, A, basis, n_columns)
    status, iterations = _run_simplex(A, bv, rc, basis, tol, max_iterations)
    if status == "unbounded":  # phase 1 is bounded below by zero
        return LPResult("infeasible", iterations=iterations)
    if _objective_value(phase_one_cost, bv, basis) > _FEAS_TOL:
        return LPResult("infeasible", iterations=iterations)

    # Drive artificials out of the basis.  A row that cannot be pivoted is a
    # redundant constraint, hence dropped.
    dropped: list[int] = []
    for row_index in range(len(basis)):
        if basis[row_index] not in artificials:
            continue
        replacement = -1
        for j in range(n_columns):
            if j in artificials or j in basis:
                continue
            if abs(_entry(A, row_index, j)) > _FEAS_TOL:
                replacement = j
                break
        if replacement < 0:
            dropped.append(row_index)
        else:
            _pivot(A, bv, rc, basis, row_index, replacement)

    kept_columns = [j for j in range(n_columns) if j not in artificials]
    if dropped:
        A = _delete_rows(A, dropped)
        bv = _delete_values(bv, dropped)
        removed_rows = set(dropped)
        basis = [value for index, value in enumerate(basis) if index not in removed_rows]
    A = _delete_columns(A, sorted(artificials))

    if not basis:
        if any(cost[j] < -tol for j in kept_columns if j < n_vars):
            return LPResult("unbounded", iterations=iterations)
        return LPResult("optimal", [0.0] * n_vars, 0.0, iterations)

    position = {old: new for new, old in enumerate(kept_columns)}
    basis = [position[j] for j in basis]
    kept_cost = [cost[j] if j < n_vars else 0.0 for j in kept_columns]

    rc = _reduced_costs(kept_cost, A, basis, len(kept_columns))
    status, extra_iterations = _run_simplex(A, bv, rc, basis, tol, max_iterations)
    iterations += extra_iterations
    if status == "unbounded":
        return LPResult("unbounded", iterations=iterations)

    x = [0.0] * n_vars
    for row_index, column_index in enumerate(basis):
        original = kept_columns[column_index]
        if original < n_vars:
            x[original] = max(0.0, float(bv[row_index]))
    return LPResult("optimal", x, _objective_value(kept_cost, bv, basis), iterations)


def _as_matrix(rows):
    return _np.array(rows, dtype=float) if _np is not None else rows


def _as_vector(values):
    return _np.array(values, dtype=float) if _np is not None else values


def _reduced_costs(cost, A, basis, n_columns):
    if _np is not None:
        cost_array = cost if isinstance(cost, _np.ndarray) else _np.array(cost, dtype=float)
        return cost_array - cost_array[basis] @ A
    rc = list(cost)
    for row_index, column_index in enumerate(basis):
        factor = cost[column_index]
        if factor == 0.0:
            continue
        row = A[row_index]
        for j in range(n_columns):
            rc[j] -= factor * row[j]
    return rc


def _objective_value(cost, b, basis) -> float:
    total = 0.0
    for row_index, column_index in enumerate(basis):
        total += cost[column_index] * float(b[row_index])
    return total


def _entry(A, row_index, column_index) -> float:
    return float(A[row_index][column_index]) if _np is None else float(A[row_index, column_index])


def _pivot(A, b, rc, basis, leave: int, enter: int) -> None:
    """Gauss-Jordan pivot on (leave, enter), updating costs and the basis."""
    if _np is not None:
        pivot_value = A[leave, enter]
        A[leave] = A[leave] / pivot_value
        b[leave] = b[leave] / pivot_value
        factors = A[:, enter].copy()
        factors[leave] = 0.0
        A -= factors[:, None] * A[leave]
        b -= factors * b[leave]
        enter_cost = float(rc[enter])
        rc -= enter_cost * A[leave]
        basis[leave] = enter
        return

    row = A[leave]
    inverse = 1.0 / row[enter]
    for j in range(len(row)):
        row[j] *= inverse
    b[leave] *= inverse
    for i in range(len(A)):
        if i == leave:
            continue
        factor = A[i][enter]
        if factor == 0.0:
            continue
        target = A[i]
        for j in range(len(row)):
            target[j] -= factor * row[j]
        b[i] -= factor * b[leave]
    enter_cost = rc[enter]
    for j in range(len(row)):
        rc[j] -= enter_cost * row[j]
    basis[leave] = enter


def _delete_rows(A, rows):
    if _np is not None:
        return _np.delete(A, rows, axis=0)
    removed = set(rows)
    return [row for index, row in enumerate(A) if index not in removed]


def _delete_columns(A, columns):
    if not columns:
        return A
    if _np is not None:
        return _np.delete(A, columns, axis=1)
    removed = set(columns)
    return [[value for index, value in enumerate(row) if index not in removed] for row in A]


def _delete_values(values, indices):
    if _np is not None:
        return _np.delete(values, indices)
    removed = set(indices)
    return [value for index, value in enumerate(values) if index not in removed]


def _run_simplex(A, b, rc, basis, tol, max_iterations) -> tuple[str, int]:
    n_rows = len(basis)
    n_columns = len(rc)
    iterations = 0
    use_bland = False
    stall = 0
    while iterations < max_iterations:
        iterations += 1
        enter = _pricing(rc, n_columns, tol, use_bland)
        if enter < 0:
            return "optimal", iterations

        best_ratio = float("inf")
        leave = -1
        for i in range(n_rows):
            coefficient = _entry(A, i, enter)
            if coefficient <= tol:
                continue
            ratio = float(b[i]) / coefficient
            closer = ratio < best_ratio - 1e-12
            tied_but_smaller = abs(ratio - best_ratio) <= 1e-12 and (leave < 0 or basis[i] < basis[leave])
            if closer or tied_but_smaller:
                best_ratio = ratio
                leave = i
        if leave < 0:
            return "unbounded", iterations

        _pivot(A, b, rc, basis, leave, enter)

        # Degenerate pivots can cycle; fall back to Bland's rule for good.
        if best_ratio <= 1e-12:
            stall += 1
            if stall >= _STALL_LIMIT:
                use_bland = True
        else:
            stall = 0
    return "optimal", iterations


def _pricing(rc, n_columns, tol, use_bland) -> int:
    if use_bland:
        for j in range(n_columns):
            if rc[j] < -tol:
                return j
        return -1
    if _np is not None:
        best = int(rc.argmin())
        return best if rc[best] < -tol else -1
    best = 0
    for j in range(1, n_columns):
        if rc[j] < rc[best]:
            best = j
    return best if rc[best] < -tol else -1


class LPModel:
    """An LP that stays warm across solves.

    ``minimize`` builds a fresh tableau every call: phase 1, artificial
    elimination, then phase 2.  A branch-and-cut node instead perturbs the same
    LP over and over (one cut, then another), so it can keep the tableau, the
    basis and :math:`B^{-1}` between solves.  Adding a cut then costs one
    Gauss-Jordan sweep plus a handful of dual-simplex pivots, and adding a
    column (pricing) is a matrix-vector product against :math:`B^{-1}`.

    The row/column bookkeeping follows the standard form used by
    :func:`minimize`: one slack for ``<=`` rows, a surplus plus an artificial
    for ``>=`` rows, an artificial for ``==`` rows.  Artificials stay in the
    tableau as permanently blocked columns instead of being deleted, which
    keeps column indices stable while rows are appended.

    Two conventions keep callers away from the standard form:

    * row numbers are the ones the model reports through :attr:`n_rows` and
      accept back in ``add_columns``;
    * a row is interned with a sign (``>=`` rows are negated so that their new
      slack has coefficient ``+1``), and :attr:`duals` is scaled by that sign,
      so ``reduced_cost(e) == cost(e) - sum_r duals[r] * coefficient(r, e)``
      holds for the coefficients the caller supplied.

    :meth:`solve` returns ``x`` indexed by *column*, i.e. a list of length
    :attr:`n_columns`: entries ``0 .. n_vars-1`` are the original variables and
    the indices returned by :meth:`add_columns` address the generated ones.
    """

    def __init__(
        self,
        n_vars: int,
        objective: list[float],
        *,
        tol: float = _TOL,
        max_iterations: int = 100_000,
    ) -> None:
        if len(objective) != n_vars:
            raise ValueError("objective width does not match the number of variables")
        self.n_vars = int(n_vars)
        self.tol = float(tol)
        self.max_iterations = int(max_iterations)
        self._cost = [float(value) for value in objective]
        self._objective = [float(value) for value in objective]
        self._history: list[tuple[dict[int, float], float, str]] = []
        self._added: list[tuple[float, dict[int, float]]] = []
        self._built = False
        self._T = None
        self._b = None
        self._rc = None
        self._basis: list[int] = []
        self._binv = None
        self._blocked: list[bool] = []
        self._signs: list[float] = []
        self._feasible = True
        self.build_rows = 0
        self.total_iterations = 0
        self.status = "new"
        self.duals: list[float] = []

    # ------------------------------------------------------------ public API

    @property
    def n_rows(self) -> int:
        return len(self._basis) if self._built else len(self._history)

    @property
    def n_columns(self) -> int:
        if not self._built:
            return self.n_vars
        return len(self._rc)

    def add_row(self, coefficients, rhs: float, relation: str = ">=") -> None:
        """Append a constraint whose coefficients are indexed by column.

        Before the first :meth:`solve` the only columns are the original
        variables, so the indices are variable numbers; afterwards a coefficient
        may also address a column returned by :meth:`add_columns`.
        """
        if relation not in ("<=", ">=", "=="):
            raise ValueError(f"unsupported relation {relation!r}")
        coefs = _sparse(coefficients)
        limit = self.n_columns
        for index in coefs:
            if not 0 <= index < limit:
                raise ValueError(f"row refers to column {index} of {limit}")
        value = float(rhs)
        self._history.append((coefs, value, relation))
        if self.status == "infeasible":
            # Constraints only ever pile up: the model stays infeasible.
            return
        if not self._built:
            return
        if relation == "==" or self.status == "unbounded":
            # Both need a fresh phase 1.  A rebuild replays the history through
            # the standard-form layout, which moves the artificial columns and
            # would silently invalidate the indices priced-in columns were
            # handed out under, so refuse instead of guessing.
            if self._added:
                raise RuntimeError("cannot rebuild an LP that already has generated columns")
            self._built = False  # a new row may restore boundedness
            return
        self._append_row(coefs, value, relation)


    def add_columns(self, columns: list[tuple[float, dict[int, float]]]) -> list[int]:
        """Append variables given by ``(cost, {row: coefficient})`` pairs.

        ``row`` is a row number as reported by :attr:`n_rows`; the reduction to
        standard form is applied internally, and the returned list holds the
        column index of every new variable (usable as an index into the ``x``
        of :meth:`solve`).  Appending a column costs one matrix-vector product
        against :math:`B^{-1}` instead of a fresh phase 1, which is what makes
        pricing affordable.
        """
        if not columns:
            return []
        if not self._built:
            raise RuntimeError("build the model with solve() before adding columns")
        rows = len(self._basis)
        signs = self._signs
        dense: list[list[float]] = []
        costs: list[float] = []
        for cost, entries in columns:
            vector = [0.0] * rows
            for row_index, value in entries.items():
                if not 0 <= row_index < rows:
                    raise IndexError(f"column refers to row {row_index} of {rows}")
                vector[row_index] += signs[row_index] * float(value)
            dense.append(vector)
            costs.append(float(cost))

        base = len(self._rc)
        if _np is not None:
            if rows:
                matrix = _np.array(dense, dtype=float).T
            else:
                matrix = _np.zeros((0, len(dense)))
            y = _np.asarray(self._cost, dtype=float)[self._basis] @ self._binv
            rc_new = _np.array(costs, dtype=float) - y @ matrix
            self._T = _np.hstack([self._T, self._binv @ matrix])
            self._rc = _np.concatenate([self._rc, rc_new])
        else:
            y = self._dual_vector()
            appended: list[tuple[list[float], float]] = []
            for vector, cost in zip(dense, costs):
                column = [0.0] * rows
                for i in range(rows):
                    total = 0.0
                    row_i = self._binv[i]
                    for k in range(rows):
                        if vector[k]:
                            total += row_i[k] * vector[k]
                    column[i] = total
                appended.append((column, cost - sum(y[i] * vector[i] for i in range(rows))))
            for i in range(rows):
                self._T[i].extend(column[i] for column, _ in appended)
            self._rc.extend(reduced for _, reduced in appended)

        self._cost.extend(costs)
        self._blocked.extend([False] * len(costs))
        self._added.extend((cost, dict(entries)) for cost, entries in columns)
        return [base + offset for offset in range(len(costs))]

    def solve(self) -> LPResult:
        if self.status == "infeasible":
            return LPResult("infeasible")
        if not self._built:
            self._build()
            if self.status in ("infeasible", "unbounded"):
                return LPResult(self.status)
        iterations = 0
        if not self._feasible:
            status, count = self._dual_simplex(self.max_iterations)
            iterations += count
            if status == "infeasible":
                self.total_iterations += iterations
                self.status = "infeasible"
                return LPResult("infeasible", iterations=iterations)
            self._feasible = True
        status, count = self._primal_simplex(self.max_iterations)
        iterations += count
        self.total_iterations += iterations
        if status == "unbounded":
            self.status = "unbounded"
            return LPResult("unbounded", iterations=iterations)
        self.status = "optimal"
        x = [0.0] * self.n_columns
        total = 0.0
        for row_index, column_index in enumerate(self._basis):
            value = float(self._b[row_index])
            x[column_index] = max(0.0, value)
            total += self._cost[column_index] * value
        self.duals = [
            sign * value for sign, value in zip(self._signs, self._dual_vector())
        ]
        return LPResult("optimal", x, total, iterations)

    # --------------------------------------------------------------- building

    def _build(self) -> None:
        rows: list[list[float]] = []
        rhs: list[float] = []
        ops: list[str] = []
        signs: list[float] = []
        for coefs, value, relation in self._history:
            row = [0.0] * self.n_vars
            for index, entry in coefs.items():
                if not 0 <= index < self.n_vars:
                    raise ValueError("constraint width does not match the number of variables")
                row[index] = entry
            sign = 1.0
            if value < 0:
                row = [-entry for entry in row]
                value = -value
                relation = {"<=": ">=", ">=": "<=", "==": "=="}[relation]
                sign = -1.0
            rows.append(row)
            rhs.append(value)
            ops.append(relation)
            signs.append(sign)

        n_rows = len(rows)
        self.build_rows = n_rows
        self._signs = signs
        artificials = sum(1 for op in ops if op in (">=", "=="))
        n_columns = self.n_vars + n_rows + artificials
        tableau = [[0.0] * n_columns for _ in range(n_rows)]
        basis: list[int] = []
        cost = [0.0] * n_columns
        blocked = [False] * n_columns
        artificial_columns: list[int] = []
        next_artificial = self.n_vars + n_rows
        for index, (row, op) in enumerate(zip(rows, ops)):
            tableau[index][: self.n_vars] = row
            slack = self.n_vars + index
            if op == "<=":
                tableau[index][slack] = 1.0
                basis.append(slack)
            elif op == ">=":
                tableau[index][slack] = -1.0
                tableau[index][next_artificial] = 1.0
                cost[next_artificial] = 1.0
                basis.append(next_artificial)
                artificial_columns.append(next_artificial)
                next_artificial += 1
            else:
                tableau[index][next_artificial] = 1.0
                cost[next_artificial] = 1.0
                basis.append(next_artificial)
                artificial_columns.append(next_artificial)
                next_artificial += 1

        phase_two = [
            self._objective[column] if column < self.n_vars else 0.0
            for column in range(n_columns)
        ]

        self._built = True
        self._basis = basis
        self._cost = cost
        self._blocked = blocked
        if n_rows == 0:
            self._T = _empty_matrix(n_columns)
            self._b = _as_vector([])
            self._binv = _identity(0)
            self._cost = list(phase_two)
            self._rc = _as_vector(list(phase_two))
            self._feasible = True
            if any(value < -self.tol for value in phase_two):
                self.status = "unbounded"
            return

        self._T = _as_matrix(tableau)
        self._b = _as_vector(rhs)
        self._binv = _identity(n_rows)
        self._rc = _reduced_costs(cost, self._T, basis, n_columns)

        status, iterations = self._primal_simplex(self.max_iterations)
        self.total_iterations += iterations
        if status == "unbounded":
            # Phase 1 is bounded below by zero, so this means "infeasible".
            self.status = "infeasible"
            self._feasible = False
            return
        phase_one = _objective_value(cost, self._b, basis)
        if phase_one > _FEAS_TOL:
            self.status = "infeasible"
            self._feasible = False
            return

        artificial = set(artificial_columns)
        for row_index in range(len(basis)):
            if basis[row_index] not in artificial:
                continue
            for column in range(n_columns):
                if column in artificial or abs(_entry(self._T, row_index, column)) <= _FEAS_TOL:
                    continue
                self._pivot(row_index, column)
                break

        for column in artificial_columns:
            blocked[column] = True
        for column in range(n_columns):
            cost[column] = phase_two[column]
        self._rc = _reduced_costs(cost, self._T, self._basis, n_columns)
        if _np is not None and artificial_columns:
            # Artificials never enter again, so pin their reduced cost at +inf:
            # pricing then becomes a plain argmin, and pivots keep them at inf
            # (inf - finite == inf).
            self._rc[artificial_columns] = _np.inf
        self._feasible = True

    def _append_row(self, coefs: dict[int, float], rhs: float, relation: str) -> None:
        n_rows = len(self._basis)
        width = self.n_columns
        row = [0.0] * (width + 1)
        for index, value in coefs.items():
            if not 0 <= index < width:
                raise ValueError(f"row refers to column {index} of {width}")
            row[index] = value
        sign = 1.0
        if relation == ">=":
            row = [-value for value in row]
            rhs = -rhs
            sign = -1.0
        row[width] = 1.0  # the new slack

        # The basis becomes [[B, 0], [basic, 1]], so B^-1 gains a zero column
        # and the row [-basic B^-1, 1].  ``basic`` has to be read off before the
        # row is reduced below.
        basic = [row[column] for column in self._basis]

        for row_index in range(n_rows):
            factor = basic[row_index]
            if abs(factor) <= self.tol:
                continue
            tableau_row = _row(self._T, row_index)
            for column in range(width):
                row[column] -= factor * float(tableau_row[column])
            rhs -= factor * float(self._b[row_index])

        self._T = _append_row_matrix(_append_column(self._T), row)
        self._b = _append_value(self._b, rhs)
        self._rc = _append_value(self._rc, 0.0)
        self._binv = _extend_inverse(self._binv, basic)
        self._basis.append(width)
        self._cost.append(0.0)
        self._blocked.append(False)
        self._signs.append(sign)
        if rhs < -self.tol:
            self._feasible = False

    # ------------------------------------------------------------- simplexes

    def _dual_vector(self) -> list[float]:
        if _np is not None:
            return list(_np.asarray(self._cost, dtype=float)[self._basis] @ self._binv)
        result = []
        for column in range(len(self._binv)):
            total = 0.0
            for row_index in range(len(self._binv)):
                factor = self._cost[self._basis[row_index]]
                if factor:
                    total += factor * self._binv[row_index][column]
            result.append(total)
        return result

    def _primal_simplex(self, max_iterations: int) -> tuple[str, int]:
        iterations = 0
        use_bland = False
        stall = 0
        n_columns = self.n_columns
        while iterations < max_iterations:
            iterations += 1
            enter = self._pricing(use_bland)
            if enter < 0:
                return "optimal", iterations
            best_ratio = float("inf")
            leave = -1
            for row_index in range(len(self._basis)):
                coefficient = _entry(self._T, row_index, enter)
                if coefficient <= self.tol:
                    continue
                ratio = float(self._b[row_index]) / coefficient
                closer = ratio < best_ratio - 1e-12
                tied = abs(ratio - best_ratio) <= 1e-12 and (leave < 0 or self._basis[row_index] < self._basis[leave])
                if closer or tied:
                    best_ratio = ratio
                    leave = row_index
            if leave < 0:
                return "unbounded", iterations
            self._pivot(leave, enter)
            if best_ratio <= 1e-12:
                stall += 1
                if stall >= _STALL_LIMIT:
                    use_bland = True
            else:
                stall = 0
        return "optimal", iterations

    def _dual_simplex(self, max_iterations: int) -> tuple[str, int]:
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            leave = -1
            worst = -self.tol
            for row_index in range(len(self._basis)):
                if float(self._b[row_index]) < worst:
                    worst = float(self._b[row_index])
                    leave = row_index
            if leave < 0:
                return "optimal", iterations
            enter = -1
            best_ratio = float("inf")
            for column in range(self.n_columns):
                if self._blocked[column]:
                    continue
                coefficient = _entry(self._T, leave, column)
                if coefficient >= -self.tol:
                    continue
                ratio = float(self._rc[column]) / (-coefficient)
                if ratio < best_ratio - 1e-12 or (abs(ratio - best_ratio) <= 1e-12 and (enter < 0 or column < enter)):
                    best_ratio = ratio
                    enter = column
            if enter < 0:
                return "infeasible", iterations
            self._pivot(leave, enter)
        return "optimal", iterations

    def _pricing(self, use_bland: bool) -> int:
        blocked = self._blocked
        if not blocked:
            return -1
        if use_bland:
            for column in range(self.n_columns):
                if not blocked[column] and float(self._rc[column]) < -self.tol:
                    return column
            return -1
        if _np is not None:
            # Blocked columns already sit at +inf in _rc, so this is a bare argmin.
            best = int(self._rc.argmin())
            return best if float(self._rc[best]) < -self.tol else -1
        best = -1
        for column in range(self.n_columns):
            if blocked[column]:
                continue
            if best < 0 or self._rc[column] < self._rc[best]:
                best = column
        return best if best >= 0 and self._rc[best] < -self.tol else -1

    def _pivot(self, leave: int, enter: int) -> None:
        if _np is not None:
            pivot = self._T[leave, enter]
            self._T[leave] = self._T[leave] / pivot
            self._b[leave] = self._b[leave] / pivot
            self._binv[leave] = self._binv[leave] / pivot
            factors = self._T[:, enter].copy()
            factors[leave] = 0.0
            pivot_row = self._T[leave]
            self._T -= factors[:, None] * pivot_row
            self._b -= factors * self._b[leave]
            self._binv -= factors[:, None] * self._binv[leave]
            self._rc -= self._rc[enter] * pivot_row
            self._basis[leave] = enter
            return

        row = self._T[leave]
        inverse = 1.0 / row[enter]
        for index in range(len(row)):
            row[index] *= inverse
        self._b[leave] *= inverse
        self._binv[leave] = [value * inverse for value in self._binv[leave]]
        for index in range(len(self._T)):
            if index == leave:
                continue
            factor = self._T[index][enter]
            if factor == 0.0:
                continue
            target = self._T[index]
            for column in range(len(row)):
                target[column] -= factor * row[column]
            self._b[index] -= factor * self._b[leave]
            inverse_row = self._binv[index]
            for column in range(len(self._binv)):
                inverse_row[column] -= factor * self._binv[leave][column]
        enter_cost = self._rc[enter]
        for column in range(len(row)):
            self._rc[column] -= enter_cost * row[column]
        self._basis[leave] = enter


def _sparse(coefficients) -> dict[int, float]:
    if isinstance(coefficients, dict):
        return {int(index): float(value) for index, value in coefficients.items() if value}
    return {index: float(value) for index, value in enumerate(coefficients) if value}


def _row(matrix, index):
    return matrix[index]


def _identity(n: int):
    if _np is not None:
        return _np.eye(n)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _empty_matrix(n_columns: int):
    if _np is not None:
        return _np.zeros((0, n_columns))
    return []


def _append_row_matrix(matrix, row):
    if _np is not None:
        return _np.vstack([matrix, _np.array([row], dtype=float)])
    return [list(entry) for entry in matrix] + [list(row)]


def _append_column(matrix):
    if _np is not None:
        return _np.hstack([matrix, _np.zeros((matrix.shape[0], 1))])
    for row in matrix:
        row.append(0.0)
    return matrix


def _append_value(values, value):
    if _np is not None:
        return _np.append(values, value)
    return list(values) + [value]


def _extend_inverse(matrix, basic):
    """Inverse of the basis after appending a row with basic coefficients ``basic``.

    With ``B' = [[B, 0], [basic, 1]]`` the inverse is
    ``[[B^-1, 0], [-basic B^-1, 1]]`` - the last row is *not* just ``e_n``, which
    is the trap this helper used to fall into.
    """
    n_rows = len(basic)
    if _np is not None:
        bigger = _np.zeros((n_rows + 1, n_rows + 1))
        bigger[:n_rows, :n_rows] = matrix
        bigger[n_rows, :n_rows] = -(_np.asarray(basic, dtype=float) @ matrix)
        bigger[n_rows, n_rows] = 1.0
        return bigger
    last = [0.0] * n_rows
    for row_index in range(n_rows):
        factor = basic[row_index]
        if not factor:
            continue
        source = matrix[row_index]
        for column in range(n_rows):
            last[column] -= factor * source[column]
    bigger = [list(row) + [0.0] for row in matrix]
    bigger.append(last + [1.0])
    return bigger
