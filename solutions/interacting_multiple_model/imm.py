"""Interacting multiple-model estimator.

Mixing and mode probabilities follow Labbe's filterpy IMMEstimator:
cbar = mu @ M, omega[i, j] = M[i, j] * mu[i] / cbar[j].
Prediction has no external input.

The covariance is the diagonal of the full matrix, stored as a sparse diagonal.
A Kalman update would otherwise fill K R K^T into a dense E^2 by E^2 matrix.
"""

import numpy as np
from scipy import sparse


def _column(values):
    return np.asarray(values, dtype=float).reshape(-1, 1)


def _variance(cov):
    values = np.asarray(cov, dtype=float)
    if values.ndim == 1:
        return np.maximum(values, 0.0).copy()
    return np.maximum(np.diag(values), 0.0).copy()


def _apply(operator, vector):
    out = operator @ np.asarray(vector, dtype=float).reshape(-1)
    return _column(np.asarray(out, dtype=float).reshape(-1))


def _variance_map(operator, variance, squared=None):
    """Diagonal of A diag(variance) A^T, without forming that product."""
    if squared is not None:
        reduced = squared @ variance
    else:
        matrix = operator.tocsr()
        counts = np.diff(matrix.indptr)
        if matrix.nnz == 0:
            return np.zeros(matrix.shape[0])
        contrib = np.square(matrix.data) * variance[matrix.indices]
        reduced = np.add.reduceat(contrib, matrix.indptr[:-1])
        reduced[counts == 0] = 0.0
    if not np.isfinite(reduced).all():
        reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(reduced, dtype=float).ravel()


def _log_gauss(innov, cov):
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e12
    try:
        quad = float(np.asarray(innov).reshape(-1) @ np.linalg.solve(cov, np.asarray(innov).reshape(-1)))
    except np.linalg.LinAlgError:
        return -1e12
    if not np.isfinite(quad):
        return -1e12
    return -0.5 * (innov.size * np.log(2.0 * np.pi) + logdet + quad)


def _sparse_gram(observation, variance):
    scaled = observation.tocsr(copy=True)
    scaled.data = observation.data * variance[observation.indices]
    gram = scaled @ observation.T
    gram = gram.toarray() if sparse.issparse(gram) else np.asarray(gram, dtype=float)
    return np.atleast_2d(np.asarray(gram, dtype=float))


class IMMEstimator:
    """Two-or-more linear filters with the filterpy IMM recursion."""

    def __init__(self, mu, transition):
        self.mu = np.asarray(mu, dtype=float).reshape(-1)
        self.mu = self.mu / self.mu.sum()
        self.M = np.asarray(transition, dtype=float)
        self.N = self.mu.size
        if self.M.shape != (self.N, self.N):
            raise ValueError("transition must be N by N")
        self.xs: list[np.ndarray] = []
        self.ps: list[np.ndarray] = []
        self.x = np.zeros((0, 1))
        self.p = np.zeros(0)
        self.operators: list = []
        self.squares: list = []
        self.biases: list[np.ndarray] = []
        self.process: list[np.ndarray] = []
        self.repeats = 1
        self.likelihood = np.zeros(self.N)
        self.omega = np.zeros((self.N, self.N))
        self.cbar = np.zeros(self.N)
        self._compute_mixing_probabilities()

    @property
    def P(self):
        return sparse.diags(np.asarray(self.p, dtype=float).ravel(), format="csr")

    def spawn(self):
        """A private filter that shares the read-only transition operators."""
        child = IMMEstimator(np.full(self.N, 1.0 / self.N), self.M)
        child.operators = self.operators
        child.squares = self.squares
        child.biases = self.biases
        child.process = self.process
        child.repeats = self.repeats
        if self.xs:
            child.xs = [state.copy() for state in self.xs]
            child.ps = [variance.copy() for variance in self.ps]
            child.x = self.x.copy()
            child.p = self.p.copy()
            child.mu = self.mu.copy()
            child.omega = self.omega.copy()
            child.cbar = self.cbar.copy()
        return child

    def _require_models(self):
        if not self.operators or not self.xs:
            raise RuntimeError("IMM models are not set")

    def set_models(self, operators, biases, process, x0, cov0, repeats=1):
        self.operators = operators
        self.squares = [_square_operator(operator) for operator in operators]
        self.biases = [_column(bias) for bias in biases]
        self.process = [_variance(noise) for noise in process]
        self.repeats = max(1, int(repeats))
        x0 = _column(x0)
        variance = _variance(cov0)
        self.xs = [x0.copy() for _ in range(self.N)]
        self.ps = [variance.copy() for _ in range(self.N)]
        self.x = x0.copy()
        self.p = variance.copy()
        self._compute_state_estimate()

    def reset(self, x0, cov0):
        x0 = _column(x0)
        variance = _variance(cov0)
        self.xs = [x0.copy() for _ in range(self.N)]
        self.ps = [variance.copy() for _ in range(self.N)]
        self.x = x0.copy()
        self.p = variance.copy()
        self.mu[:] = 1.0 / self.N
        self._compute_mixing_probabilities()
        self._compute_state_estimate()

    def predict(self):
        """Mix, then propagate each mode. There is no control input."""
        self._require_models()
        mixed_x, mixed_p = self._mix()
        for index in range(self.N):
            state = mixed_x[index]
            variance = mixed_p[index]
            for _ in range(self.repeats):
                state = _apply(self.operators[index], state) + self.biases[index]
                variance = _variance_map(self.operators[index], variance, self.squares[index])
            if not np.isfinite(state).all():
                state = np.nan_to_num(state, nan=0.0)
            variance = np.maximum(variance + self.process[index], 1e-12)
            self.xs[index] = state
            self.ps[index] = variance
        self._compute_state_estimate()

    def update(self, packs):
        """Update every mode. packs is None when this step has no measurement."""
        if packs is None:
            return
        for index, pack in enumerate(packs):
            self.likelihood[index] = np.exp(np.clip(self._correct(index, pack), -700.0, 0.0))
        self.mu = self.cbar * self.likelihood
        total = float(np.nansum(self.mu))
        if not np.isfinite(total) or total <= 0:
            self.mu = np.full(self.N, 1.0 / self.N)
        else:
            self.mu = self.mu / total
        self._compute_mixing_probabilities()
        self._compute_state_estimate()

    def _correct(self, index, pack):
        y, observation, offset, noise = pack
        if np.asarray(y).size == 0:
            return 0.0
        observation = observation.tocsr() if sparse.issparse(observation) else sparse.csr_matrix(observation)
        state = self.xs[index].reshape(-1).astype(float, copy=True)
        variance = self.ps[index]
        predicted = np.asarray(observation @ state, dtype=float).ravel() + np.asarray(offset, dtype=float).ravel()
        innov = np.asarray(y, dtype=float).ravel() - predicted
        measure = np.atleast_2d(np.asarray(noise, dtype=float))
        if _columns_are_disjoint(observation, variance.size) and _is_diagonal(measure):
            loglik, state, variance = _correct_disjoint(state, variance, observation, innov, measure)
            self.xs[index] = _column(state if np.isfinite(state).all() else self.xs[index])
            self.ps[index] = variance
            return loglik
        gram = _sparse_gram(observation, variance)
        gram = 0.5 * (gram + gram.T) + measure
        gram = 0.5 * (gram + gram.T) + 1e-8 * np.eye(innov.size)
        loglik = _log_gauss(innov, gram)
        try:
            solved = np.linalg.solve(gram, innov)
            weight = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            return -1e12
        state = state + variance * np.asarray(observation.T @ solved, dtype=float).ravel()
        self.xs[index] = _column(state if np.isfinite(state).all() else self.xs[index])
        self.ps[index] = _joseph_diagonal(observation, variance, weight, measure)
        return loglik

    def _mix(self):
        mixed_x, mixed_p = [], []
        for index in range(self.N):
            weights = self.omega[:, index]
            state = np.zeros_like(self.xs[0])
            for source, weight in enumerate(weights):
                state = state + weight * self.xs[source]
            variance = np.zeros_like(self.ps[0])
            center = state.reshape(-1)
            for source, weight in enumerate(weights):
                gap = self.xs[source].reshape(-1) - center
                variance = variance + weight * (np.square(gap) + self.ps[source])
            mixed_x.append(state)
            mixed_p.append(np.maximum(variance, 0.0))
        return mixed_x, mixed_p

    def _compute_state_estimate(self):
        acc = np.zeros_like(self.xs[0])
        for weight, state in zip(self.mu, self.xs):
            acc = acc + weight * state
        self.x = acc
        variance = np.zeros_like(self.ps[0])
        center = acc.reshape(-1)
        for weight, state, mode_variance in zip(self.mu, self.xs, self.ps):
            gap = state.reshape(-1) - center
            variance = variance + weight * (np.square(gap) + mode_variance)
        self.p = np.maximum(variance, 0.0)

    def _compute_mixing_probabilities(self):
        self.cbar = self.mu @ self.M
        self.cbar = np.where(self.cbar <= 1e-15, 1e-15, self.cbar)
        for source in range(self.N):
            for target in range(self.N):
                self.omega[source, target] = (self.M[source, target] * self.mu[source]) / self.cbar[target]


def _square_operator(operator):
    if sparse.issparse(operator):
        return None
    return np.square(np.asarray(operator, dtype=float))


def _is_diagonal(measure):
    if measure.size == 0:
        return True
    off = measure - np.diag(np.diag(measure))
    return float(np.max(np.abs(off))) <= 1e-12


def _columns_are_disjoint(observation, width):
    if observation.nnz == 0:
        return True
    counts = np.bincount(observation.indices, minlength=width)
    return int(counts.max()) <= 1


def _correct_disjoint(state, variance, observation, innov, measure):
    """Scalar Joseph update. Observation columns do not overlap, so S is diagonal."""
    counts = np.diff(observation.indptr)
    rows = np.repeat(np.arange(observation.shape[0]), counts)
    quad = np.zeros(innov.size)
    if observation.nnz:
        np.add.at(quad, rows, np.square(observation.data) * variance[observation.indices])
    sensor = np.diag(measure)
    scale = np.maximum(quad + sensor, 1e-8)
    loglik = -0.5 * (innov.size * np.log(2.0 * np.pi) + float(np.sum(np.log(scale)) + np.sum(np.square(innov) / scale)))
    gain = innov / scale
    updated_state = state.copy()
    updated_variance = variance.copy()
    if observation.nnz:
        shrinkage = variance[observation.indices] * np.square(observation.data) / scale[rows]
        updated_state[observation.indices] += variance[observation.indices] * observation.data * gain[rows]
        updated_variance[observation.indices] = (
            np.square(1.0 - shrinkage) * variance[observation.indices]
            + np.square(variance[observation.indices]) * np.square(observation.data) * sensor[rows] / np.square(scale[rows])
        )
    if not np.isfinite(updated_state).all():
        updated_state = state
    if not np.isfinite(updated_variance).all():
        updated_variance = variance
    return loglik, updated_state, np.maximum(updated_variance, 1e-12)


def _joseph_diagonal(observation, variance, weight, noise):
    """Diagonal of the Joseph update. Each state column is touched by few rows."""
    observation = observation.tocsr()
    wrapped = weight @ noise @ weight
    row_of = np.empty(observation.nnz, dtype=int)
    for row in range(observation.shape[0]):
        row_of[observation.indptr[row]:observation.indptr[row + 1]] = row
    if observation.nnz == 0:
        return np.maximum(variance, 1e-12)
    order = np.argsort(observation.indices, kind="mergesort")
    columns = observation.indices[order]
    values = observation.data[order]
    rows = row_of[order]
    unique, starts, counts = np.unique(columns, return_index=True, return_counts=True)
    leverage = np.zeros(variance.size)
    measurement = np.zeros(variance.size)
    single = counts == 1
    chosen = starts[single]
    cols = columns[chosen]
    gain = values[chosen]
    active = rows[chosen]
    leverage[cols] = np.square(gain) * weight[active, active]
    measurement[cols] = np.square(gain) * wrapped[active, active]
    for column, start, count in zip(unique[~single], starts[~single], counts[~single]):
        block = slice(start, start + count)
        coeff = values[block]
        active_rows = rows[block]
        quadratic = coeff @ weight[np.ix_(active_rows, active_rows)] @ coeff
        sensor = coeff @ wrapped[np.ix_(active_rows, active_rows)] @ coeff
        leverage[column] = quadratic
        measurement[column] = sensor
    shrinkage = variance * leverage
    updated = np.square(1.0 - shrinkage) * variance + np.square(variance) * measurement
    if not np.isfinite(updated).all():
        updated = np.nan_to_num(variance, nan=1e-12, posinf=1e12, neginf=1e-12)
    return np.maximum(updated, 1e-12)
