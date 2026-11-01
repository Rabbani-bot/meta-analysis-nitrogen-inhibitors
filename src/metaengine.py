"""Meta-analysis engine implementing the methods offered in MetaWin (and metafor), in plain NumPy/SciPy.

Effect sizes : log response ratio (lnRR, Hedges et al. 1999), Hedges' g
Models       : fixed effect; random effects with DerSimonian-Laird or REML tau^2
Inference    : z or Knapp-Hartung (t) confidence intervals; prediction interval;
               Q, I^2, H^2; bias-corrected bootstrap CI
Structure    : subgroup (categorical) analysis with Q_between; weighted meta-regression
Bias / robustness : Egger regression, trim-and-fill (L0), Rosenthal fail-safe N,
               leave-one-out, cumulative meta-analysis
Validated in tests against the BCG vaccine benchmark (metafor documentation values).
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats


# ---------- effect sizes ----------
def lnrr(m_t, sd_t, n_t, m_c, sd_c, n_c):
    """Log response ratio and its sampling variance (Hedges, Gurevitch & Curtis 1999)."""
    m_t, sd_t, n_t, m_c, sd_c, n_c = map(np.asarray, (m_t, sd_t, n_t, m_c, sd_c, n_c))
    if np.any(m_t <= 0) or np.any(m_c <= 0):
        raise ValueError("lnRR requires positive means")
    yi = np.log(m_t / m_c)
    vi = sd_t**2 / (n_t * m_t**2) + sd_c**2 / (n_c * m_c**2)
    return yi, vi


def hedges_g(m_t, sd_t, n_t, m_c, sd_c, n_c):
    m_t, sd_t, n_t, m_c, sd_c, n_c = map(np.asarray, (m_t, sd_t, n_t, m_c, sd_c, n_c))
    sp = np.sqrt(((n_t - 1) * sd_t**2 + (n_c - 1) * sd_c**2) / (n_t + n_c - 2))
    j = 1 - 3 / (4 * (n_t + n_c) - 9)
    g = j * (m_t - m_c) / sp
    v = (n_t + n_c) / (n_t * n_c) + g**2 / (2 * (n_t + n_c))
    return g, v


# ---------- core models ----------
@dataclass
class Result:
    k: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pi_low: float
    pi_high: float
    z_or_t: float
    p: float
    tau2: float
    Q: float
    Q_df: int
    Q_p: float
    I2: float
    H2: float
    method: str

    def pct(self):
        """Back-transform lnRR to % change."""
        f = lambda x: 100 * (np.exp(x) - 1)
        return {"estimate_pct": f(self.estimate), "ci_low_pct": f(self.ci_low), "ci_high_pct": f(self.ci_high),
                "pi_low_pct": f(self.pi_low), "pi_high_pct": f(self.pi_high)}

    def as_dict(self):
        return asdict(self)


def _q_stat(y, v):
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - mu) ** 2))


def tau2_dl(y, v):
    w = 1 / v
    Q = _q_stat(y, v)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return max(0.0, (Q - (len(y) - 1)) / c)


def tau2_reml(y, v, tol=1e-10, max_iter=500):
    """REML estimate of tau^2 by Fisher scoring (as in metafor), starting from DL."""
    t2 = tau2_dl(y, v)
    for _ in range(max_iter):
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        P_y = w * (y - mu)
        # REML score and information
        trP = np.sum(w) - np.sum(w**2) / np.sum(w)
        score = 0.5 * (np.sum(P_y**2) - trP)
        trPP = np.sum(w**2) - 2 * np.sum(w**3) / np.sum(w) + (np.sum(w**2) / np.sum(w)) ** 2
        info = 0.5 * trPP
        step = score / info
        new = max(0.0, t2 + step)
        if abs(new - t2) < tol:
            return new
        t2 = new
    return t2


def meta(y, v, method="REML", knha=False, level=0.95) -> Result:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if method == "FE":
        t2 = 0.0
    elif method == "DL":
        t2 = tau2_dl(y, v)
    elif method == "REML":
        t2 = tau2_reml(y, v)
    else:
        raise ValueError(method)
    w = 1 / (v + t2)
    mu = np.sum(w * y) / np.sum(w)
    se = np.sqrt(1 / np.sum(w))
    alpha = 1 - level
    if knha and k > 1:
        s2 = np.sum(w * (y - mu) ** 2) / (k - 1)
        se = se * np.sqrt(max(s2, 1e-12))
        crit = stats.t.ppf(1 - alpha / 2, k - 1)
        stat = mu / se
        p = 2 * stats.t.sf(abs(stat), k - 1)
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        stat = mu / se
        p = 2 * stats.norm.sf(abs(stat))
    Q = _q_stat(y, v)
    wf = 1 / v
    vtyp = (k - 1) * np.sum(wf) / (np.sum(wf) ** 2 - np.sum(wf**2))
    I2 = 100 * t2 / (t2 + vtyp) if k > 1 else 0.0
    H2 = (t2 + vtyp) / vtyp if k > 1 else 1.0
    pcrit = stats.t.ppf(1 - alpha / 2, max(k - 2, 1))
    pi_half = pcrit * np.sqrt(t2 + se**2)
    return Result(k, float(mu), float(se), float(mu - crit * se), float(mu + crit * se),
                  float(mu - pi_half), float(mu + pi_half), float(stat), float(p), float(t2),
                  Q, k - 1, float(stats.chi2.sf(Q, k - 1)), float(I2), float(H2),
                  method + ("+KnHa" if knha else ""))


# ---------- structure ----------
def subgroups(y, v, groups, method="REML"):
    """Separate random-effects model per group; Q_between from a mixed-effects model
    (common tau^2 estimated from within-group residual heterogeneity, as in MetaWin)."""
    y, v, groups = np.asarray(y, float), np.asarray(v, float), np.asarray(groups)
    levels = list(dict.fromkeys(groups))
    per = {g: meta(y[groups == g], v[groups == g], method) for g in levels if (groups == g).sum() >= 2}
    # pooled within-group tau^2 (DL-type)
    Qw = sum(_q_stat(y[groups == g], v[groups == g]) for g in levels)
    c = 0.0
    for g in levels:
        wg = 1 / v[groups == g]
        c += np.sum(wg) - np.sum(wg**2) / np.sum(wg)
    t2 = max(0.0, (Qw - (len(y) - len(levels))) / c)
    w = 1 / (v + t2)
    mu_all = np.sum(w * y) / np.sum(w)
    Qb = 0.0
    for g in levels:
        m = groups == g
        mu_g = np.sum(w[m] * y[m]) / np.sum(w[m])
        Qb += np.sum(w[m]) * (mu_g - mu_all) ** 2
    df = len(levels) - 1
    return per, {"Q_between": float(Qb), "df": df, "p": float(stats.chi2.sf(Qb, df)), "tau2_within": float(t2)}


def metareg(y, v, X, method="REML"):
    """Weighted least-squares meta-regression with residual tau^2 (method-of-moments / REML-like)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    k, p = X.shape

    def fit(t2):
        W = np.diag(1 / (v + t2))
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        b = XtWX_inv @ X.T @ W @ y
        return b, XtWX_inv

    # residual heterogeneity by REML fixed-point
    t2 = 0.0
    for _ in range(500):
        b, _ = fit(t2)
        r = y - X @ b
        W = np.diag(1 / (v + t2))
        P = W - W @ X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W
        new = max(0.0, t2 + (r @ W @ W @ r - np.trace(P)) / np.trace(P @ P))
        if abs(new - t2) < 1e-10:
            break
        t2 = new
    b, cov = fit(t2)
    se = np.sqrt(np.diag(cov))
    z = b / se
    Wf = np.diag(1 / v)
    resid = y - X @ np.linalg.solve(X.T @ Wf @ X, X.T @ Wf @ y)
    QE = float(resid @ Wf @ resid)
    t2_null = tau2_reml(y, v)
    R2 = max(0.0, 100 * (t2_null - t2) / t2_null) if t2_null > 0 else 0.0
    return {"coef": b, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)), "tau2_resid": t2,
            "QE": QE, "QE_df": k - p, "QE_p": float(stats.chi2.sf(QE, k - p)), "R2_pct": R2}


# ---------- bias & robustness ----------
def egger(y, v):
    se = np.sqrt(v)
    z = y / se
    prec = 1 / se
    X = np.column_stack([np.ones_like(prec), prec])
    b, res, *_ = np.linalg.lstsq(X, z, rcond=None)
    n = len(y)
    s2 = np.sum((z - X @ b) ** 2) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = b[0] / np.sqrt(cov[0, 0])
    return {"intercept": float(b[0]), "se": float(np.sqrt(cov[0, 0])), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), n - 2))}


def trim_and_fill(y, v, method="REML", side=None, max_iter=50):
    """Duval & Tweedie trim-and-fill with the L0 estimator."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if side is None:  # missing studies assumed on the side opposite to Egger asymmetry
        side = "left" if egger(y, v)["intercept"] > 0 else "right"
    sign = 1 if side == "left" else -1
    yy = sign * y
    k0 = 0
    for _ in range(max_iter):
        order = np.argsort(yy)
        trimmed = order[: k - k0] if k0 > 0 else order
        mu = meta(yy[trimmed], v[trimmed], "FE").estimate
        dev = yy - mu
        ranks = stats.rankdata(np.abs(dev))
        Tn = np.sum(ranks[dev > 0])
        new = int(max(0, round((4 * Tn - k * (k + 1)) / (2 * k - 1))))
        if new == k0:
            break
        k0 = new
    order = np.argsort(yy)
    keep = order[: k - k0] if k0 > 0 else order
    mu = meta(yy[keep], v[keep], "FE").estimate
    largest = order[k - k0:] if k0 > 0 else np.array([], int)
    filled_y = sign * (2 * mu - yy[largest])
    filled_v = v[largest]
    adj = meta(np.r_[y, filled_y], np.r_[v, filled_v], method)
    return {"k0": k0, "side": side, "filled_y": filled_y, "filled_v": filled_v, "adjusted": adj}


def failsafe_n(y, v, alpha=0.05):
    z = y / np.sqrt(v)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(max(0.0, (np.sum(z) ** 2) / zc**2 - len(y)))


def leave_one_out(y, v, method="REML"):
    return [meta(np.delete(y, i), np.delete(v, i), method) for i in range(len(y))]


def cumulative(y, v, order, method="REML"):
    idx = np.argsort(order, kind="stable")
    return [meta(y[idx[: i + 1]], v[idx[: i + 1]], method) for i in range(1, len(y))], idx


def bootstrap_ci(y, v, B=2000, seed=0, method="REML", level=0.95):
    """Bias-corrected (BC) bootstrap CI of the pooled effect (resampling studies)."""
    rng = np.random.default_rng(seed)
    est = meta(y, v, method).estimate
    k = len(y)
    boots = np.empty(B)
    for b in range(B):
        i = rng.integers(0, k, k)
        boots[b] = meta(y[i], v[i], method).estimate
    z0 = stats.norm.ppf(np.mean(boots < est))
    a = 1 - level
    lo = stats.norm.cdf(2 * z0 + stats.norm.ppf(a / 2))
    hi = stats.norm.cdf(2 * z0 + stats.norm.ppf(1 - a / 2))
    return float(np.quantile(boots, lo)), float(np.quantile(boots, hi)), boots
