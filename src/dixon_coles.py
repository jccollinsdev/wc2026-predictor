"""Time-decayed Dixon-Coles bivariate Poisson model (custom MLE with analytic gradient).

log lambda_home = atk[home] - def[away] + home_adv * (0 if neutral else 1)
log lambda_away = atk[away] - def[home]

Match likelihood: P(x,y) = tau(x,y,lh,la,rho) * Poisson(x;lh) * Poisson(y;la)
  tau(0,0)=1-lh*la*rho ; tau(0,1)=1+lh*rho ; tau(1,0)=1+la*rho ; tau(1,1)=1-rho ; else 1

Weighted by w = exp(-xi * days_before) * (friendly_factor if friendly).
L2 on atk/def breaks the additive degeneracy and shrinks sparse teams (good for internationals).

Analytic gradient -> L-BFGS-B fits ~250 teams (~500 params) over ~20k matches in a few seconds,
making walk-forward refits feasible on CPU.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (DC_TRAIN_FROM, DC_TIME_DECAY, DC_FRIENDLY_WEIGHT, DC_REG,
                    DC_MAX_GOALS, DC_RHO_BOUNDS)

_LOG_FACT = np.array([math.lgamma(k + 1) for k in range(DC_MAX_GOALS + 1)])


def _poisson_pmf_col(lmbda, kmax):
    """Vector of Poisson pmf for k=0..kmax for a scalar lambda."""
    k = np.arange(kmax + 1)
    return np.exp(-lmbda + k * np.log(lmbda) - _LOG_FACT[: kmax + 1])


class DixonColes:
    def __init__(self, time_decay=DC_TIME_DECAY, friendly_weight=DC_FRIENDLY_WEIGHT,
                 reg=DC_REG, train_from=DC_TRAIN_FROM):
        self.xi = time_decay
        self.friendly_weight = friendly_weight
        self.reg = reg
        self.train_from = pd.Timestamp(train_from)
        self.teams = None
        self.team_idx = None
        self.atk = None
        self.def_ = None
        self.home = None
        self.rho = None
        self.ref_date = None

    # ---- fitting -------------------------------------------------------------
    def fit(self, played: pd.DataFrame, ref_date, verbose=False):
        ref_date = pd.Timestamp(ref_date)
        self.ref_date = ref_date
        d = played[(played["date"] < ref_date) & (played["date"] >= self.train_from)].copy()
        if len(d) == 0:
            raise ValueError("no training matches in window")

        teams = sorted(pd.concat([d["home_team"], d["away_team"]]).unique())
        self.teams = teams
        self.team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = d["home_team"].map(self.team_idx).values
        a = d["away_team"].map(self.team_idx).values
        x = d["home_score"].values.astype(float)
        y = d["away_score"].values.astype(float)
        homeflag = (~d["neutral"].values).astype(float)
        days = (ref_date - d["date"]).dt.days.values.astype(float)
        w = np.exp(-self.xi * days)
        is_friendly = d["tournament"].str.lower().str.contains("friendly").values
        w = w * np.where(is_friendly, self.friendly_weight, 1.0)

        # low-score masks for tau
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        self._cache = dict(h=h, a=a, x=x, y=y, homeflag=homeflag, w=w,
                           m00=m00, m01=m01, m10=m10, m11=m11, n=n)

        # init: atk/def 0, home 0.25, rho -0.08
        theta0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.08]])
        bounds = [(-3.0, 3.0)] * n + [(-3.0, 3.0)] * n + [(0.0, 1.0), DC_RHO_BOUNDS]

        res = minimize(self._objective_grad, theta0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 400, "ftol": 1e-9})
        self._unpack(res.x)
        if verbose:
            print(f"DC fit: teams={n} matches={len(d)} success={res.success} "
                  f"nll={res.fun:.1f} iters={res.nit} home={self.home:.3f} rho={self.rho:.3f}")
        return self

    def _unpack(self, theta):
        n = self._cache["n"]
        self.atk = theta[:n].copy()
        self.def_ = theta[n:2 * n].copy()
        self.home = float(theta[2 * n])
        self.rho = float(theta[2 * n + 1])

    def _objective_grad(self, theta):
        c = self._cache
        n = c["n"]
        h, a, x, y, hf, w = c["h"], c["a"], c["x"], c["y"], c["homeflag"], c["w"]
        atk = theta[:n]
        df_ = theta[n:2 * n]
        home = theta[2 * n]
        rho = theta[2 * n + 1]

        log_lh = atk[h] - df_[a] + home * hf
        log_la = atk[a] - df_[h]
        np.clip(log_lh, -8, 8, out=log_lh)
        np.clip(log_la, -8, 8, out=log_la)
        lh = np.exp(log_lh)
        la = np.exp(log_la)

        # Poisson log-likelihood (drop factorial constants; they don't affect argmin)
        ll = w * (x * log_lh - lh + y * log_la - la)

        # tau term
        tau = np.ones_like(lh)
        m00, m01, m10, m11 = c["m00"], c["m01"], c["m10"], c["m11"]
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-9, None)
        ll = ll + w * np.log(tau)

        nll = -np.sum(ll) + self.reg * (np.dot(atk, atk) + np.dot(df_, df_))

        # ----- gradient -----
        g_atk = np.zeros(n)
        g_def = np.zeros(n)
        # Poisson residuals
        rh = (x - lh)        # d/d log_lh of poisson part
        ra = (y - la)
        # scatter poisson contributions (note: we accumulate dLL/dparam, negate at end)
        np.add.at(g_atk, h, w * rh)
        np.add.at(g_def, a, -w * rh)
        np.add.at(g_atk, a, w * ra)
        np.add.at(g_def, h, -w * ra)
        g_home = np.sum(w * rh * hf)
        g_rho = 0.0

        # tau contributions: dlogtau/dlh * lh chains to atk[h], def[a], home
        dlt_dlh = np.zeros_like(lh)
        dlt_dla = np.zeros_like(la)
        dlt_drho = np.zeros_like(lh)
        # (0,0)
        inv = 1.0 / tau[m00]
        dlt_dlh[m00] = (-la[m00] * rho) * inv
        dlt_dla[m00] = (-lh[m00] * rho) * inv
        dlt_drho[m00] = (-lh[m00] * la[m00]) * inv
        # (0,1)
        inv = 1.0 / tau[m01]
        dlt_dlh[m01] = rho * inv
        dlt_drho[m01] = lh[m01] * inv
        # (1,0)
        inv = 1.0 / tau[m10]
        dlt_dla[m10] = rho * inv
        dlt_drho[m10] = la[m10] * inv
        # (1,1)
        inv = 1.0 / tau[m11]
        dlt_drho[m11] = -1.0 * inv

        # chain: dlogtau/datk[h] = dlt_dlh * lh ; etc.
        ch_h = w * dlt_dlh * lh   # affects atk[h] (+), def[a] (-), home (+*hf)
        ch_a = w * dlt_dla * la   # affects atk[a] (+), def[h] (-)
        np.add.at(g_atk, h, ch_h)
        np.add.at(g_def, a, -ch_h)
        np.add.at(g_atk, a, ch_a)
        np.add.at(g_def, h, -ch_a)
        g_home += np.sum(ch_h * hf)
        g_rho += np.sum(w * dlt_drho)

        # objective is nll = -ll + reg*||.||  -> grad = -(dLL) + 2*reg*param
        grad = np.empty_like(theta)
        grad[:n] = -g_atk + 2 * self.reg * atk
        grad[n:2 * n] = -g_def + 2 * self.reg * df_
        grad[2 * n] = -g_home
        grad[2 * n + 1] = -g_rho
        return nll, grad

    # ---- prediction ----------------------------------------------------------
    def _lambdas(self, home_team, away_team, neutral):
        ai = self.team_idx.get(home_team)
        bi = self.team_idx.get(away_team)
        atk_h = self.atk[ai] if ai is not None else 0.0
        def_h = self.def_[ai] if ai is not None else 0.0
        atk_a = self.atk[bi] if bi is not None else 0.0
        def_a = self.def_[bi] if bi is not None else 0.0
        hf = 0.0 if neutral else 1.0
        lh = math.exp(atk_h - def_a + self.home * hf)
        la = math.exp(atk_a - def_h)
        return lh, la, (ai is None or bi is None)

    def score_grid(self, lh, la, kmax=DC_MAX_GOALS, apply_tau=True, rho=None):
        rho = self.rho if rho is None else rho
        ph = _poisson_pmf_col(lh, kmax)
        pa = _poisson_pmf_col(la, kmax)
        grid = np.outer(ph, pa)  # grid[x, y]
        if apply_tau:
            grid[0, 0] *= (1.0 - lh * la * rho)
            grid[0, 1] *= (1.0 + lh * rho)
            grid[1, 0] *= (1.0 + la * rho)
            grid[1, 1] *= (1.0 - rho)
        grid = np.clip(grid, 0, None)
        grid /= grid.sum()
        return grid

    def predict(self, home_team, away_team, neutral=False, kmax=DC_MAX_GOALS):
        lh, la, missing = self._lambdas(home_team, away_team, neutral)
        grid = self.score_grid(lh, la, kmax)
        return _grid_summary(grid, lh, la, missing)


def _grid_summary(grid, lh, la, missing=False):
    kmax = grid.shape[0] - 1
    idx = np.indices(grid.shape)
    X, Y = idx[0], idx[1]
    p_home = grid[X > Y].sum()
    p_draw = grid[X == Y].sum()
    p_away = grid[X < Y].sum()
    ml = np.unravel_index(np.argmax(grid), grid.shape)
    exp_home = (X * grid).sum()
    exp_away = (Y * grid).sum()
    # top scorelines
    flat = [((i, j), grid[i, j]) for i in range(kmax + 1) for j in range(kmax + 1)]
    flat.sort(key=lambda kv: -kv[1])
    top = [(s, float(p)) for s, p in flat[:6]]
    # most likely score within each outcome bucket
    def best(mask_fn):
        best_s, best_p = None, -1
        for i in range(kmax + 1):
            for j in range(kmax + 1):
                if mask_fn(i, j) and grid[i, j] > best_p:
                    best_p, best_s = grid[i, j], (i, j)
        return best_s, float(best_p)
    return dict(
        lambda_home=lh, lambda_away=la, missing=missing,
        p_home=float(p_home), p_draw=float(p_draw), p_away=float(p_away),
        most_likely_score=(int(ml[0]), int(ml[1])),
        exp_goals_home=float(exp_home), exp_goals_away=float(exp_away),
        top_scorelines=top,
        best_home_win=best(lambda i, j: i > j),
        best_draw=best(lambda i, j: i == j),
        best_away_win=best(lambda i, j: i < j),
        grid=grid,
    )


if __name__ == "__main__":
    from data import load_results, played_matches
    df = load_results()
    pm = played_matches(df)

    # gradient check on a tiny sub-window
    m = DixonColes(train_from="2024-01-01")
    m.ref_date = pd.Timestamp("2026-06-12")
    sub = pm[(pm.date >= "2024-01-01") & (pm.date < "2026-06-12")]
    # build cache via fit-prep by calling _objective on random theta
    teams = sorted(pd.concat([sub.home_team, sub.away_team]).unique())
    m.team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    m._cache = None
    # reuse fit's cache builder by calling fit partially is complex; do a direct mini build:
    h = sub["home_team"].map(m.team_idx).values
    a = sub["away_team"].map(m.team_idx).values
    x = sub["home_score"].values.astype(float); y = sub["away_score"].values.astype(float)
    hf = (~sub["neutral"].values).astype(float)
    days = (m.ref_date - sub["date"]).dt.days.values.astype(float)
    w = np.exp(-m.xi * days)
    m._cache = dict(h=h, a=a, x=x, y=y, homeflag=hf, w=w,
                    m00=(x==0)&(y==0), m01=(x==0)&(y==1), m10=(x==1)&(y==0), m11=(x==1)&(y==1), n=n)
    rng = np.random.default_rng(0)
    theta = np.concatenate([rng.normal(0, 0.3, n), rng.normal(0, 0.3, n), [0.25, -0.08]])
    f0, g = m._objective_grad(theta)
    # finite diff on 8 random coords
    eps = 1e-5
    maxerr = 0
    for idx in rng.choice(len(theta), 8, replace=False):
        tp = theta.copy(); tp[idx] += eps
        tm = theta.copy(); tm[idx] -= eps
        fd = (m._objective_grad(tp)[0] - m._objective_grad(tm)[0]) / (2 * eps)
        err = abs(fd - g[idx])
        maxerr = max(maxerr, err)
    print(f"gradient check max|analytic-fd| = {maxerr:.2e}  (should be < 1e-3)")

    # full fit + USA/Paraguay prediction
    m2 = DixonColes().fit(pm, ref_date="2026-06-12", verbose=True)
    pred = m2.predict("United States", "Paraguay", neutral=False)
    print("\nUSA vs Paraguay (Dixon-Coles):")
    print(f"  xG: USA {pred['exp_goals_home']:.2f} - {pred['exp_goals_away']:.2f} Paraguay")
    print(f"  1X2: home {pred['p_home']:.3f} draw {pred['p_draw']:.3f} away {pred['p_away']:.3f}")
    print(f"  most likely score: {pred['most_likely_score']}")
    print(f"  top scorelines: {pred['top_scorelines'][:5]}")
