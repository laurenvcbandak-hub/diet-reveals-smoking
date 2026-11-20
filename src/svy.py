"""
Design-based (Taylor linearization) GLMs for NHANES, matching R survey::svyglm defaults.
Subpopulations keep all PSUs of the full design (absent PSUs contribute zero).

Usage:
    from svy import Design
    des = Design(demo_df)                    # P_DEMO with SDMVSTRA / SDMVPSU / WTMECPRP
    m = des.glm("y ~ x + z", data, "logit")  # data needs columns w, SDMVSTRA, SDMVPSU
    m.ci("x", exp=True); m.p["x"]; m.wald(["C(cat)"])
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


class Design:
    def __init__(self, demo):
        d = demo.loc[demo.WTMECPRP > 0, ["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
        self.psus = pd.MultiIndex.from_frame(d.rename(columns={"SDMVSTRA": "s", "SDMVPSU": "p"}))
        self.n_strata = d.SDMVSTRA.nunique()
        self.ddf = len(d) - self.n_strata

    def glm(self, formula, data, family="gaussian"):
        return SvyFit(self, formula, data, family)

    def mean(self, x, data):
        """Weighted mean (or proportion) of x with design-based 95% CI."""
        d = data.assign(_y=x).dropna(subset=["_y"])
        return self.glm("_y ~ 1", d).ci("Intercept")


class SvyFit:
    def __init__(self, des, formula, data, family):
        fam = sm.families.Binomial() if family == "logit" else sm.families.Gaussian()
        self.des = des
        self.res = smf.glm(formula, data, family=fam, var_weights=data.w).fit()
        used = data.loc[self.res.model.data.row_labels]
        self.n = len(used)
        X, y = self.res.model.exog, self.res.model.endog
        w, mu = used.w.values, np.asarray(self.res.fittedvalues)
        v = mu * (1 - mu) if family == "logit" else np.ones_like(mu)
        H = (X * (w * v)[:, None]).T @ X
        U = pd.DataFrame(X * (w * (y - mu))[:, None])
        U["s"], U["p"] = used.SDMVSTRA.values, used.SDMVPSU.values
        tot = U.groupby(["s", "p"]).sum().reindex(des.psus, fill_value=0.0)
        M = np.zeros((X.shape[1],) * 2)
        for _, g in tot.groupby(level="s"):
            d = g.values - g.values.mean(axis=0)
            M += len(g) / (len(g) - 1) * d.T @ d
        Hi = np.linalg.inv(H)
        idx = self.res.params.index
        self.V = pd.DataFrame(Hi @ M @ Hi, index=idx, columns=idx)
        self.b = self.res.params
        self.se = pd.Series(np.sqrt(np.diag(self.V)), index=idx)
        self.p = pd.Series(2 * stats.t.sf(np.abs(self.b / self.se), des.ddf), index=idx)

    def ci(self, term, exp=False, scale=1.0):
        q = stats.t.ppf(0.975, self.des.ddf)
        b, se = self.b[term] * scale, self.se[term] * scale
        out = (b, b - q * se, b + q * se)
        return tuple(float(x) for x in (np.exp(out) if exp else out))

    def wald(self, prefixes):
        idx = [t for t in self.b.index if any(t.startswith(x) for x in prefixes)]
        b, V, q = self.b[idx].values, self.V.loc[idx, idx].values, len(idx)
        F = float(b @ np.linalg.solve(V, b)) / q
        return float(stats.f.sf(F, q, self.des.ddf - q + 1))
