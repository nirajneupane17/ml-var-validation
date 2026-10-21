"""Models, validation battery, and scorecard. Imported by the runner scripts."""
import warnings, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
try:
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
    warnings.simplefilter("ignore", IterationLimitWarning)
except Exception:
    pass

WINDOW, REFIT, VOL_Q = 500, 21, 0.80   # 500-day window, monthly refit, 80th-pct stress

# ---------- data helpers ----------
def load_returns(datadir, name):
    df = pd.read_csv(f"{datadir}/returns_{name}.csv", parse_dates=["date"]).set_index("date")
    return df["logret"].astype(float).dropna()

def make_features(r, lags=5):
    df = pd.DataFrame(index=r.index)
    for k in range(1, lags+1): df[f"ret_lag{k}"] = r.shift(k)
    df["absret_lag1"] = r.shift(1).abs()
    df["vol5"] = r.shift(1).rolling(5).std(); df["vol21"] = r.shift(1).rolling(21).std()
    df["y"] = r
    return df.dropna()

# ---------- forecasters (one-step-ahead, walk-forward) ----------
def hs_var(r, a, window=WINDOW, start=None):
    v = r.values; n = len(v); start = start or window; o = {}
    for t in range(start, n): o[r.index[t]] = np.quantile(v[t-window:t], a)
    return pd.Series(o, name="HistoricalSim")

def garch_var(r, a, window=WINDOW, start=None, refit=REFIT):
    v = r.values*100; n = len(v); start = start or window; o = {}
    from arch import arch_model
    res = None
    for i, t in enumerate(range(start, n)):
        if res is None or i % refit == 0:
            res = arch_model(v[t-window:t], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        f = res.forecast(horizon=1, reindex=False); sg = np.sqrt(f.variance.values[-1,0]); mu = res.params.get("mu",0)
        nu = res.params["nu"]; q = stats.t.ppf(a, nu)*np.sqrt((nu-2)/nu)
        o[r.index[t]] = (mu+sg*q)/100
    return pd.Series(o, name="GARCH-t")

def gbm_var(f, a, window=WINDOW, start=None, refit=REFIT, **kw):
    X = f.drop(columns="y").values; y = f["y"].values; idx = f.index
    n = len(f); start = start or window; o = {}; m = None
    p = dict(n_estimators=100, max_depth=2, learning_rate=0.05); p.update(kw)
    for i, t in enumerate(range(start, n)):
        if m is None or i % refit == 0:
            m = GradientBoostingRegressor(loss="quantile", alpha=a, subsample=0.9, random_state=0, **p).fit(X[t-window:t], y[t-window:t])
        o[idx[t]] = float(m.predict(X[t:t+1])[0])
    return pd.Series(o, name="GBM-Quantile")

def qreg_var(f, a, window=WINDOW, start=None, refit=REFIT):
    import statsmodels.api as sm
    X = sm.add_constant(f.drop(columns="y")).values; y = f["y"].values; idx = f.index
    n = len(f); start = start or window; o = {}; b = None
    for i, t in enumerate(range(start, n)):
        if b is None or i % refit == 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b = sm.QuantReg(y[t-window:t], X[t-window:t]).fit(q=a, max_iter=2000).params
            except Exception:
                b = np.zeros(X.shape[1]) if b is None else b
        o[idx[t]] = float(X[t] @ b)
    return pd.Series(o, name="LinearQuantile")

# ---------- validation battery ----------
def exceed(r, v):
    d = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna(); return (d["r"] < d["v"]).rename("hit")

def kupiec(h, a):
    x = int(h.sum()); n = len(h); pi = x/n if n else np.nan
    lr = np.nan if x in (0,n) else -2*((n-x)*np.log(1-a)+x*np.log(a)-(n-x)*np.log(1-pi)-x*np.log(pi))
    return {"n": n, "breaches": x, "obs_rate": pi, "p_value": 1-stats.chi2.cdf(lr,1) if np.isfinite(lr) else np.nan}

def christoffersen(h, a):
    hh = h.astype(int).values; n00=n01=n10=n11=0
    for i in range(1,len(hh)):
        p,c = hh[i-1],hh[i]; n00+=p==0 and c==0; n01+=p==0 and c==1; n10+=p==1 and c==0; n11+=p==1 and c==1
    n0,n1 = n00+n01,n10+n11; p01=n01/n0 if n0 else 0; p11=n11/n1 if n1 else 0; p=(n01+n11)/(n0+n1) if (n0+n1) else 0
    ll = lambda q,x,y: 0.0 if q<=0 or q>=1 else x*np.log(1-q)+y*np.log(q)
    lr_ind = -2*(ll(p,n00,n01)+ll(p,n10,n11)-ll(p01,n00,n01)-ll(p11,n10,n11))
    k = kupiec(h,a); x=k["breaches"]; nn=k["n"]; pi=x/nn if nn else np.nan
    uc = np.nan if x in (0,nn) else -2*((nn-x)*np.log(1-a)+x*np.log(a)-(nn-x)*np.log(1-pi)-x*np.log(pi))
    lr = (uc+lr_ind) if np.isfinite(uc) else np.nan
    return {"p_cc": 1-stats.chi2.cdf(lr,2) if np.isfinite(lr) else np.nan}

def vol_regime(r, q=VOL_Q, window=21):
    vol = r.rolling(window).std(); thr = vol.expanding(min_periods=252).quantile(q)
    return pd.Series(np.where(vol>thr,"stress","calm"), index=r.index)

def regime_breakdown(h, rg, a):
    d = pd.concat([h.rename("h"), rg.rename("g")], axis=1).dropna()
    return {str(k): kupiec(g["h"],a)["obs_rate"] for k,g in d.groupby("g")}

def pinball(r, v, a):
    d = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna(); e = d["r"]-d["v"]
    return np.maximum(a*e, (a-1)*e)

def pbo(perf, S=10):
    T,N = perf.shape
    if N < 2: return np.nan
    m = (T//S)*S; perf = perf[:m]; blocks = np.array_split(np.arange(m), S); lg = []
    for isc in itertools.combinations(range(S), S//2):
        isr = np.concatenate([blocks[b] for b in isc]); oos = np.concatenate([blocks[b] for b in range(S) if b not in isc])
        ns = int(np.argmax(perf[isr].mean(0))); w = stats.rankdata(perf[oos].mean(0))[ns]/(N+1)
        w = min(max(w,1e-6),1-1e-6); lg.append(np.log(w/(1-w)))
    return float(np.mean(np.array(lg) <= 0))

def conformal(r, v, a, calib=250):
    d = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
    rr, vv = d["r"].values, d["v"].values; n = len(d); adj = np.full(n, np.nan)
    for t in range(calib, n):
        e = rr[t-calib:t]-vv[t-calib:t]; adj[t] = vv[t]+np.quantile(e, a)
    mk = ~np.isnan(adj)
    return {"cov_before": float((rr[mk]<vv[mk]).mean()), "cov_after": float((rr[mk]<adj[mk]).mean())}

def scorecard(rows, a):
    df = pd.DataFrame(rows)
    f = lambda c,ok: df.__setitem__(c, np.where(ok,"PASS","FAIL"))
    f("Kupiec", df["kupiec_p"]>=0.05); f("Christoffersen", df["christ_p"]>=0.05)
    f("RegimeRobust", (df["stress_rate"]/a)<=2.0); f("OverfitControl", df["PBO"].fillna(0)<=0.50)
    f("Calibrated", (df["cov_before"]-a).abs()<=0.01)
    fl = ["Kupiec","Christoffersen","RegimeRobust","OverfitControl","Calibrated"]
    df["overall"] = np.where((df[fl]=="PASS").all(axis=1),"PASS","FAIL")
    return df
