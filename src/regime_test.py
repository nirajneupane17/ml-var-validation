"""Formal logistic test of the regime effect (odds ratio + p-value)."""
import os, numpy as np, pandas as pd, statsmodels.api as sm
import pipeline as P
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
A = 0.05; r = P.load_returns(DATA, "sp500"); f = P.make_features(r)
fs = P.WINDOW; rs = P.WINDOW + (len(r)-len(f)); rg = P.vol_regime(r)
fc = {"HistoricalSim": P.hs_var(r,A,P.WINDOW,rs), "GBM-Quantile": P.gbm_var(f,A,P.WINDOW,fs)}
for nm, v in fc.items():
    d = pd.concat([r.rename("r"), v.rename("v"), rg.rename("s")], axis=1).dropna()
    hit = (d["r"]<d["v"]).astype(int); s = (d["s"]=="stress").astype(int)
    res = sm.Logit(hit.values, sm.add_constant(s.values)).fit(disp=0, cov_type="HC1")
    print(f"{nm}: odds ratio={np.exp(res.params[1]):.2f}  p={res.pvalues[1]:.2e}  "
          f"(calm {hit[s==0].mean():.3f} vs stress {hit[s==1].mean():.3f})")
