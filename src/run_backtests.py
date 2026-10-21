"""Full grid: 6 assets x {0.05, 0.01}, monthly refit -> results/*.csv"""
import os, numpy as np, pandas as pd
import pipeline as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
ASSETS = ["sp500","nasdaq","btc","tlt_bonds","eurusd","usdjpy"]; ALPHAS = [0.05, 0.01]
PBO_CFGS = [dict(n_estimators=50,max_depth=2,learning_rate=.05), dict(n_estimators=100,max_depth=2,learning_rate=.05),
            dict(n_estimators=200,max_depth=3,learning_rate=.05), dict(n_estimators=300,max_depth=3,learning_rate=.10),
            dict(n_estimators=400,max_depth=4,learning_rate=.10)]

allrows = []
for name in ASSETS:
    r = P.load_returns(DATA, name); f = P.make_features(r)
    fs = P.WINDOW; rs = P.WINDOW + (len(r)-len(f)); rg = P.vol_regime(r)
    for a in ALPHAS:
        fc = {"HistoricalSim": P.hs_var(r,a,P.WINDOW,rs), "GARCH-t": P.garch_var(r,a,P.WINDOW,rs),
              "GBM-Quantile": P.gbm_var(f,a,P.WINDOW,fs), "LinearQuantile": P.qreg_var(f,a,P.WINDOW,fs)}
        pbo_ml = np.nan
        if name == "sp500":
            perf = pd.concat([-P.pinball(r, P.gbm_var(f,a,P.WINDOW,fs,refit=63,**c), a) for c in PBO_CFGS], axis=1).dropna().values
            pbo_ml = P.pbo(perf)
        rows = []
        for m, v in fc.items():
            h = P.exceed(r,v); k = P.kupiec(h,a); c = P.christoffersen(h,a); rb = P.regime_breakdown(h,rg,a); cf = P.conformal(r,v,a)
            rows.append({"asset":name,"alpha":a,"model":m,"breaches":k["breaches"],"obs_rate":round(k["obs_rate"],4),
                "kupiec_p":round(k["p_value"],4) if k["p_value"]==k["p_value"] else np.nan,
                "christ_p":round(c["p_cc"],4) if c["p_cc"]==c["p_cc"] else np.nan,
                "calm_rate":round(rb.get("calm",np.nan),4),"stress_rate":round(rb.get("stress",np.nan),4),
                "Lambda":round(rb.get("stress",np.nan)/a,2),
                "PBO":round(pbo_ml,3) if (m=="GBM-Quantile" and pbo_ml==pbo_ml) else np.nan,
                "cov_before":round(cf["cov_before"],4),"cov_after":round(cf["cov_after"],4)})
        P.scorecard(rows,a).to_csv(f"{OUT}/scorecard_{name}_a{int(a*100)}.csv", index=False)
        allrows += rows
        pd.DataFrame(allrows).to_csv(f"{OUT}/grid_ALL.csv", index=False)
        print(f"done {name} a={a}")
print("grid complete ->", OUT)
