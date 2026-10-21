"""Publication figures -> figures/*.pdf,*.png"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pipeline as P
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE,"data"); OUT = os.path.join(HERE,"figures"); os.makedirs(OUT, exist_ok=True)
NAVY="#2f3b52"; RUST="#9c6b3f"
plt.rcParams.update({"figure.dpi":120,"font.size":10,"font.family":"serif","axes.grid":True,
    "grid.alpha":.25,"axes.spines.top":False,"axes.spines.right":False})
A=0.05; r=P.load_returns(DATA,"sp500"); f=P.make_features(r); fs=P.WINDOW; rs=P.WINDOW+(len(r)-len(f))
fc={"Hist. Sim.":P.hs_var(r,A,P.WINDOW,rs),"GARCH-$t$":P.garch_var(r,A,P.WINDOW,rs),
    "GBM Quantile":P.gbm_var(f,A,P.WINDOW,fs),"Lin. Quantile":P.qreg_var(f,A,P.WINDOW,fs)}
rg=P.vol_regime(r); names=list(fc); x=np.arange(len(names))
def sf(fig,nm): fig.tight_layout(); fig.savefig(f"{OUT}/{nm}.pdf"); fig.savefig(f"{OUT}/{nm}.png",dpi=200); plt.close(fig)
calm=[P.regime_breakdown(P.exceed(r,fc[n]),rg,A).get("calm",np.nan) for n in names]
stress=[P.regime_breakdown(P.exceed(r,fc[n]),rg,A).get("stress",np.nan) for n in names]
fig,ax=plt.subplots(figsize=(7,3.8)); ax.bar(x-0.2,calm,0.38,label="Calm",color="#7f8ca3"); ax.bar(x+0.2,stress,0.38,label="Stress",color=NAVY)
ax.axhline(A,ls="--",color=RUST,lw=1.2,label="Target 0.05"); ax.set_xticks(x); ax.set_xticklabels(names,fontsize=9)
ax.set_ylabel("Breach rate"); ax.set_title("VaR breach rate by regime: S&P 500, 95% VaR"); ax.legend(frameon=False,fontsize=8); sf(fig,"fig1_breach_by_regime")
fig,ax=plt.subplots(figsize=(9.5,3.6)); cols=[NAVY,RUST,"#4a7c59","#8a8a8a"]
for i,n in enumerate(names):
    h=P.exceed(r,fc[n]).astype(float); ax.plot(h.index,h.rolling(250).mean(),lw=.9,label=n,color=cols[i])
ax.axhline(A,ls="--",color="k",lw=1); ax.set_title("Rolling 250-day breach rate vs. 5% target: S&P 500")
ax.set_ylabel("Rolling breach rate"); ax.legend(ncol=4,fontsize=8,frameon=False); sf(fig,"fig2_rolling_coverage")
g=pd.read_csv(f"{HERE}/results/grid_ALL.csv") if os.path.exists(f"{HERE}/results/grid_ALL.csv") else None
if g is not None:
    assets=["sp500","nasdaq","btc","tlt_bonds","eurusd","usdjpy"]; labels=["S&P 500","Nasdaq","Bitcoin","Treasuries","EUR/USD","USD/JPY"]
    l95=[g[(g.asset==a)&(g.alpha==0.05)&(g.model=="GBM-Quantile")]["Lambda"].values[0] for a in assets]
    l99=[g[(g.asset==a)&(g.alpha==0.01)&(g.model=="GBM-Quantile")]["Lambda"].values[0] for a in assets]
    xa=np.arange(len(assets)); fig,ax=plt.subplots(figsize=(7.5,3.8))
    ax.bar(xa-0.2,l95,0.38,label="95% VaR",color="#7f8ca3"); ax.bar(xa+0.2,l99,0.38,label="99% VaR",color=NAVY)
    ax.axhline(2.0,ls="--",color=RUST,lw=1.2,label="Threshold $\\Lambda$=2"); ax.set_xticks(xa); ax.set_xticklabels(labels,fontsize=8,rotation=10)
    ax.set_ylabel("Stress amplification $\\Lambda$"); ax.set_title("Stress amplification of the ML model by asset and VaR level"); ax.legend(frameon=False,fontsize=8); sf(fig,"fig3_lambda_by_asset")
print("figures ->", OUT)
