"""Download public market data -> data/*.csv.  Requires internet."""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
START = "2000-01-01"

ASSETS = {
    "sp500":     {"source": "yahoo", "ticker": "^GSPC",   "stooq": "^SPX"},
    "nasdaq":    {"source": "yahoo", "ticker": "^IXIC",   "stooq": "^NDQ"},
    "btc":       {"source": "yahoo", "ticker": "BTC-USD", "stooq": None},
    "tlt_bonds": {"source": "yahoo", "ticker": "TLT",     "stooq": "TLT.US"},
    "eurusd":    {"source": "fred",  "ticker": "DEXUSEU"},
    "usdjpy":    {"source": "fred",  "ticker": "DEXJPUS"},
}
STRESS = {"vix": "VIXCLS", "nfci": "NFCI", "stlfsi": "STLFSI4"}
CRISES = [("2007-07-01","2009-06-30"),("2011-07-01","2011-12-31"),
          ("2015-08-01","2016-02-29"),("2018-10-01","2018-12-31"),
          ("2020-02-15","2020-06-30"),("2022-01-01","2022-12-31"),
          ("2023-03-01","2023-05-31")]

def yahoo(ticker, stooq=None):
    import yfinance as yf
    try:
        s = yf.download(ticker, start=START, auto_adjust=True, progress=False)["Close"]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        s = s.dropna()
        if len(s) > 100: return s
    except Exception as e:
        print(f"  yahoo {ticker}: {e}")
    if stooq:
        from pandas_datareader import data as pdr
        return pdr.DataReader(stooq, "stooq", START).sort_index()["Close"].dropna()
    raise RuntimeError(f"cannot download {ticker}")

def fred(series):
    from pandas_datareader import data as pdr
    return pdr.DataReader(series, "fred", START).iloc[:, 0].dropna()

def logret(px):
    r = np.log(px.astype(float)).diff().dropna(); r.index = pd.to_datetime(r.index); return r

returns = {}
for name, cfg in ASSETS.items():
    print("downloading", name)
    px = yahoo(cfg["ticker"], cfg.get("stooq")) if cfg["source"]=="yahoo" else fred(cfg["ticker"])
    r = logret(px); returns[name] = r
    pd.DataFrame({"date": r.index, "logret": r.values}).to_csv(f"{OUT}/returns_{name}.csv", index=False)

for name, series in STRESS.items():
    try:
        s = fred(series)
        pd.DataFrame({"date": s.index, name: s.values}).to_csv(f"{OUT}/stress_{name}.csv", index=False)
    except Exception as e:
        print("  skip", name, e)

r = returns["sp500"]
cal = pd.Series("calm", index=r.index)
for s, e in CRISES: cal.loc[(r.index>=s)&(r.index<=e)] = "stress"
vol = r.rolling(21).std(); thr = vol.expanding(min_periods=252).quantile(0.80)
vr = pd.Series(np.where(vol>thr,"stress","calm"), index=r.index)
pd.DataFrame({"date": r.index, "regime": cal.values}).to_csv(f"{OUT}/regime_calendar_sp500.csv", index=False)
pd.DataFrame({"date": r.index, "regime": vr.values}).to_csv(f"{OUT}/regime_vol_sp500.csv", index=False)
print("done ->", OUT)
