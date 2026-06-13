"""
Wybór optymalnego pojedynczego targetu pod regresję pIC50 (maksymalizacja R^2).

Dane: checkpoints/df_before_mlp.pkl (zagregowany df, 1 wiersz na parę związek-target).
SMILES nie ma w pickle (dociągane z bazy), więc jako proxy "uczalności" pIC50 ze
struktury używamy cech fizykochemicznych (te same, na których stał baseline tabelaryczny).
Target dobrze przewidywalny z fizchem zwykle ma silny SAR i jest też dobry dla MLP/GNN.

Dla top targetów (wg liczby pomiarów IC50) trenujemy szybki RandomForest na losowym
splicie 80/20 i raportujemy R^2/RMSE. Dodatkowo liczymy statystyki sygnału (std/IQR pIC50).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

SEED = 42
PKL = "checkpoints/df_before_mlp.pkl"
MIN_ROWS = 800          # minimalna liczba związków, by target był sensownym kandydatem
TOP_BY_COUNT = 60       # ile najliczniejszych targetów testujemy
RF_TREES = 300

FEATURES = [
    "mw_freebase", "alogp", "hba", "hbd", "psa", "rtb",
    "num_ro5_violations", "aromatic_rings", "qed_weighted",
    "np_likeness_score", "full_mwt",
    "psa_per_mw", "hba_per_mw", "alogp_per_mw", "aromatic_fraction",
    "rtb_per_mw", "hbd_hba_ratio",
    "mw_bin_ord", "alogp_bin_ord", "log_psa", "log_rtb", "log_hbd",
]

def eval_target(sub: pd.DataFrame) -> dict:
    sub = sub.dropna(subset=["pic50"]).copy()
    feats = [f for f in FEATURES if f in sub.columns]
    X = sub[feats].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    y = sub["pic50"].astype(float).values
    nunique_feat = [c for c in X.columns if X[c].nunique() > 1]
    X = X[nunique_feat]
    if len(sub) < 50 or X.shape[1] == 0:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=SEED,
    )
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return {
        "r2": r2_score(yte, pred),
        "rmse": root_mean_squared_error(yte, pred),
        "n_test": len(yte),
    }


def analyze(df: pd.DataFrame, label: str) -> pd.DataFrame:
    counts = df["target_chembl_id"].value_counts()
    cand = counts[counts >= MIN_ROWS].head(TOP_BY_COUNT).index.tolist()
    rows = []
    for tid in cand:
        sub = df[df["target_chembl_id"] == tid]
        name = sub["target_name"].iloc[0]
        res = eval_target(sub)
        if res is None:
            continue
        rows.append({
            "target_chembl_id": tid,
            "target_name": (name[:38] if isinstance(name, str) else name),
            "n": len(sub),
            "pic50_std": round(sub["pic50"].std(), 3),
            "pic50_iqr": round(sub["pic50"].quantile(0.75) - sub["pic50"].quantile(0.25), 3),
            "R2": round(res["r2"], 4),
            "RMSE": round(res["rmse"], 4),
        })
    out = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print(f"\n================ {label} ================")
    print(f"Kandydaci (n>={MIN_ROWS}): {len(out)}   |  proxy: RandomForest na cechach fizchem (random 80/20)")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(out.to_string(index=False))
    return out


def main():
    print("Wczytywanie pickle...")
    df = pd.read_pickle(PKL)
    print(f"df: {df.shape}")

    # 1) IC50 only (spójne z obecnym MLP/GNN, które filtrowały IC50)
    ic50 = df[df["standard_type"] == "IC50"]
    out_ic50 = analyze(ic50, "IC50 only")

    # 2) IC50 + Ki (więcej danych na target; is_ki jako cecha)
    out_all = analyze(df, "IC50 + Ki")

    print("\n================ TOP 10 REKOMENDACJI (IC50 only, wg R2) ================")
    print(out_ic50.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
