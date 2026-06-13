"""
Realna weryfikacja R^2 na Morgan FP (jak w sekcji 8 MLP) dla kandydujących targetów.

Dla każdego targetu:
  1. z checkpointu bierzemy zagregowane pary IC50 (1 wiersz/parę, pic50 = mediana)
  2. dociągamy canonical_smiles z bazy (compound_structures)
  3. liczymy Morgan FP (radius=2, 2048 bit) — identycznie jak notebook
  4. trenujemy szybki RandomForest (proxy dla MLP, ale na TYCH SAMYCH cechach co MLP)
     w dwóch splitach: random 80/20 oraz scaffold (Murcko) 80/20
  5. raportujemy R^2 / RMSE

RF na Morgan FP jest szybszy niż MLP, a dobrze przybliża osiągalny sygnał strukturalny.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

RDLogger.DisableLog("rdApp.*")
SEED = 42
DB_URL = "postgresql+psycopg2://admin:chembl_pass@localhost:5432/chembl_36"
PKL = "checkpoints/df_before_mlp.pkl"

CANDIDATES = {
    "CHEMBL1163125": "BRD4",
    "CHEMBL203": "EGFR",
    "CHEMBL206": "Estrogen receptor",
    "CHEMBL5113": "Orexin receptor 1",
    "CHEMBL5464": "RIPK1",
}

FPSIZE, FP_RADIUS = 2048, 2


def fetch_smiles(molregnos, engine, chunk=40000):
    molregnos = np.unique(molregnos.astype(np.int64))
    parts = []
    for i in range(0, len(molregnos), chunk):
        ids = ",".join(str(int(x)) for x in molregnos[i:i+chunk])
        q = f"SELECT molregno, canonical_smiles FROM compound_structures WHERE molregno IN ({ids})"
        parts.append(pd.read_sql(text(q), engine))
    return pd.concat(parts, ignore_index=True)


def fp_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FPSIZE), dtype=np.int8)


def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(m)
    except Exception:
        return None
    if core.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(core)


def scaffold_split(scaffs, frac_train=0.8, rng=None):
    rng = rng or np.random.default_rng(SEED)
    groups = {}
    for i, s in enumerate(scaffs):
        groups.setdefault(s, []).append(i)
    gl = list(groups.values())
    rng.shuffle(gl)
    n = len(scaffs)
    train, test = [], []
    for idxs in gl:
        if len(train) < frac_train * n:
            train += idxs
        else:
            test += idxs
    return np.array(train), np.array(test)


def rf_r2(X, y, tr, te):
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               max_features="sqrt", n_jobs=-1, random_state=SEED)
    rf.fit(X[tr], y[tr])
    p = rf.predict(X[te])
    return r2_score(y[te], p), root_mean_squared_error(y[te], p)


def main():
    df = pd.read_pickle(PKL)
    engine = create_engine(DB_URL)
    rows = []
    for tid, name in CANDIDATES.items():
        sub = df[(df["target_chembl_id"] == tid) & (df["standard_type"] == "IC50")][
            ["molregno", "pic50"]].copy()
        smap = fetch_smiles(sub["molregno"].to_numpy(), engine)
        sub = sub.merge(smap, on="molregno", how="inner").dropna(subset=["canonical_smiles"])
        feats, ys, scaffs = [], [], []
        for smi, pic in zip(sub["canonical_smiles"], sub["pic50"]):
            fp = fp_of(smi)
            sc = scaffold_of(smi)
            if fp is None or sc is None:
                continue
            feats.append(fp); ys.append(pic); scaffs.append(sc)
        X = np.vstack(feats); y = np.array(ys, dtype=float)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr_r, te_r = perm[:cut], perm[cut:]
        r2_rand, rmse_rand = rf_r2(X, y, tr_r, te_r)
        tr_s, te_s = scaffold_split(scaffs, 0.8, rng)
        r2_scaf, rmse_scaf = rf_r2(X, y, tr_s, te_s)
        rows.append({
            "target": tid, "name": name, "n": len(y),
            "n_scaffolds": len(set(scaffs)),
            "R2_random": round(r2_rand, 4), "RMSE_random": round(rmse_rand, 4),
            "R2_scaffold": round(r2_scaf, 4), "RMSE_scaffold": round(rmse_scaf, 4),
        })
        print(f"done {name:20s} n={len(y):5d}  R2_rand={r2_rand:.4f}  R2_scaf={r2_scaf:.4f}", flush=True)

    out = pd.DataFrame(rows).sort_values("R2_random", ascending=False)
    print("\n==== Morgan FP (2048) + RandomForest — realny R^2 ====")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
