"""Warstwa modelu dla aplikacji: featuryzacja RDKit + GIN + predykcja pIC50 (BRD4).

Kod featuryzacji i architektura modelu są **wierną kopią** z analysis.ipynb
(sekcje 9.2-9.3), aby wczytany checkpoint `checkpoints/gin_brd4.pt` pasował 1:1.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "gin_brd4.pt"

# Metryki modelu GIN (BRD4) z analysis.ipynb — zbiory testowe (held-out).
# scaffold = podział wg rdzeni Murcko (trudniejszy, ocenia generalizację na NOWE chemotypy).
MODEL_METRICS = {
    "scaffold": {"R2": 0.640, "RMSE": 0.739, "MAE": 0.549, "fold": 5.5, "n_test": 769},
    "random": {"R2": 0.736, "RMSE": 0.603, "MAE": 0.462, "fold": 4.0},
}

# --- Słowniki kategorii (identyczne jak w notebooku) ---
ATOM_LIST = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _onehot(val, choices: list) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(val) if val in choices else -1] = 1.0
    return vec


def atom_features(atom) -> list[float]:
    return (
        _onehot(atom.GetAtomicNum(), ATOM_LIST)
        + _onehot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        + _onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + _onehot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetFormalCharge()), float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond) -> list[float]:
    return _onehot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


_probe = Chem.MolFromSmiles("CC")
ATOM_FEATURES_DIM = len(atom_features(_probe.GetAtomWithIdx(0)))
BOND_FEATURES_DIM = len(bond_features(_probe.GetBondWithIdx(0)))


def mol_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    xs = [atom_features(a) for a in mol.GetAtoms()]
    if len(xs) == 0:
        return None
    x = torch.tensor(xs, dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURES_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MoleculeGIN(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=bond_dim))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.do = nn.Dropout(dropout)  # moduł (bez parametrów) — pozwala na MC Dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.atom_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(x, edge_index, edge_attr)))
            h = self.do(h)
            x = x + h
        x = global_mean_pool(x, batch)
        return self.head(x)


@lru_cache(maxsize=1)
def load_model(ckpt_path: str | None = None):
    """Wczytuje model produkcyjny (cache'owany). Zwraca (model, meta)."""
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not path.is_file():
        raise FileNotFoundError(
            f"Brak checkpointu modelu: {path}. Uruchom sekcję 9.20 w analysis.ipynb."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MoleculeGIN(
        atom_dim=ckpt["atom_dim"],
        bond_dim=ckpt["bond_dim"],
        hidden=ckpt["hidden"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("target", {"name": "BRD4", "chembl_id": "CHEMBL1163125"})


def predict_pic50(smiles: str, ckpt_path: str | None = None) -> dict:
    """Predykcja pIC50 wobec BRD4. Bezpieczna na błędny input (nie rzuca wyjątku)."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {"ok": False, "error": "Pusty input — podaj łańcuch SMILES."}

    smi = smiles.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"ok": False, "error": f"Niepoprawny SMILES: '{smi}'"}

    graph = mol_to_graph(smi)
    if graph is None:
        return {"ok": False, "error": "Nie udało się zbudować grafu molekularnego."}

    model, target = load_model(ckpt_path)
    batch = next(iter(PyGDataLoader([graph], batch_size=1)))
    with torch.inference_mode():
        pic50 = float(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).reshape(-1)[0])
    ic50_nm = float(10.0 ** (9.0 - pic50))

    return {
        "ok": True,
        "target": f"{target['name']} ({target['chembl_id']})",
        "input_smiles": smi,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "pIC50_pred": round(pic50, 3),
        "IC50_nM_pred": round(ic50_nm, 2),
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
    }


def _set_mc_dropout(model, on: bool) -> None:
    """Włącza/wyłącza WYŁĄCZNIE warstwy Dropout (BatchNorm zostaje w eval)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(on)


def predict_pic50_uncertainty(smiles: str, n_samples: int = 40, ckpt_path: str | None = None) -> dict:
    """Predykcja pIC50 z oszacowaniem niepewności metodą **MC Dropout**.

    Uruchamia model n razy z aktywnym dropoutem (BatchNorm w eval) i zwraca
    rozkład predykcji: średnia, odchylenie std oraz próbki. Wąski rozkład =
    wysoka pewność modelu, szeroki = niska.
    """
    base = predict_pic50(smiles, ckpt_path)
    if not base.get("ok") or n_samples <= 0:
        return base

    model, _ = load_model(ckpt_path)
    graph = mol_to_graph(smiles.strip())
    batch = next(iter(PyGDataLoader([graph], batch_size=1)))

    model.eval()
    _set_mc_dropout(model, True)
    samples = []
    with torch.inference_mode():
        for _ in range(n_samples):
            samples.append(
                float(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).reshape(-1)[0])
            )
    _set_mc_dropout(model, False)  # przywróć pełny eval

    arr = np.asarray(samples, dtype=float)
    std = float(arr.std())
    base.update({
        "mc_mean": round(float(arr.mean()), 3),
        "mc_std": round(std, 3),
        "mc_samples": samples,
        "confidence": "wysoka" if std < 0.3 else ("średnia" if std < 0.6 else "niska"),
    })
    return base
