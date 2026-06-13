"""Narzędzia dla LLM (function calling / tool use) oraz helpery dla UI.

Każde narzędzie jest odporne na błędny SMILES i zwraca zwięzły, czytelny wynik
(string) — taki, jaki LLM dobrze interpretuje. Helpery (`get_descriptors`,
`render_2d`) służą bezpośrednio UI (zwracają struktury danych / obraz).
"""
from __future__ import annotations

import io

import math

from langchain_core.tools import tool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from chem_model import predict_pic50, predict_pic50_uncertainty

RDLogger.DisableLog("rdApp.*")

# Referencyjny, silny inhibitor BRD4 do mapy podobieństwa (JQ1, CHEMBL ~ JQ1).
JQ1_SMILES = "CC(C)(C)OC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2c(C)c(C)sc2-n2c(C)nnc12"

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_JQ1_MOL = Chem.MolFromSmiles(JQ1_SMILES)
_JQ1_FP = _MORGAN_GEN.GetFingerprint(_JQ1_MOL)


def _potency_category(pic50: float) -> str:
    if pic50 >= 7:
        return "silny inhibitor"
    if pic50 >= 5:
        return "umiarkowany inhibitor"
    return "słaby / nieaktywny"


def tanimoto_to_jq1(smiles: str) -> float | None:
    """Podobieństwo Tanimoto (Morgan r=2) do referencyjnego inhibitora BRD4 (JQ1)."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return round(DataStructs.TanimotoSimilarity(_MORGAN_GEN.GetFingerprint(mol), _JQ1_FP), 3)


# ----------------------------- Helpery dla UI -----------------------------
def get_descriptors(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return {
        "Masa cząsteczkowa (MW)": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "Akceptory H (HBA)": Lipinski.NumHAcceptors(mol),
        "Donory H (HBD)": Lipinski.NumHDonors(mol),
        "Wiązania rotacyjne": Lipinski.NumRotatableBonds(mol),
        "Pierścienie aromatyczne": Lipinski.NumAromaticRings(mol),
        "Ciężkie atomy": mol.GetNumHeavyAtoms(),
        "Naruszenia reguły Lipinskiego (Ro5)": _ro5_violations(mol),
    }


def _ro5_violations(mol) -> int:
    v = 0
    if Descriptors.MolWt(mol) > 500:
        v += 1
    if Descriptors.MolLogP(mol) > 5:
        v += 1
    if Lipinski.NumHDonors(mol) > 5:
        v += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        v += 1
    return v


def render_2d(smiles: str, size: tuple[int, int] = (420, 320)):
    """Zwraca obiekt PIL.Image ze strukturą 2D albo None dla błędnego SMILES."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def render_2d_png_bytes(smiles: str, size: tuple[int, int] = (420, 320)) -> bytes | None:
    img = render_2d(smiles, size)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_2d_zoom_svg(smiles: str, size: tuple[int, int] = (560, 460)) -> str | None:
    """Powiększona struktura 2D (SVG) z numeracją atomów — „zoom" struktury."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.bondLineWidth = 2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _embed_3d(smiles: str):
    """Zwraca cząsteczkę z wodorami i współrzędnymi 3D (ETKDG+MMFF), albo None."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return mol


def mol_block_3d(smiles: str) -> str | None:
    """Generuje współrzędne 3D (ETKDG + MMFF) i zwraca MOL block, albo None."""
    mol = _embed_3d(smiles)
    return Chem.MolToMolBlock(mol) if mol is not None else None


def _charge_to_hex(c: float, vmax: float = 0.4) -> str:
    """Diwergentna mapa: ładunek ujemny → czerwony, dodatni → niebieski, ~0 → biały."""
    if c is None or math.isnan(c):
        c = 0.0
    t = max(-1.0, min(1.0, c / vmax))
    if t >= 0:  # dodatni → niebieski
        r = g = int(255 * (1 - t)); b = 255
    else:       # ujemny → czerwony
        t = -t
        r = 255; g = b = int(255 * (1 - t))
    return f"0x{r:02x}{g:02x}{b:02x}"


def mol_3d_html(smiles: str, width: int = 460, height: int = 420,
                color_by: str = "element") -> str | None:
    """Interaktywny widok 3D (py3Dmol / 3Dmol.js).

    color_by="element" — standardowe kolory atomów;
    color_by="charge"  — kolorowanie wg ładunków cząstkowych Gasteigera.
    """
    mol = _embed_3d(smiles)
    if mol is None:
        return None
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(Chem.MolToMolBlock(mol), "mol")

    if color_by == "charge":
        AllChem.ComputeGasteigerCharges(mol)
        view.setStyle({}, {"stick": {"radius": 0.14}, "sphere": {"scale": 0.22}})
        for i, atom in enumerate(mol.GetAtoms()):
            try:
                q = atom.GetDoubleProp("_GasteigerCharge")
            except Exception:
                q = 0.0
            col = _charge_to_hex(q)
            view.setStyle({"serial": i},
                          {"stick": {"radius": 0.14, "color": col},
                           "sphere": {"scale": 0.22, "color": col}})
    else:
        view.setStyle({}, {"stick": {"radius": 0.16}, "sphere": {"scale": 0.22}})

    view.setBackgroundColor("0x0e1117")
    view.zoomTo()
    view.spin(True)
    return view._make_html()


def similarity_map_svg(smiles: str, ref_smiles: str = JQ1_SMILES,
                       size: tuple[int, int] = (440, 380)) -> str | None:
    """Mapa podobieństwa (Morgan) badanej cząsteczki względem referencji (JQ1).

    Zielone obszary = fragmenty zwiększające podobieństwo do referencyjnego
    inhibitora BRD4, różowe = zmniejszające. Zwraca SVG albo None.
    """
    probe = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    ref = Chem.MolFromSmiles(ref_smiles)
    if probe is None or ref is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        fpf = lambda m, idx: SimilarityMaps.GetMorganFingerprint(m, idx, radius=2, fpType="bv")  # noqa: E731
        SimilarityMaps.GetSimilarityMapForFingerprint(ref, probe, fpf, drawer)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ----------------------------- Narzędzia LLM -----------------------------
@tool
def predict_brd4_pic50(smiles: str) -> str:
    """Przewiduje aktywność (pIC50 oraz IC50 w nM) cząsteczki wobec białka BRD4
    na podstawie jej SMILES, wraz z oszacowaniem niepewności (MC Dropout),
    oceną drug-likeness (reguła Lipinskiego) i podobieństwem do referencyjnego
    inhibitora JQ1. Używaj zawsze, gdy użytkownik pyta o aktywność biologiczną,
    siłę inhibicji, pIC50 lub IC50 podanej cząsteczki."""
    r = predict_pic50_uncertainty(smiles, n_samples=25)
    if not r.get("ok"):
        return f"BŁĄD: {r.get('error')}"

    mol = Chem.MolFromSmiles(r["canonical_smiles"])
    ro5 = _ro5_violations(mol)
    tani = tanimoto_to_jq1(r["canonical_smiles"])
    cat = _potency_category(r["pIC50_pred"])

    lines = [
        f"Predykcja dla {r['target']} (model GIN, ChEMBL 36):",
        f"- pIC50 = {r['pIC50_pred']}  →  kategoria: {cat}",
        f"- IC50 ≈ {r['IC50_nM_pred']} nM",
    ]
    if r.get("mc_std") is not None:
        fold = round(10 ** r["mc_std"], 1)
        lines.append(
            f"- Niepewność modelu (MC Dropout): ±{r['mc_std']} pIC50 "
            f"(≈ ×{fold} w skali IC50), pewność: {r.get('confidence')}"
        )
    lines += [
        f"- SMILES (kanoniczny): {r['canonical_smiles']}",
        f"- MW = {r['MW']} g/mol, LogP = {r['LogP']}",
        f"- Reguła Lipinskiego (Ro5): {ro5} naruszeń (0 = dobry profil drug-like)",
        f"- Podobieństwo do JQ1 (Tanimoto, Morgan r=2): {tani} "
        f"({'wysokie' if (tani or 0) >= 0.4 else ('umiarkowane' if (tani or 0) >= 0.2 else 'niskie')})",
    ]
    return "\n".join(lines)


@tool
def molecule_descriptors(smiles: str) -> str:
    """Liczy deskryptory fizykochemiczne cząsteczki z RDKit (masa, LogP, TPSA,
    donory/akceptory wodoru, wiązania rotacyjne, naruszenia reguły Lipinskiego).
    Używaj, gdy użytkownik pyta o właściwości fizykochemiczne / drug-likeness."""
    d = get_descriptors(smiles)
    if d is None:
        return f"BŁĄD: niepoprawny SMILES: '{smiles}'"
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


@tool
def molecular_weight(smiles: str) -> str:
    """Zwraca masę cząsteczkową (g/mol) dla podanego SMILES."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return f"BŁĄD: niepoprawny SMILES: '{smiles}'"
    return f"Masa cząsteczkowa = {round(Descriptors.MolWt(mol), 2)} g/mol"


@tool
def logp(smiles: str) -> str:
    """Zwraca lipofilowość LogP (Crippen) dla podanego SMILES."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
    if mol is None:
        return f"BŁĄD: niepoprawny SMILES: '{smiles}'"
    return f"LogP = {round(Descriptors.MolLogP(mol), 2)}"


@tool
def similarity_to_jq1(smiles: str) -> str:
    """Liczy podobieństwo strukturalne (Tanimoto, fingerprint Morgana r=2) podanej
    cząsteczki do JQ1 — wzorcowego, silnego inhibitora BRD4. Używaj, gdy chcesz
    ocenić, jak bardzo związek przypomina znany aktywny chemotyp."""
    t = tanimoto_to_jq1(smiles)
    if t is None:
        return f"BŁĄD: niepoprawny SMILES: '{smiles}'"
    level = "wysokie" if t >= 0.4 else ("umiarkowane" if t >= 0.2 else "niskie")
    return (f"Podobieństwo Tanimoto do JQ1 = {t} ({level}). "
            f"Wyższe podobieństwo do znanego inhibitora zwykle sprzyja aktywności wobec BRD4.")


ALL_TOOLS = [predict_brd4_pic50, molecule_descriptors, molecular_weight, logp, similarity_to_jq1]
