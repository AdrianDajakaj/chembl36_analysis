"""Eksport wykresów z analysis.ipynb do app/assets/presentation/."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "analysis.ipynb"
OUT = ROOT / "app" / "assets" / "presentation"

# (indeks_komórki, indeks_png_w_outputach, nazwa_pliku)
EXPORTS = [
    (165, 0, "08_mlp_learning.png"),
    (165, 1, "08_mlp_pred_vs_actual.png"),
    (165, 2, "08_mlp_residuals.png"),
    (165, 3, "08_mlp_metrics.png"),
]


def png_from_cell(cell, png_idx: int) -> bytes:
    seen = 0
    for o in cell.get("outputs", []):
        if "image/png" not in o.get("data", {}):
            continue
        if seen == png_idx:
            raw = o["data"]["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            return base64.b64decode(raw)
        seen += 1
    raise IndexError(f"Brak PNG #{png_idx} w komórce")


def main() -> None:
    nb = json.loads(NB.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for cell_idx, png_idx, name in EXPORTS:
        data = png_from_cell(nb["cells"][cell_idx], png_idx)
        path = OUT / name
        path.write_bytes(data)
        print(f"{path} ({len(data)} B)")
    alias = OUT / "08_mlp_results.png"
    alias.write_bytes((OUT / "08_mlp_learning.png").read_bytes())
    print(f"{alias} (alias → 08_mlp_learning.png)")


if __name__ == "__main__":
    main()
