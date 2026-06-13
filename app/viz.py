"""Efektowne, interaktywne wizualizacje (Plotly) dla aplikacji BRD4.

Funkcje przyjmują `preds` = lista krotek (smiles, pred_dict), gdzie pred_dict
pochodzi z predict_pic50 (zawiera m.in. pIC50_pred, IC50_nM_pred, canonical_smiles).
Style w ciemnym motywie, z przezroczystym tłem (wtapia się w UI Streamlit).
"""
from __future__ import annotations

import plotly.graph_objects as go
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

# Paleta stref siły inhibitora
C_WEAK = "#e74c3c"      # < 5
C_MODERATE = "#f39c12"  # 5–7
C_STRONG = "#1abc9c"    # > 7
PALETTE = ["#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#f1c40f", "#2ecc71"]

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=30, t=70, b=40),
    font=dict(size=13),
)


def _zone(pic50: float) -> str:
    return C_STRONG if pic50 >= 7 else (C_MODERATE if pic50 >= 5 else C_WEAK)


def _short(smiles: str, n: int = 22) -> str:
    return smiles[:n] + "…" if len(smiles) > n else smiles


def _mol_props(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    return {
        "MW": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "TPSA": Descriptors.TPSA(m),
        "HBA": Lipinski.NumHAcceptors(m),
        "HBD": Lipinski.NumHDonors(m),
        "RotB": Lipinski.NumRotatableBonds(m),
    }


def potency_gauge(pred: dict) -> go.Figure:
    """Półokrągły gauge przewidzianego pIC50 z kolorowymi strefami siły.

    Jeśli pred zawiera `mc_std` (MC Dropout), nanosi szare pasmo niepewności
    ±1σ wokół wartości oraz informację o pewności w podtytule.
    """
    v = pred["pIC50_pred"]
    std = pred.get("mc_std")
    sub = f"IC50 ≈ {pred['IC50_nM_pred']:.0f} nM"
    steps = [
        {"range": [0, 5], "color": "rgba(231,76,60,0.25)"},
        {"range": [5, 7], "color": "rgba(243,156,18,0.25)"},
        {"range": [7, 10], "color": "rgba(26,188,156,0.25)"},
    ]
    if std is not None:
        lo, hi = max(0, v - std), min(10, v + std)
        steps.append({"range": [lo, hi], "color": "rgba(236,240,241,0.45)"})
        sub += f" &nbsp;|&nbsp; ±{std:.2f} σ (pewność: {pred.get('confidence', '—')})"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=v,
        number={"suffix": " pIC50", "font": {"size": 34}},
        delta={"reference": 5.0, "increasing": {"color": C_STRONG}, "decreasing": {"color": C_WEAK}},
        title={"text": f"Aktywność wobec BRD4<br><span style='font-size:0.8em'>{sub}</span>"},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": _zone(v), "thickness": 0.3},
            "steps": steps,
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": v},
        },
    ))
    fig.update_layout(height=330, **_LAYOUT)
    return fig


def confidence_plot(pred: dict) -> go.Figure:
    """Rozkład predykcji z MC Dropout — „pewność" modelu (im węższy, tym pewniej)."""
    samples = pred.get("mc_samples") or []
    if not samples:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych MC Dropout", showarrow=False)
        fig.update_layout(height=360, **_LAYOUT)
        return fig
    mean = pred.get("mc_mean", sum(samples) / len(samples))
    std = pred.get("mc_std", 0.0)
    color = _zone(mean)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=18, histnorm="probability density",
        marker=dict(color=color, opacity=0.55, line=dict(color="rgba(255,255,255,0.3)", width=1)),
        name="rozkład predykcji",
    ))
    fig.add_trace(go.Violin(
        x=samples, orientation="h", side="positive", width=0.0,
        line_color=color, fillcolor="rgba(0,0,0,0)", points=False, hoverinfo="skip",
        showlegend=False, yaxis="y2",
    ))
    fig.add_vrect(x0=mean - std, x1=mean + std, fillcolor="rgba(236,240,241,0.12)", line_width=0)
    fig.add_vline(x=mean, line=dict(color="white", width=2, dash="solid"),
                  annotation_text=f"μ={mean:.2f}", annotation_position="top")
    fig.update_layout(
        title=f"Pewność modelu (MC Dropout, n={len(samples)}) — σ={std:.2f} → pewność: {pred.get('confidence', '—')}",
        xaxis_title="przewidziane pIC50", yaxis_title="gęstość",
        yaxis2=dict(overlaying="y", showticklabels=False, range=[0, 1]),
        bargap=0.05, height=360, showlegend=False, **_LAYOUT,
    )
    return fig


def pic50_bar(preds: list[tuple[str, dict]]) -> go.Figure:
    """Poziomy słupkowy pIC50 z zacienionymi strefami siły i hoverem."""
    labels = [_short(pr["canonical_smiles"]) for _, pr in preds]
    vals = [pr["pIC50_pred"] for _, pr in preds]
    colors = [_zone(v) for v in vals]
    hover = [
        f"<b>{_short(pr['canonical_smiles'], 40)}</b><br>pIC50 = {pr['pIC50_pred']:.2f}"
        f"<br>IC50 ≈ {pr['IC50_nM_pred']:.0f} nM" for _, pr in preds
    ]
    errs = [pr.get("mc_std") for _, pr in preds]
    error_x = None
    if any(e is not None for e in errs):
        error_x = dict(type="data", array=[e or 0 for e in errs], visible=True,
                       color="rgba(255,255,255,0.7)", thickness=1.5, width=4)
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.4)", width=1)),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        hovertext=hover, hoverinfo="text", error_x=error_x,
    ))
    xmax = max(8.0, max(vals) + 1.2)
    fig.add_vrect(x0=0, x1=5, fillcolor=C_WEAK, opacity=0.08, line_width=0)
    fig.add_vrect(x0=5, x1=7, fillcolor=C_MODERATE, opacity=0.08, line_width=0)
    fig.add_vrect(x0=7, x1=xmax, fillcolor=C_STRONG, opacity=0.08, line_width=0)
    title = "Przewidziana aktywność (pIC50) — strefy: słaby / umiarkowany / silny"
    if error_x is not None:
        title += "<br><span style='font-size:0.8em'>wąsy = niepewność ±1σ (MC Dropout)</span>"
    fig.update_layout(
        title=title,
        xaxis_title="pIC50", xaxis_range=[0, xmax],
        height=120 * len(preds) + 180, **_LAYOUT,
    )
    return fig


def descriptor_radar(preds: list[tuple[str, dict]]) -> go.Figure:
    """Radar znormalizowanego profilu fizykochemicznego (porównanie cząsteczek)."""
    cats = ["MW", "LogP", "TPSA", "HBA", "HBD", "RotB"]
    # zakresy referencyjne do normalizacji 0-1
    norm = {
        "MW": lambda x: min(x / 600, 1),
        "LogP": lambda x: min(max((x + 2) / 9, 0), 1),
        "TPSA": lambda x: min(x / 150, 1),
        "HBA": lambda x: min(x / 12, 1),
        "HBD": lambda x: min(x / 6, 1),
        "RotB": lambda x: min(x / 12, 1),
    }
    fig = go.Figure()
    for i, (smi, pr) in enumerate(preds):
        p = _mol_props(smi)
        if not p:
            continue
        vals = [norm[c](p[c]) for c in cats]
        raw = [p[c] for c in cats]
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=_short(pr["canonical_smiles"], 24),
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ", 0.18)").replace("rgb", "rgba") if color.startswith("rgb") else color,
            opacity=0.6,
            customdata=raw + [raw[0]],
            hovertemplate="%{theta}: %{customdata:.1f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title="Profil fizykochemiczny (znormalizowany 0–1)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        height=460, showlegend=True, **_LAYOUT,
    )
    return fig


def mw_logp_scatter(preds: list[tuple[str, dict]]) -> go.Figure:
    """Mapa przestrzeni MW–LogP z zaznaczonym obszarem zgodnym z regułą Lipinskiego."""
    fig = go.Figure()
    # Obszar drug-like (Ro5: MW<=500, LogP<=5)
    fig.add_shape(type="rect", x0=0, y0=-2, x1=500, y1=5,
                  fillcolor="rgba(26,188,156,0.12)", line=dict(color=C_STRONG, width=1, dash="dash"))
    fig.add_annotation(x=250, y=5.4, text="strefa zgodna z regułą Lipinskiego (Ro5)",
                       showarrow=False, font=dict(color=C_STRONG, size=11))

    xs, ys, txt, cols = [], [], [], []
    for i, (smi, pr) in enumerate(preds):
        p = _mol_props(smi)
        if not p:
            continue
        xs.append(p["MW"]); ys.append(p["LogP"])
        txt.append(_short(pr["canonical_smiles"], 28))
        cols.append(_zone(pr["pIC50_pred"]))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=txt, textposition="top center",
        marker=dict(size=16, color=cols, line=dict(color="white", width=1.5), symbol="diamond"),
        hovertemplate="MW=%{x:.0f}<br>LogP=%{y:.2f}<extra>%{text}</extra>",
    ))
    xmax = max(560, (max(xs) if xs else 0) + 60)
    ymin = min(-2, (min(ys) if ys else 0) - 1)
    ymax = max(6, (max(ys) if ys else 0) + 1)
    grid = dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_layout(
        title="Przestrzeń chemiczna: masa cząsteczkowa vs lipofilowość (LogP)",
        xaxis=dict(title="Masa cząsteczkowa (g/mol)", range=[0, xmax], **grid),
        yaxis=dict(title="LogP", range=[ymin, ymax], **grid),
        height=460, **_LAYOUT,
    )
    return fig
