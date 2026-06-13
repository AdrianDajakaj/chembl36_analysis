"""Streamlit UI — predykcja aktywności wobec BRD4 z asystentem LLM.

Przepływ (zgodny z założeniami projektu):
  user podaje tekst/SMILES → Streamlit pakuje prompt → LLM rozpoznaje input i
  WYWOŁUJE narzędzia (predykcja GNN + RDKit) → UI pokazuje strukturę 2D, tabelę
  cech, predykcję oraz ślad wywołań narzędzi (dowód orkiestracji).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit.components.v1 as components  # noqa: E402

from agent import run_agent  # noqa: E402
from chem_model import MODEL_METRICS, predict_pic50_uncertainty  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from tools import (  # noqa: E402
    get_descriptors,
    mol_3d_html,
    render_2d_png_bytes,
    render_2d_zoom_svg,
    similarity_map_svg,
)
from viz import (  # noqa: E402
    confidence_plot,
    descriptor_radar,
    mw_logp_scatter,
    pic50_bar,
    potency_gauge,
)

RDLogger.DisableLog("rdApp.*")

st.set_page_config(page_title="BRD4 Activity Predictor", page_icon="🧪", layout="wide")


def extract_all_smiles(text: str) -> list[str]:
    """Znajduje wszystkie poprawne, unikalne SMILES w tekście.

    Deduplikacja po formie KANONICZNEJ — RDKit traktuje tekst po spacji jako
    nazwę cząsteczki, więc cały string i sam token mogą dać tę samą strukturę.
    """
    if not text:
        return []
    seen_canon, out = set(), []
    for tok in [text.strip()] + text.split():
        c = tok.strip().strip(".,;:!?()")
        if len(c) < 2:
            continue
        mol = Chem.MolFromSmiles(c)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            continue
        seen_canon.add(canon)
        out.append(canon)
    return out


def potency_label(pic50: float) -> str:
    if pic50 >= 7:
        return "silny inhibitor"
    if pic50 >= 5:
        return "umiarkowany"
    return "słaby"




PRES_DIR = Path(__file__).resolve().parent / "assets" / "presentation"


def _fig(name: str):
    """Ścieżka do wykresu z prezentacji albo None, jeśli plik nie istnieje."""
    p = PRES_DIR / name
    return str(p) if p.exists() else None


def _show_fig(name: str, caption: str) -> None:
    path = _fig(name)
    if path:
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"_(brak pliku wykresu: {name})_")


def render_presentation() -> None:
    """Zakładka prezentacyjna: cel, dane, metodyka, wyniki i wykresy z notebooka."""
    sc, rd = MODEL_METRICS["scaffold"], MODEL_METRICS["random"]

    st.header("📊 Predykcja aktywności małych cząsteczek wobec BRD4")
    st.markdown(
        "Projekt QSAR: na bazie **ChEMBL 36** budujemy model regresji **pIC50** dla "
        "pojedynczego celu białkowego **BRD4** (`CHEMBL1163125`, bromodomena — cel onkologiczny), "
        "a następnie udostępniamy go przez asystenta LLM z **function calling** i narzędziami RDKit."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "R² (scaffold)",
        f"{sc['R2']:.2f}",
        help="GIN + Huber, podział Murcko — generalizacja na nowe chemotypy.",
    )
    c2.metric("R² (random)", f"{rd['R2']:.2f}", help="GIN + Huber, losowy podział 80/10/10.")
    c3.metric(
        "RMSE (scaffold)",
        f"{sc['RMSE']:.2f}",
        help=f"błąd krotności IC50 ≈ ×{sc['fold']:.1f} (10^RMSE)",
    )
    c4.metric("Próg projektu", "R² > 0.5", help="Spełniony przez wszystkie warianty GIN.")

    st.divider()
    st.subheader("1) Problem i cel")
    st.markdown(
        "- **Zadanie:** przewidzieć aktywność biologiczną związku — siłę inhibicji BRD4 "
        "(pIC50 = −log₁₀ IC50[M]).\n"
        "- **Dlaczego pIC50:** rozkład IC50 rozciąga się na rzędy wielkości; skala log jest "
        "stabilna numerycznie i standardowa w QSAR.\n"
        "- **Dlaczego jeden target:** model globalny (≈2900 celów) dawał niskie R² (~0.18, RF); "
        "prosty GCN na wszystkich targetach ~0.06. Zawężenie do BRD4 zamienia problem w spójne "
        "single-target QSAR."
    )

    st.subheader("2) Dane, pipeline i EDA")
    st.markdown(
        "Z ChEMBL 36 wyekstrahowaliśmy **~1,5 mln** pomiarów IC50/Ki; po czyszczeniu zostało "
        "**~1,44 mln** wierszy, a po agregacji medianą do pary związek–target — **~1,05 mln** "
        "obserwacji dla **~2900** celów.\n\n"
        "Modelowanie BRD4 (sekcja 8.2b): **7689** pomiarów IC50 z poprawnymi SMILES, "
        "**~8821** par związek–target, **~3200** unikalnych scaffoldów Murcko.\n\n"
        "**Integralność pipeline'u — trzy naprawione wycieki:**\n"
        "- **Wyciek cech** — usunięto `bei`, `sei`, `le`, `lle` (zawierały pIC50 we wzorze; fałszywe R²≈0.999).\n"
        "- **Wyciek obserwacji** — agregacja medianą pomiarów tej samej pary związek–target.\n"
        "- **Wyciek związków** — GroupShuffleSplit / scaffold split (ten sam związek nie trafia "
        "jednocześnie do train i test)."
    )
    col1, col2 = st.columns(2)
    with col1:
        _show_fig(
            "02_top_targets.png",
            "EDA (cała baza): top 15 celów — BRD4 (#6, ~14,9 tys. pomiarów) to dobrze zbadany target.",
        )
    with col2:
        _show_fig(
            "01_pic50_distribution.png",
            "EDA (cała baza): rozkład pIC50 — dominacja IC50; Ki ma nieco wyższą medianę.",
        )

    st.subheader("3) Metodyka modelowania")
    st.markdown(
        "- **MLP (Morgan/ECFP, 2048 bit):** baseline na odciskach palca cząsteczki.\n"
        "- **GIN (edge-aware, GINEConv):** graf z 33 cechami atomu i 7 cechami wiązań, "
        "BatchNorm + Dropout + połączenia rezydualne (następca słabego GCN z 5 cechami atomu).\n"
        "- **Dwa podziały:** *random* 80/10/10 oraz *scaffold* Murcko (uczciwa ocena na nowych rdzeniach).\n"
        "- **Metryki:** R², RMSE/MAE oraz **błąd krotności IC50** = 10^RMSE (np. RMSE=0.74 → ~×5.5).\n"
        "- **Funkcja straty:** MSE vs Huber (δ=1.0) — Huber wygrywa na wszystkich metrykach GIN."
    )

    st.subheader("4) Wyniki")
    _show_fig(
        "03_r2_comparison.png",
        "Porównanie R² na BRD4 — wszystkie warianty GIN > 0.5; MLP scaffold (0.49) poniżej progu.",
    )
    with st.expander("Tabela metryk (sekcja 9.16 notebooka)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    ["Dummy (mean)", "test", "—", 1.175, 0.986, 0.000, "—"],
                    ["MLP (Morgan FP)", "random", "MSE", 0.742, 0.565, 0.601, "×5.5"],
                    ["MLP (Morgan FP)", "scaffold", "MSE", 0.877, 0.671, 0.493, "×7.6"],
                    ["GIN (edge-aware)", "random", "MSE", 0.653, 0.509, 0.691, "×4.5"],
                    ["GIN (edge-aware)", "random", "Huber", 0.603, 0.462, 0.736, "×4.0"],
                    ["GIN (edge-aware)", "scaffold", "MSE", 0.766, 0.572, 0.613, "×5.8"],
                    ["GIN (edge-aware)", "scaffold", "Huber", 0.739, 0.549, 0.640, "×5.5"],
                ],
                columns=["Model", "Split", "Loss", "RMSE", "MAE", "R²", "błąd IC50"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Model produkcyjny w aplikacji: **GIN · random · Huber** "
            f"(R²={rd['R2']:.3f}, checkpoint `gin_brd4.pt`)."
        )

    _show_fig(
        "04_gnn_pred_vs_actual.png",
        "GIN (MSE): krzywe uczenia + pred vs actual — random (R²=0.69) i scaffold (R²=0.61).",
    )
    col1, col2 = st.columns(2)
    with col1:
        _show_fig(
            "05_gnn_residuals.png",
            "GIN: rozkład reszt — random vs scaffold, MSE vs Huber (4 konfiguracje).",
        )
    with col2:
        _show_fig(
            "06_loss_random.png",
            "MSE vs Huber — random split (Huber lepszy na RMSE, MAE i R²).",
        )
    _show_fig(
        "06_loss_scaffold.png",
        "MSE vs Huber — scaffold split (Murcko); Huber również wygrywa.",
    )

    with st.expander("Baseline MLP (Morgan FP) — pełna wizualizacja z sekcji 8.17", expanded=False):
        _show_fig("08_mlp_learning.png", "Krzywe uczenia — train vs val MSE (random i scaffold).")
        col1, col2 = st.columns(2)
        with col1:
            _show_fig("08_mlp_pred_vs_actual.png", "Pred vs actual — random R²≈0.60, scaffold R²≈0.49.")
        with col2:
            _show_fig("08_mlp_residuals.png", "Rozkład reszt — scaffold ma szerszy rozkład błędów.")
        _show_fig("08_mlp_metrics.png", "Metryki testowe — scaffold trudniejszy (niższe R², wyższe RMSE).")

    st.subheader("5) Analiza błędów (mismatch analysis)")
    st.markdown(
        "Analiza na **scaffold splicie + MSE** (najtrudniejszy scenariusz, nowe rdzenie w teście). "
        "Model ma tendencję do **zaniżania najsilniejszych inhibitorów** (regresja do średniej)."
    )
    _show_fig(
        "07_mismatch_analysis.png",
        "Bias vs pIC50, błąd vs MW/LogP, rozkład |błędu| z progami 0.5 i 1.0 log.",
    )
    st.markdown(
        f"Kalibracja (scaffold test, n={sc['n_test']}): **55%** predykcji w granicy 0.5 log "
        f"(≈ ×3 IC50), **82%** w granicy 1.0 log (≈ ×10), mediana |błędu| ≈ **0.42**, "
        "bias ≈ **−0.04** (niewielkie zaniżanie)."
    )

    st.subheader("6) Wnioski")
    st.markdown(
        "- **GIN > MLP** na BRD4; wszystkie warianty GIN przekraczają próg R²>0.5, "
        "MLP na scaffold split — nie (R²≈0.49).\n"
        "- **Model produkcyjny:** GIN + Huber, random split (R²≈0.74); jakość na nowych "
        f"rdzeniach potwierdza scaffold Huber (R²≈{sc['R2']:.2f}).\n"
        "- **Niepewność** (MC Dropout) i **błąd krotności IC50** dają praktyczną interpretację predykcji.\n"
        "- **Aplikacja LLM** łączy model z narzędziami RDKit: plan → wywołania → interpretacja, "
        "z wizualizacjami 2D/3D, mapą podobieństwa do JQ1 (referencyjny inhibitor) i licznikiem kosztów."
    )
    st.caption("Wykresy wyeksportowane z notebooka analitycznego (analysis.ipynb, sekcje 5–9).")


# --------------------------- Sidebar: konfiguracja ---------------------------
st.session_state.setdefault("usage_tot", {"in": 0, "out": 0, "cost": 0.0, "n": 0})


def render_usage() -> None:
    """Odświeża licznik zużycia w panelu bocznym (na żywo)."""
    u = st.session_state["usage_tot"]
    with usage_box.container():
        c1, c2 = st.columns(2)
        c1.metric("Zapytania", u["n"])
        c2.metric("Koszt sesji", f"${u['cost']:.4f}")
        st.caption(f"Tokeny: {u['in']:,} wej. / {u['out']:,} wyj.  ·  model: `{model_name}`")
        if u["n"]:
            st.caption(f"Śr. koszt / zapytanie: ${u['cost'] / u['n']:.4f}")


with st.sidebar:
    st.header("⚙️ Konfiguracja")
    st.caption("Silnik LLM: **OpenAI** (najpewniejszy tool calling).")
    model_name = st.text_input("Model OpenAI", value=os.getenv("OPENAI_MODEL", "gpt-4.1"))
    os.environ["OPENAI_MODEL"] = model_name
    api_key = st.text_input("OPENAI_API_KEY", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["LLM_PROVIDER"] = "openai"

    st.divider()
    st.subheader("🎨 Wizualizacje")
    show_3d = st.toggle("Podgląd 3D (py3Dmol)", value=True,
                        help="Generuje konformer 3D (ETKDG+MMFF) i obraca cząsteczkę.")
    color_by = st.radio("Kolorowanie 3D", ["element", "charge"], index=0,
                        format_func=lambda x: "Pierwiastki" if x == "element" else "Ładunki (Gasteiger)",
                        horizontal=True)
    show_simmap = st.toggle("Mapa podobieństwa vs JQ1", value=True,
                            help="Które fragmenty zbliżają cząsteczkę do referencyjnego inhibitora BRD4 (JQ1).")
    mc_samples = st.slider("Próbki MC Dropout (pewność)", 0, 80, 30, step=10,
                           help="0 = wyłącz oszacowanie niepewności. Więcej = stabilniejsza σ, ale wolniej.")

    st.divider()
    st.subheader("📊 Zużycie sesji")
    usage_box = st.empty()
    render_usage()
    if st.button("↺ Wyzeruj licznik"):
        st.session_state.usage_tot = {"in": 0, "out": 0, "cost": 0.0, "n": 0}
        st.rerun()

    st.divider()
    st.subheader("🎯 Target")
    st.markdown("**BRD4** — Bromodomain-containing protein 4\n\n`CHEMBL1163125` · *Homo sapiens*")
    st.caption("Model: GIN (edge-aware) trenowany na ChEMBL 36, regresja pIC50.")

    st.markdown("**📐 Jakość modelu (GIN + Huber, test)**")
    _sc, _rd = MODEL_METRICS["scaffold"], MODEL_METRICS["random"]
    r1, r2 = st.columns(2)
    r1.metric("R² scaffold", f"{_sc['R2']:.2f}",
              help="Podział wg rdzeni Murcko — generalizacja na NOWE chemotypy (trudniejszy).")
    r2.metric("R² random", f"{_rd['R2']:.2f}",
              help="Losowy podział — wariant wczytany w aplikacji (checkpoint gin_brd4.pt).")
    st.caption(f"scaffold: RMSE={_sc['RMSE']:.2f} (≈ ×{_sc['fold']:.1f} IC50) · "
               f"random: RMSE={_rd['RMSE']:.2f} (≈ ×{_rd['fold']:.1f}). "
               f"Próg R²>0.5 — spełniony przez wszystkie warianty GIN.")

    st.divider()
    if st.button("🗑️ Wyczyść rozmowę"):
        st.session_state.history = []
        st.rerun()


st.title("🧪 BRD4 Activity Predictor + Asystent LLM")
st.caption("Podaj SMILES lub zapytaj o cząsteczkę w języku naturalnym. "
           "LLM sam wywoła model GNN i narzędzia RDKit.")

tab_app, tab_pres = st.tabs(["💬 Asystent", "📊 Prezentacja projektu"])

if "history" not in st.session_state:
    st.session_state.history = []  # lista (role, content)

# chat_input musi być na poziomie strony (nie wewnątrz zakładki/kontenera)
user_input = st.chat_input("Wpisz SMILES lub pytanie…")

with tab_pres:
    render_presentation()

with tab_app:
    # Przykłady do szybkiego startu
    with st.expander("💡 Przykładowe zapytania", expanded=False):
        st.markdown("**🎯 Predykcja aktywności (pIC50/IC50 wobec BRD4)**")
        st.markdown(
            "- `Cc1cc(-c2nc3ncccc3n2Cc2ccccc2)cn(C)c1=O` — przewidź aktywność\n"
            "- `Jak silnym inhibitorem BRD4 jest aspiryna? CC(=O)Oc1ccccc1C(=O)O`\n"
            "- `Czy kofeina działa na BRD4? Cn1cnc2c1c(=O)n(C)c(=O)n2C`\n"
            "- `Oceń aktywność biologiczną paracetamolu wobec BRD4: CC(=O)Nc1ccc(O)cc1`"
        )
        st.markdown("**⚖️ Porównania wielu cząsteczek**")
        st.markdown(
            "- `Który z nich jest lepszym inhibitorem BRD4: CCO czy "
            "Cc1cc(-c2nc3ncccc3n2Cc2ccccc2)cn(C)c1=O?`\n"
            "- `Porównaj aktywność: CC(=O)Oc1ccccc1C(=O)O oraz Cn1cnc2c1c(=O)n(C)c(=O)n2C`\n"
            "- `Uszereguj od najsilniejszego: c1ccccc1, CCO, "
            "Cc1cc(-c2nc3ncccc3n2Cc2ccccc2)cn(C)c1=O`"
        )
        st.markdown("**🧪 Właściwości fizykochemiczne i drug-likeness**")
        st.markdown(
            "- `Podaj właściwości fizykochemiczne dla c1ccccc1`\n"
            "- `Czy ibuprofen spełnia regułę Lipinskiego? CC(C)Cc1ccc(C(C)C(=O)O)cc1`\n"
            "- `Porównaj masę i LogP dla CCO oraz CCCCCCCC`\n"
            "- `Ile wynosi TPSA i liczba donorów wodoru dla CC(=O)Nc1ccc(O)cc1?`"
        )
        st.markdown("**🔗 Podobieństwo do znanego inhibitora (JQ1)**")
        st.markdown(
            "- `Jak bardzo aspiryna przypomina JQ1? CC(=O)Oc1ccccc1C(=O)O`\n"
            "- `Czy ta cząsteczka jest podobna do znanego inhibitora BRD4? "
            "Cc1cc(-c2nc3ncccc3n2Cc2ccccc2)cn(C)c1=O`"
        )
        st.markdown("**🚫 Obsługa błędów**")
        st.markdown("- `Przewidź aktywność dla to-nie-jest-smiles-123`")

    # --------------------------- Render historii ---------------------------
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    # --------------------------- Obsługa nowego inputu ---------------------------
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.history.append(("user", user_input))

        # === 1) ODPOWIEDŹ LLM (plan → narzędzia → interpretacja) — NAJPIERW ===
        with st.chat_message("assistant"):
            with st.spinner("Asystent analizuje (LLM + narzędzia)…"):
                lc_history = []
                for role, content in st.session_state.history[:-1]:
                    lc_history.append(
                        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
                    )
                result = run_agent(user_input, history=lc_history)

            u = result.get("usage") or {}
            tot = st.session_state["usage_tot"]
            tot["in"] += int(u.get("input_tokens", 0) or 0)
            tot["out"] += int(u.get("output_tokens", 0) or 0)
            tot["cost"] += float(u.get("cost_usd", 0.0) or 0.0)
            tot["n"] += 1
            render_usage()

            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown(result["answer"] or "_(brak treści)_")
                st.session_state.history.append(("assistant", result["answer"] or ""))
                if u:
                    st.caption(f"💸 To zapytanie: {u.get('input_tokens', 0)}+{u.get('output_tokens', 0)} tok. "
                               f"≈ ${u.get('cost_usd', 0):.4f} ({u.get('model', model_name)})")

            plan = result.get("plan") or ""
            turns = result.get("turns") or []
            n_calls = sum(len(t["calls"]) for t in turns) if turns else len(result.get("trace") or [])
            if plan or turns:
                with st.expander(f"🧠 Plan i rozumowanie LLM ({n_calls} wywołań narzędzi)", expanded=True):
                    if plan:
                        st.markdown("**🧭 Plan analizy (przed wykonaniem):**")
                        st.info(plan)
                    for ti, turn in enumerate(turns, 1):
                        st.markdown(f"#### ⚙️ Wykonanie — krok {ti}")
                        if turn["thought"]:
                            st.markdown(f"**🧩 Rozumowanie:**\n\n{turn['thought']}")
                        for step in turn["calls"]:
                            st.markdown(f"**`{step['tool']}`** — argumenty: `{step['args']}`")
                            st.code(str(step["result"]), language="text")
                        if ti < len(turns):
                            st.divider()
            elif result["trace"]:
                with st.expander(f"🔧 Narzędzia wywołane przez LLM ({len(result['trace'])})", expanded=True):
                    for i, step in enumerate(result["trace"], 1):
                        st.markdown(f"**{i}. `{step['tool']}`** — argumenty: `{step['args']}`")
                        st.code(str(step["result"]), language="text")

        # === 2) MATERIAŁ DOWODOWY: struktury, cechy i wizualizacje — POD SPODEM ===
        smiles_list = extract_all_smiles(user_input)

        # Panel: dla KAŻDEJ wykrytej cząsteczki — struktura 2D + cechy + predykcja
        preds: list[tuple[str, dict]] = []
        if smiles_list:
            st.subheader(f"🔬 Wykryte cząsteczki ({len(smiles_list)}) — struktura 2D/3D, cechy, predykcja")
            for smi in smiles_list:
                col1, col2 = st.columns([1, 1])
                with col1:
                    png = render_2d_png_bytes(smi)
                    if png:
                        st.image(png, caption=smi)
                with col2:
                    desc = get_descriptors(smi)
                    if desc:
                        st.dataframe(
                            pd.DataFrame({"Cecha": list(desc.keys()), "Wartość": list(desc.values())}),
                            hide_index=True, use_container_width=True,
                        )
                    pred = predict_pic50_uncertainty(smi, n_samples=mc_samples)
                    if pred.get("ok"):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("pIC50 (predykcja)", pred["pIC50_pred"])
                        m2.metric("IC50 ≈ [nM]", pred["IC50_nM_pred"])
                        if pred.get("mc_std") is not None:
                            m3.metric("Niepewność ±σ", pred["mc_std"], help=f"Pewność: {pred.get('confidence')}")
                        st.caption(f"Ocena: **{potency_label(pred['pIC50_pred'])}**")
                        preds.append((smi, pred))

                # Dodatki: zoom 2D + 3D (z kolorowaniem) + mapa podobieństwa do JQ1
                with st.expander("🔬 Zoom struktury, podgląd 3D i mapa podobieństwa (vs JQ1)", expanded=False):
                    vt = st.tabs(["🔎 Zoom 2D", "🧬 Podgląd 3D", "🗺️ Mapa podobieństwa"])
                    with vt[0]:
                        svg = render_2d_zoom_svg(smi)
                        if svg:
                            components.html(svg, height=470)
                        st.caption("Powiększona struktura z numeracją atomów.")
                    with vt[1]:
                        if show_3d:
                            html = mol_3d_html(smi, color_by=color_by)
                            if html:
                                components.html(html, height=440)
                                if color_by == "charge":
                                    st.caption("Kolory = ładunki cząstkowe Gasteigera: 🔴 ujemny · ⚪ ~0 · 🔵 dodatni.")
                                else:
                                    st.caption("Konformer 3D (ETKDG + MMFF), kolory wg pierwiastków.")
                            else:
                                st.info("Nie udało się wygenerować konformeru 3D dla tej cząsteczki.")
                        else:
                            st.caption("Podgląd 3D wyłączony w panelu bocznym.")
                    with vt[2]:
                        if show_simmap:
                            sim = similarity_map_svg(smi)
                            if sim:
                                components.html(sim, height=400)
                                st.caption("Vs **JQ1** (referencyjny inhibitor BRD4). "
                                           "Zielono = fragmenty zwiększające podobieństwo, różowo = zmniejszające.")
                            else:
                                st.info("Nie udało się wygenerować mapy podobieństwa.")
                        else:
                            st.caption("Mapa podobieństwa wyłączona w panelu bocznym.")
                st.divider()

            # Wizualizacja wyników predykcji (wymóg 5.0: „wizualizuje wyniki")
            if preds:
                st.subheader("📈 Wizualizacja wyników")
                has_unc = any(pr.get("mc_samples") for _, pr in preds)
                if len(preds) == 1:
                    names = ["🎯 Siła działania", "🕸️ Profil cech", "🗺️ Przestrzeń MW–LogP"]
                    if has_unc:
                        names.append("🎲 Pewność (MC Dropout)")
                    tabs = st.tabs(names)
                    with tabs[0]:
                        st.plotly_chart(potency_gauge(preds[0][1]), use_container_width=True)
                    with tabs[1]:
                        st.plotly_chart(descriptor_radar(preds), use_container_width=True)
                    with tabs[2]:
                        st.plotly_chart(mw_logp_scatter(preds), use_container_width=True)
                    if has_unc:
                        with tabs[3]:
                            st.plotly_chart(confidence_plot(preds[0][1]), use_container_width=True)
                else:
                    names = ["📊 Porównanie pIC50", "🕸️ Profil cech", "🗺️ Przestrzeń MW–LogP"]
                    if has_unc:
                        names.append("🎲 Pewność (MC Dropout)")
                    tabs = st.tabs(names)
                    with tabs[0]:
                        st.plotly_chart(pic50_bar(preds), use_container_width=True)
                    with tabs[1]:
                        st.plotly_chart(descriptor_radar(preds), use_container_width=True)
                    with tabs[2]:
                        st.plotly_chart(mw_logp_scatter(preds), use_container_width=True)
                    if has_unc:
                        with tabs[3]:
                            for smi_u, pr_u in preds:
                                if pr_u.get("mc_samples"):
                                    st.plotly_chart(confidence_plot(pr_u), use_container_width=True)
