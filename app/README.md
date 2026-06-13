# BRD4 Activity Predictor — aplikacja LLM + UI

Interfejs Streamlit z asystentem LLM, który **sam wywołuje** model GNN (GIN) oraz
narzędzia RDKit (function calling / tool use) do oceny aktywności związków wobec
białka **BRD4**.

## Architektura

```
Streamlit (streamlit_app.py)
   │  tekst / SMILES od użytkownika
   ▼
Agent LLM (agent.py)  ──bind_tools──►  narzędzia (tools.py)
   │  OpenAI (gpt-4.1)                    ├─ predict_brd4_pic50  → model GNN (chem_model.py → gin_brd4.pt)
   │  function / tool calling             ├─ molecule_descriptors (RDKit)
   ▼                                      ├─ molecular_weight / logp (RDKit)
odpowiedź + ślad wywołań narzędzi         └─ (UI) render_2d, get_descriptors
```

- `chem_model.py` — featuryzacja RDKit + architektura GIN + `predict_pic50` oraz
  `predict_pic50_uncertainty` (MC Dropout) (wczytuje `../checkpoints/gin_brd4.pt`)
- `tools.py` — narzędzia LangChain (odporne na błędny SMILES) + helpery dla UI
  (`render_2d`, `render_2d_zoom_svg`, `mol_3d_html` — py3Dmol)
- `agent.py` — LLM OpenAI + ręczna pętla tool-callingu zwracająca ślad
- `viz.py` — interaktywne wykresy Plotly (gauge, słupki z ±σ, radar, mapa MW–LogP, rozkład pewności)
- `streamlit_app.py` — UI w dwóch zakładkach: **💬 Asystent** (czat + struktura 2D/3D, zoom,
  predykcja, ślad narzędzi) oraz **📊 Prezentacja projektu** (cel, dane, metodyka, wyniki i
  wykresy z notebooka)
- `assets/presentation/` — wykresy wyeksportowane z `analysis.ipynb` do zakładki prezentacji

## Wizualizacje i dodatki

- **Podgląd 3D (py3Dmol)** — konformer 3D (ETKDG + MMFF), interaktywny i obracający się.
  Tryb kolorowania **Pierwiastki** lub **Ładunki Gasteigera** (🔴 ujemny · ⚪ ~0 · 🔵 dodatni).
- **Zoom struktury** — powiększony rysunek 2D z numeracją atomów.
- **Mapa podobieństwa (vs JQ1)** — RDKit similarity map względem referencyjnego inhibitora
  BRD4: zielono = fragmenty zwiększające podobieństwo, różowo = zmniejszające.
- **Pewność modelu (MC Dropout)** — wielokrotna predykcja z aktywnym dropoutem daje
  rozkład pIC50: średnia ±σ (wąsy na słupkach, pasmo na gauge, histogram w zakładce „Pewność").
  Liczbę próbek MC reguluje suwak w panelu bocznym (0 = wyłączone).

## Wymagania wstępne

1. Zainstaluj zależności:
   ```bash
   pip install -r ../requirements.txt
   ```
2. Upewnij się, że istnieje checkpoint modelu `../checkpoints/gin_brd4.pt`
   (generowany przez sekcję **9.20** w `analysis.ipynb`).
3. Skonfiguruj LLM — skopiuj `../.env.example` do `../.env` i ustaw `OPENAI_API_KEY`
   (klucz można też wkleić w panelu bocznym aplikacji).

## Uruchomienie

```bash
cd app
streamlit run streamlit_app.py
```

Aplikacja otworzy się w przeglądarce (domyślnie http://localhost:8501).

## Jak używać

Wpisz w pole czatu np.:
- `Cc1cc(-c2nc3ncccc3n2Cc2ccccc2)cn(C)c1=O` — sama struktura → predykcja + cechy
- „Jak silnym inhibitorem BRD4 jest aspiryna? `CC(=O)Oc1ccccc1C(=O)O`” — LLM rozpozna SMILES i wywoła model
- „Podaj właściwości fizykochemiczne dla `c1ccccc1`” — LLM użyje narzędzi RDKit

Sekcja **„Narzędzia wywołane przez LLM"** pokazuje, że model językowy faktycznie
orkiestruje narzędzia (a nie tylko generuje tekst).
