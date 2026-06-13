# ChEMBL 36 — analiza bioaktywności i predykcja pIC50 (BRD4)

End-to-end projekt QSAR na bazie [ChEMBL 36](https://www.ebi.ac.uk/chembl/): ekstrakcja SQL → czyszczenie → EDA → modele baseline → **MLP (Morgan FP)** → **GIN (graf molekularny)** na celu **BRD4**, z aplikacją **Streamlit + LLM** (function calling, RDKit, MC Dropout).

## Kluczowe wyniki

### Pipeline danych

| Etap | Wartość |
|------|---------|
| Surowe pomiary IC50/Ki | ~1,5 mln |
| Po czyszczeniu | 1 435 971 wierszy |
| Po agregacji (mediana na parę związek–target) | 1 048 338 obserwacji |
| Unikalne związki | 808 633 |
| Unikalne targety | 2 973 |

### Model globalny (sekcja 7 — deskryptory fizykochemiczne)

| Model | R² (test) | Uwagi |
|-------|-----------|--------|
| Dummy (średnia) | ~0 | baseline |
| Ridge Regression | 0,09 | |
| **Random Forest** | **0,18** | najlepszy baseline tabelaryczny |
| LightGBM | 0,18 | |

Split: GroupShuffleSplit 80/20 po `molregno` (bez wspólnych związków train/test).

### Single-target QSAR — BRD4 (`CHEMBL1163125`)

Zbiór: **7689** pomiarów IC50 z poprawnymi SMILES, split **80/10/10** (random + scaffold Murcko).

| Model | Split | Loss | R² | błąd IC50 (×) |
|-------|-------|------|-----|---------------|
| MLP (Morgan FP 2048) | random | MSE | 0,60 | ×5,5 |
| MLP (Morgan FP) | scaffold | MSE | 0,49 | ×7,6 |
| GIN (GINEConv) | random | Huber | **0,74** | ×4,0 |
| GIN (GINEConv) | scaffold | Huber | **0,64** | ×5,5 |

**Próg projektu:** R² > 0,5 — spełniony przez wszystkie warianty GIN (także scaffold).

**Model produkcyjny:** GIN + Huber, random split → `checkpoints/gin_brd4.pt` (sekcja 9.20 notebooka).

Integralność: naprawiono wyciek cech (`bei`/`sei`/`le`/`lle`), wyciek obserwacji (agregacja par) i wyciek związków (GroupShuffleSplit / scaffold).

## Struktura repozytorium

```
chembl36_analysis/
├── analysis.ipynb              # Główny notebook (sekcje 1–9, ~200+ komórek)
├── requirements.txt            # Zależności Python (notebook + app)
├── docker-compose.yml          # PostgreSQL 17 + RDKit
├── setup_chembl.sh             # Pobieranie i import dumpu ChEMBL 36
├── .env.example                # Szablon zmiennych (DB, OpenAI)
├── checkpoints/
│   └── gin_brd4.pt             # Wytrenowany GIN (BRD4)
├── app/                        # Aplikacja Streamlit + LLM
│   ├── streamlit_app.py        # UI: Asystent + Prezentacja projektu
│   ├── agent.py                # Orkiestracja LLM (OpenAI, tool calling)
│   ├── tools.py                # Narzędzia LangChain + RDKit
│   ├── chem_model.py           # GIN + predict_pic50 (+ MC Dropout)
│   ├── viz.py                  # Wykresy Plotly
│   ├── assets/presentation/    # Wykresy wyeksportowane z notebooka
│   └── README.md               # Dokumentacja aplikacji
├── scripts/                    # Skrypty pomocnicze (TOC notebooka, eksport PNG, …)
├── reference/
│   └── chembl_36_schema.png
└── downloads/                  # (gitignored) dump PostgreSQL
```

## Notebook — spis treści

1. **Połączenie z bazą** — konfiguracja, sanity check  
2. **Informacje o bazie** — schemat ER, referencja 73 tabel  
3. **Ekstrakcja danych** — join 6 tabel ChEMBL  
4. **Czyszczenie** — duplikaty, pIC50, braki, outliery  
5. **EDA** — rozkłady, korelacje, Lipiński, targety, pary związek–target, t-SNE, agregacja  
6. **Inżynieria cech** — 22 cechy, usunięcie wyciekających deskryptorów, VIF  
7. **Modele bazowe** — Dummy, Ridge, RF, LightGBM, SHAP, krzywa uczenia  
8. **MLP (Morgan FP)** — ECFP 2048, random + scaffold split, wizualizacja wyników  
   - **8.2b** — zawężenie do targetu BRD4  
9. **GIN (Graph Isomorphism Network)** — GINEConv, MSE vs Huber, porównanie z MLP  
   - **9.16** — tabela zbiorcza · **9.19** — mismatch analysis · **9.20** — `predict_pic50` + zapis checkpointu  

Spis treści w notebooku (komórka 1) ma kotwice `#sec-*` — klikalne w Cursor/VS Code.

## Aplikacja Streamlit

Interfejs z dwoma zakładkami:

- **Asystent** — czat w języku naturalnym + SMILES; LLM wywołuje model GIN i narzędzia RDKit (struktura 2D/3D, deskryptory, mapa podobieństwa vs **JQ1**, MC Dropout).
- **Prezentacja projektu** — skrót narracji + wykresy z notebooka.

```bash
# Po instalacji zależności i wygenerowaniu checkpointu (sekcja 9.20):
cp .env.example .env   # uzupełnij OPENAI_API_KEY
streamlit run app/streamlit_app.py
```

Szczegóły: [app/README.md](app/README.md).

Lokalnie (poza gitem): notatki do prezentacji, zestawienia krok po kroku — patrz `.gitignore`.

## Wymagania

- **Docker Desktop** (min. ~8 GB RAM dla kontenera)
- **Python 3.10+**
- ~3 GB miejsca na dump ChEMBL 36
- **GPU** — opcjonalne (trening GIN działa też na CPU; w notebooku użyto PyTorch)
- **OpenAI API key** — tylko dla zakładki Asystent (LLM)

## Instalacja

### 1. Klon repozytorium

```bash
git clone <repo-url>
cd chembl36_analysis
```

### 2. Baza danych (PostgreSQL + RDKit)

```bash
docker-compose up -d
chmod +x setup_chembl.sh
./setup_chembl.sh
```

Import trwa ok. 10–20 min. Obraz: `jeffchen94/postgres-rdkit:17-rdkit_2025_09_3-trixie`, port **5432**.

### 3. Środowisko Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Notebook

Otwórz `analysis.ipynb` w VS Code / JupyterLab i uruchom sekwencyjnie (lub **Run All** — sekcje 8–9 wymagają GPU/CPU i ~kilku–kilkunastu minut treningu).

Checkpoint `checkpoints/gin_brd4.pt` powstaje w **sekcji 9.20**. Jeśli jest w repozytorium, aplikacja działa bez ponownego treningu.

### 5. Aplikacja (opcjonalnie)

```bash
cp .env.example .env
# Edytuj: OPENAI_API_KEY=..., opcjonalnie OPENAI_MODEL=gpt-4.1
streamlit run app/streamlit_app.py
```

## Połączenie z bazą

Zmienne w `.env` (Docker Compose):

| Zmienna | Domyślnie |
|---------|-----------|
| `DB_USER` | `admin` |
| `DB_PASSWORD` | `chembl_pass` |
| `DB_NAME` | `chembl_36` |

Connection string w notebooku:

```
postgresql+psycopg2://admin:chembl_pass@localhost:5432/chembl_36
```

## Docker — start / stop

```bash
docker-compose down          # zatrzymaj (dane w volume zostają)
docker-compose up -d         # wznów
docker-compose down -v       # usuń volume (wymaga ponownego importu)
```

## Stack technologiczny

| Warstwa | Narzędzia |
|---------|-----------|
| Baza | PostgreSQL 17, RDKit cartridge, Docker |
| Dane | ChEMBL 36, pandas, SQLAlchemy |
| Analiza | NumPy, Matplotlib, Seaborn, SciPy, statsmodels |
| ML baseline | scikit-learn, LightGBM, SHAP |
| ML strukturalny | PyTorch, PyTorch Geometric, RDKit (Morgan FP, grafy) |
| Aplikacja | Streamlit, Plotly, py3Dmol, LangChain, OpenAI API |
| Notebook | Jupyter / VS Code |

## Skrypty pomocnicze

| Skrypt | Opis |
|--------|------|
| `scripts/export_presentation_figures.py` | Eksport wykresów MLP do `app/assets/presentation/` |
| `scripts/regenerate_toc.py` | Regeneracja spisu treści notebooka |
| `scripts/fix_toc_navigation.py` | Kotwice w nagłówkach (nawigacja TOC) |
| `scripts/pick_best_target.py` | Wybór targetu pod single-target QSAR |
| `scripts/run_from_82.py` | Uruchomienie notebooka od sekcji 8 |

## Licencja danych

Dane ChEMBL: [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
