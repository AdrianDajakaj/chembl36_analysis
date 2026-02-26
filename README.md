# ChEMBL 36 Bioactivity Analysis

End-to-end analysis of the [ChEMBL 36](https://www.ebi.ac.uk/chembl/) bioactivity database — from raw SQL extraction through exploratory data analysis to baseline predictive models for pIC50.

## Key Results

| Metric | Value |
|---|---|
| Raw dataset | 1,435,971 bioactivity measurements |
| After aggregation | 1,048,338 unique (compound, target) pairs |
| Unique compounds | 808,633 |
| Unique targets | 2,973 |
| Best model | Random Forest (R² = 0.179, RMSE = 1.182) |
| Features used | 22 physicochemical descriptors (no structural fingerprints) |

## Project Structure

```
chembl36_analysis/
├── analysis.ipynb          # Main analysis notebook (96 cells, fully executable)
├── docker-compose.yml      # PostgreSQL 17 + RDKit container definition
├── setup_chembl.sh         # Automated DB download & import script
├── .env                    # Database credentials (DB_USER, DB_PASSWORD, DB_NAME)
├── .gitignore
├── reference/
│   └── chembl_36_schema.png  # ER diagram of ChEMBL 36
└── downloads/              # (gitignored) ChEMBL dump files
```

## Notebook Outline

1. **Database Connection** — connect to PostgreSQL, sanity checks
2. **Database Info** — schema diagram, 73-table reference with column descriptions
3. **Data Extraction** — single SQL query joining 6 core ChEMBL tables
4. **Data Cleaning** — missing values, type casting, pIC50 conversion, duplicate handling
5. **Exploratory Data Analysis** — 12 visualizations with takeaways:
   - pIC50 distribution, physicochemical features, correlation matrix
   - IC50 vs Ki comparison, Lipinski Ro5 compliance
   - Hexbin scatter plots, top target classes, duplicate pair analysis
   - Outlier analysis, t-SNE chemical space visualization
   - Compound–target aggregation (median per pair)
6. **Feature Engineering** — 22 features: ratios, bins, log-transforms, VIF check
7. **Baseline Models** — Dummy, Ridge, Random Forest, LightGBM
   - GroupShuffleSplit (no compound leakage)
   - Feature importance, SHAP analysis, residual diagnostics, learning curve

## Prerequisites

- **Docker Desktop** (with at least 8 GB RAM allocated)
- **Python 3.10+** with `pip`
- ~3 GB free disk space for the ChEMBL 36 database dump

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd chembl36_analysis
```

### 2. Start the PostgreSQL container

```bash
docker-compose up -d
```

This pulls the `jeffchen94/postgres-rdkit:17-rdkit_2025_09_3-trixie` image (PostgreSQL 17 + RDKit cartridge) and starts it on **port 5432**.

Verify it's running:

```bash
docker ps
# Should show chembl_container with status "Up ... (healthy)"
```

### 3. Download & import ChEMBL 36

```bash
chmod +x setup_chembl.sh
./setup_chembl.sh
```

This script will:
1. Download `chembl_36_postgresql.tar.gz` (~1.5 GB) from the EBI FTP server
2. Extract the dump file
3. Wait for the container to be ready
4. Import the data via `pg_restore` (takes ~10–20 minutes)

> **Note:** If the script fails at the import step (e.g., some `pg_restore` warnings about RDKit types), the data is usually imported correctly. You can verify by running the sanity check in the notebook.

### 4. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Python dependencies

```bash
pip install pandas sqlalchemy psycopg2-binary numpy matplotlib seaborn \
            scipy scikit-learn lightgbm statsmodels shap ipykernel
```

### 6. Run the notebook

Open `analysis.ipynb` in VS Code or JupyterLab and **Run All** cells. The full pipeline takes approximately 5–10 minutes depending on hardware.

## Database Credentials

Defined in `.env` (loaded by Docker Compose):

| Variable | Default |
|---|---|
| `DB_USER` | `admin` |
| `DB_PASSWORD` | `chembl_pass` |
| `DB_NAME` | `chembl_36` |

Connection string used in the notebook:

```
postgresql+psycopg2://admin:chembl_pass@localhost:5432/chembl_36
```

## Stopping & Restarting

```bash
# Stop the container (data persists in Docker volume)
docker-compose down

# Restart later (no re-import needed)
docker-compose up -d
```

To completely remove all data (including the Docker volume):

```bash
docker-compose down -v
```

## Tech Stack

- **Database:** PostgreSQL 17 + RDKit cartridge (Docker)
- **Data:** ChEMBL 36 (EBI, 2024)
- **Python:** pandas, SQLAlchemy, NumPy, Matplotlib, Seaborn, SciPy
- **ML:** scikit-learn, LightGBM, SHAP
- **Environment:** Jupyter / VS Code Notebooks

## License

ChEMBL data is provided under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/).
