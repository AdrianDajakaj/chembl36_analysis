"""Wykonuje analysis.ipynb od sekcji 8.2 (komórka START) do końca.

Pomija ciężką ekstrakcję 3-7 — sekcja 8.2 wczytuje df z checkpointu df_before_mlp.pkl,
a kolejna komórka (8.2b) zawęża go do BRD4. Wyniki (outputy komórek) zapisywane są
z powrotem do tego samego pliku .ipynb.
"""
import os
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = "analysis.ipynb"
START = 132  # sekcja 8.2 — wczytanie checkpointu

nb = nbformat.read(NB, as_version=4)
total = len(nb.cells)
n_code = sum(1 for c in nb.cells[START:] if c.cell_type == "code")
print(f"[run] cells total={total}  exec_from={START}  code_cells_to_run={n_code}", flush=True)

sub = nbformat.v4.new_notebook()
sub.metadata = nb.metadata
sub.cells = nb.cells[START:]  # te same obiekty -> outputy wpisują się też do nb

client = NotebookClient(
    sub,
    timeout=7200,
    kernel_name="chembl-venv",
    resources={"metadata": {"path": os.getcwd()}},
    allow_errors=False,
)

t0 = time.time()
status = "OK"
try:
    client.execute()
except CellExecutionError as e:
    status = "ERROR"
    print("[run] CELL EXECUTION ERROR:\n" + str(e)[:3000], flush=True)
except Exception as e:  # noqa: BLE001
    status = f"FATAL: {type(e).__name__}: {e}"
    print("[run] FATAL:", status, flush=True)
finally:
    nb.cells[START:] = sub.cells
    nbformat.write(nb, NB)
    print(f"[run] saved notebook  status={status}  elapsed={time.time() - t0:.0f}s", flush=True)
