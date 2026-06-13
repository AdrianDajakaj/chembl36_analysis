"""Warstwa LLM (OpenAI) z function/tool calling.

Model OpenAI (domyślnie `gpt-4.1` — mocny tool calling i interpretacja).
Pętla agenta jest ręczna i zwraca pełny
**ślad wywołań narzędzi** — to dowód, że LLM faktycznie orkiestruje narzędzia
(wymóg 4.0+).
"""
from __future__ import annotations

import os

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rdkit import Chem, RDLogger

from tools import ALL_TOOLS

RDLogger.DisableLog("rdApp.*")

PLAN_PROMPT = """Jesteś asystentem chemoinformatycznym (cel: BRD4). Twoim zadaniem
jest TYLKO zaplanowanie analizy — NIE wywołuj jeszcze żadnych narzędzi i nie podawaj
wyników liczbowych.

Napisz zwięzły plan (2-4 zdania, po polsku):
- co rozpoznajesz w zapytaniu (ile cząsteczek, jakie SMILES / nazwy),
- które narzędzia zamierzasz wywołać i w jakiej kolejności (predict_brd4_pic50 dla
  aktywności, molecule_descriptors dla właściwości, similarity_to_jq1 dla podobieństwa),
- na co zwrócisz uwagę przy interpretacji.
"""

SYSTEM_PROMPT = """Jesteś asystentem chemoinformatycznym wyspecjalizowanym w ocenie
aktywności związków wobec białka BRD4 (bromodomena, cel onkologiczny).

Masz do dyspozycji narzędzia:
- predict_brd4_pic50(smiles): przewiduje pIC50 i IC50[nM] modelem GNN (GIN) wraz z
  niepewnością (MC Dropout), oceną reguły Lipinskiego i podobieństwem do JQ1,
- molecule_descriptors(smiles): deskryptory fizykochemiczne (RDKit),
- molecular_weight(smiles), logp(smiles): pojedyncze właściwości,
- similarity_to_jq1(smiles): podobieństwo Tanimoto do wzorcowego inhibitora BRD4 (JQ1).

SPOSÓB PRACY (zawsze trzymaj się tej struktury):
1. NARZĘDZIA — plan analizy jest już ustalony; teraz FAKTYCZNIE wywołaj właściwe
   narzędzia (użyj mechanizmu tool-calling, nie opisuj ich słowami).
   Nigdy nie zgaduj liczb — bierz je wyłącznie z narzędzi.
   - aktywność biologiczna / siła inhibicji / pIC50 / IC50 → predict_brd4_pic50,
   - właściwości fizykochemiczne / drug-likeness → molecule_descriptors,
   - podobieństwo do znanych inhibitorów → similarity_to_jq1.
   Przy wielu cząsteczkach wywołaj narzędzia dla KAŻDEJ z nich.
2. INTERPRETACJA — gdy masz już wyniki narzędzi, napisz zwięzłą, ale konkretną analizę
   w formie sekcji:
   • **Aktywność**: kategoria (pIC50 > 7 silny, 5-7 umiarkowany, < 5 słaby) + IC50 w nM,
     i co to znaczy praktycznie.
   • **Pewność**: skomentuj niepewność (±σ / fold-error) — czy predykcji można ufać.
   • **Właściwości i drug-likeness**: MW, LogP, naruszenia reguły Lipinskiego (Ro5).
   • **Podobieństwo do JQ1**: czy związek przypomina znany chemotyp i jak to wspiera wynik.
   • **Wniosek**: 1 zdanie podsumowania (i ewentualne zastrzeżenia/ograniczenia modelu).
   Przy porównaniach cząsteczek wskaż, która jest lepszym kandydatem i dlaczego.

Zasady dodatkowe:
- Odpowiadaj po polsku, rzeczowo, bez lania wody.
- Jeśli SMILES jest niepoprawny, jasno o tym poinformuj i nie wymyślaj wyników.
- Pamiętaj, że model jest trenowany TYLKO dla BRD4 i ma ograniczoną pewność —
  sygnalizuj to przy nietypowych/odległych chemicznie cząsteczkach.
"""

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Cennik (USD za 1M tokenów: wejście, wyjście) — stan: czerwiec 2026.
PRICING = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.4),
}


def _price_for(model: str) -> tuple[float, float]:
    """Dopasowuje cennik po najdłuższym pasującym prefiksie nazwy modelu."""
    best = None
    for key, val in PRICING.items():
        if model.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, val)
    return best[1] if best else (2.0, 8.0)  # fallback: jak gpt-4.1


def _add_usage(usage: dict, ai) -> None:
    """Dolicza tokeny z odpowiedzi modelu (AIMessage.usage_metadata)."""
    meta = getattr(ai, "usage_metadata", None) or {}
    usage["input_tokens"] += int(meta.get("input_tokens", 0) or 0)
    usage["output_tokens"] += int(meta.get("output_tokens", 0) or 0)


def _chat(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), temperature=temperature)


def get_llm(temperature: float = 0.0, force_tools: bool = False):
    """Zwraca model czatu OpenAI z dowiązanymi narzędziami.

    force_tools=True wymusza wywołanie ≥1 narzędzia (tool_choice='required') —
    zapobiega sytuacji, gdy model opisuje narzędzia zamiast je wywołać.
    """
    llm = _chat(temperature)
    if force_tools:
        return llm.bind_tools(ALL_TOOLS, tool_choice="required")
    return llm.bind_tools(ALL_TOOLS)


def _has_smiles(text: str) -> bool:
    """Czy w tekście jest cokolwiek, co RDKit rozpozna jako cząsteczkę."""
    if not text:
        return False
    for tok in text.replace(",", " ").split():
        tok = tok.strip(".,;:!?()")
        if len(tok) >= 2 and Chem.MolFromSmiles(tok) is not None:
            return True
    return False


def run_agent(
    user_message: str,
    history: list | None = None,
    max_iters: int = 5,
) -> dict:
    """Uruchamia agenta (faza planu + faza wykonania z narzędziami).

    Zwraca {'answer', 'plan', 'turns', 'trace', 'usage', 'error'}.
    'usage' = {'input_tokens', 'output_tokens', 'cost_usd', 'model'}.
    """
    trace: list[dict] = []   # płaska lista wywołań (kompatybilność wstecz)
    turns: list[dict] = []   # bogaty ślad: {"thought": str, "calls": [...]}
    usage = {"input_tokens": 0, "output_tokens": 0}
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    def _usage_out() -> dict:
        p_in, p_out = _price_for(model)
        cost = usage["input_tokens"] / 1e6 * p_in + usage["output_tokens"] / 1e6 * p_out
        return {**usage, "cost_usd": round(cost, 6), "model": model}

    # --- FAZA 1: PLAN (osobne wywołanie, bez narzędzi → gwarantowany tekst planu) ---
    plan = ""
    try:
        planner = _chat(temperature=0.2)
        plan_msgs = [SystemMessage(content=PLAN_PROMPT)]
        plan_msgs.extend(history or [])
        plan_msgs.append(HumanMessage(content=user_message))
        plan_ai = planner.invoke(plan_msgs)
        _add_usage(usage, plan_ai)
        plan = (plan_ai.content or "").strip() if isinstance(plan_ai.content, str) else ""
    except Exception as e:  # noqa: BLE001
        return {"answer": "", "plan": "", "trace": trace, "turns": turns,
                "usage": _usage_out(), "error": f"Nie udało się zainicjalizować LLM: {e}"}

    # --- FAZA 2: WYKONANIE (tool calling; wymuszone, gdy w zapytaniu jest SMILES) ---
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history or [])
    messages.append(HumanMessage(content=user_message))
    if plan:
        messages.append(AIMessage(content=f"Plan analizy:\n{plan}"))

    force_first = _has_smiles(user_message)
    try:
        for i in range(max_iters):
            llm = get_llm(force_tools=(force_first and i == 0))
            ai: AIMessage = llm.invoke(messages)
            messages.append(ai)
            _add_usage(usage, ai)
            thought = (ai.content or "").strip() if isinstance(ai.content, str) else ""

            if not ai.tool_calls:
                return {"answer": ai.content, "plan": plan, "trace": trace,
                        "turns": turns, "usage": _usage_out(), "error": None}

            calls = []
            for tc in ai.tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                tool = TOOL_MAP.get(name)
                result = tool.invoke(args) if tool is not None else f"BŁĄD: nieznane narzędzie {name}"
                entry = {"tool": name, "args": args, "result": result}
                trace.append(entry)
                calls.append(entry)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            turns.append({"thought": thought, "calls": calls})

        return {"answer": "Przekroczono limit kroków agenta.", "plan": plan,
                "trace": trace, "turns": turns, "usage": _usage_out(), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"answer": "", "plan": plan, "trace": trace, "turns": turns,
                "usage": _usage_out(), "error": f"Błąd podczas działania agenta: {e}"}
