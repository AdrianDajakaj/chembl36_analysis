"""Naprawa hierarchii nagłówków (rozdz. 8–9) i jawnych kotwic HTML w analysis.ipynb."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "analysis.ipynb"

SEC9_TITLE = (
    "# 9. GNN: Graph Isomorphism Network → regresja pIC50 (BRD4)\n\n"
    "Model grafowy **GIN** (GINEConv) z bogatą featuryzacją RDKit — następca baseline GCN. "
    "Trenowany na tym samym zbiorze BRD4 co MLP z sekcji 8.\n"
)


def section_id(num: str) -> str:
    """ASCII anchor: 8.2b -> sec-8-2b, 9.16b -> sec-9-16b."""
    return "sec-" + num.replace(".", "-").lower()


def parse_num(title: str) -> str | None:
    m = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\.?\s", title)
    return m.group(1) if m else None


def promote_header(line: str) -> str | None:
    """Podnieś ### 8.x / ### 9.x do ## (zostaw # i ## bez zmian)."""
    m = re.match(r"^###\s+(\d+(?:\.\d+)*[a-z]?)\.?\s+(.*)$", line)
    if not m:
        return None
    num, rest = m.group(1), m.group(2)
    major = int(num.split(".")[0])
    if major not in (8, 9):
        return None
    # „8.2 Wznowienie" (bez trzeciej kropki) → 8.2a żeby nie kolidować z „8.2. Podziały"
    if num == "8.2" and not rest.startswith("Podziały"):
        num = "8.2a"
        rest = rest if rest.startswith("Wznowienie") else rest
    return f"## {num}. {rest}"


def fix_cell_source(source: str) -> str:
    """Kotwice wewnątrz nagłówka (nie nad nim) — poprawne przewijanie w Cursor/VS Code."""
    lines = source.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.rstrip("\n")
        promoted = promote_header(stripped)
        if promoted:
            stripped = promoted
        hm = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if hm:
            hashes, body = hm.group(1), hm.group(2).strip()
            body = re.sub(r'^<a id="sec-[^"]+"></a>\s*', "", body)
            num = parse_num(body)
            if num and not body.startswith('<a id="sec-'):
                anchor = section_id(num)
                body = f'<a id="{anchor}"></a>{body}'
            out.append(f"{hashes} {body}" + ("\n" if line.endswith("\n") else ""))
        else:
            # usuń samotne kotwice nad nagłówkiem (legacy)
            if re.match(r'^<a id="sec-[^"]+"></a>\s*$', stripped):
                continue
            out.append(line)
    return "".join(out)


def insert_section9_header(nb) -> None:
    """Wstaw h1 rozdziału 9 przed pierwszą komórką ### 9.1."""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        if re.search(r"^###\s+9\.1\.", src, re.M):
            nb["cells"].insert(
                i,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        '# <a id="sec-9"></a>9. GNN: Graph Isomorphism Network → regresja pIC50 (BRD4)\n\n'
                        "Model grafowy **GIN** (GINEConv) z bogatą featuryzacją RDKit — następca baseline GCN. "
                        "Trenowany na tym samym zbiorze BRD4 co MLP z sekcji 8.\n",
                    ],
                },
            )
            return


def build_toc(cells) -> str:
    lines = ["# Spis treści", ""]
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        text = "".join(c.get("source", ""))
        for m in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.M):
            level = len(m.group(1))
            title = m.group(2).strip()
            if title == "Spis treści":
                continue
            num, rest = "", title
            pm = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\.?\s+(.*)$", title)
            if pm:
                num, rest = pm.group(1), pm.group(2).strip()
            if level == 3 and not num:
                continue
            indent = "   " * (level - 1)
            anchor = section_id(num) if num else None
            if not anchor:
                continue
            if level == 1:
                lines.append(f"{num}. [{rest}](#{anchor})")
            else:
                lines.append(f"{indent}- {num} [{rest}](#{anchor})")
    lines.append("")
    return "\n".join(lines)


def main():
    nb = json.loads(NB.read_text())

    insert_section9_header(nb)

    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        new_src = fix_cell_source(src)
        if new_src != src:
            c["source"] = [new_src]

    # kotwica na początku spisu / sekcji 1-7 też
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        if not re.search(r"^#{1,3}\s+\d", src, re.M):
            continue
        c["source"] = [fix_cell_source(src)]

    nb["cells"][1]["source"] = [build_toc(nb["cells"])]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Fixed headers + HTML anchors -> {NB}")


if __name__ == "__main__":
    main()
