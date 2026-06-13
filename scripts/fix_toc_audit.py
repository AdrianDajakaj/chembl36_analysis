"""Audit TOC links vs markdown headers in analysis.ipynb."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "analysis.ipynb"


def slug(title: str) -> str:
    """VS Code / JupyterLab header anchor (GitHub-style)."""
    s = title.strip().lower()
    s = re.sub(r"[`]", "", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def main():
    nb = json.loads(NB.read_text())
    toc = "".join(nb["cells"][1]["source"])
    links = re.findall(r"\[([^\]]+)\]\(#([^)]+)\)", toc)

    headers = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        t = "".join(c.get("source", ""))
        for m in re.finditer(r"^(#{1,4})\s+(.+)$", t, re.M):
            level = len(m.group(1))
            title = m.group(2).strip()
            headers.append((level, title, slug(title), i))

    anchors = {h[2]: h for h in headers}
    broken = []
    ok = 0
    for label, anchor in links:
        if anchor in anchors:
            ok += 1
        else:
            broken.append((label, anchor))

    print(f"TOC links: {len(links)}, OK: {ok}, broken: {len(broken)}")
    for label, anchor in broken:
        # guess by section number prefix
        num = re.match(r"([\d\.]+[a-z]?)", label)
        prefix = num.group(1) if num else ""
        cands = [h for h in headers if h[1].startswith(prefix) or prefix in h[1][:12]]
        print(f"\nBROKEN: [{label}]")
        print(f"  TOC anchor: #{anchor}")
        if cands:
            print(f"  correct:    #{cands[0][2]}  ({cands[0][1]})")


if __name__ == "__main__":
    main()
