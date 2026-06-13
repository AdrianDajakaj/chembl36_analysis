"""Regeneruje spis treści w analysis.ipynb (kotwice sec-* muszą być w nagłówkach, nie nad nimi)."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "analysis.ipynb"


def section_id(num: str) -> str:
    return "sec-" + num.replace(".", "-").lower()


def parse_section(title: str) -> tuple[str, str]:
    m = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\.?\s+(.*)$", title)
    if m:
        return m.group(1), m.group(2).strip()
    return "", title


def collect_headers(cells) -> list[tuple[int, str, str, str]]:
    out = []
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        text = "".join(c.get("source", ""))
        for m in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.M):
            level = len(m.group(1))
            raw = m.group(2).strip()
            title = re.sub(r'^<a id="sec-[^"]+"></a>\s*', "", raw)
            if title == "Spis treści":
                continue
            num, rest = parse_section(title)
            if level == 3 and not num:
                continue
            if not num:
                continue
            out.append((level, num, rest, section_id(num)))
    return out


def build_toc(headers) -> str:
    lines = ["# Spis treści", ""]
    for level, num, rest, anchor in headers:
        indent = "   " * (level - 1)
        if level == 1:
            lines.append(f"{num}. [{rest}](#{anchor})")
        else:
            lines.append(f"{indent}- {num} [{rest}](#{anchor})")
    lines.append("")
    return "\n".join(lines)


def main():
    nb = json.loads(NB.read_text())
    headers = collect_headers(nb["cells"])
    nb["cells"][1]["source"] = [build_toc(headers)]
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Updated TOC: {len(headers)} entries -> {NB}")


if __name__ == "__main__":
    main()
