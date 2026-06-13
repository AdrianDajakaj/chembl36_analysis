"""Naprawa nawigacji spisu treści: kotwice wewnątrz nagłówków (nie nad nimi)."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "analysis.ipynb"

STANDALONE_ANCHOR = re.compile(r'^<a id="(sec-[^"]+)"></a>\s*\n', re.M)
HEADER_LINE = re.compile(r'^(#{1,4})\s+(.+)$', re.M)


def section_id(num: str) -> str:
    return "sec-" + num.replace(".", "-").lower()


def parse_num(title: str) -> str | None:
    m = re.match(r"^(\d+(?:\.\d+)*[a-z]?)\.?\s", title)
    return m.group(1) if m else None


def strip_standalone_anchors(text: str) -> str:
    return STANDALONE_ANCHOR.sub("", text)


def inline_anchor_in_headers(text: str) -> str:
    def repl(m: re.Match) -> str:
        hashes, body = m.group(1), m.group(2)
        if body.startswith('<a id="sec-'):
            return m.group(0)
        # usuń ewentualną kotwicę z początku treści nagłówka
        body = re.sub(r'^<a id="sec-[^"]+"></a>\s*', "", body)
        num = parse_num(body)
        if not num:
            return m.group(0)
        anchor = section_id(num)
        return f'{hashes} <a id="{anchor}"></a>{body}'

    return HEADER_LINE.sub(repl, text)


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


def main() -> None:
    nb = json.loads(NB.read_text())

    changed = 0
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        new_src = strip_standalone_anchors(src)
        new_src = inline_anchor_in_headers(new_src)
        if new_src != src:
            c["source"] = [new_src]
            changed += 1

    headers = collect_headers(nb["cells"])
    nb["cells"][1]["source"] = [build_toc(headers)]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Inline anchors in {changed} cells; TOC entries: {len(headers)} -> {NB}")


if __name__ == "__main__":
    main()
