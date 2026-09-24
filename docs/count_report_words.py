"""Conservative chapter word counts for the CM3070 LaTeX report.

Counts prose, table cells and code listings. Excludes chapter titles,
figure/table captions, references and mathematical expressions. Subsection titles count.
This is a reproducible estimate, not the university's official counter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("DRAFT_FINAL_PROJECT_REPORT_V2.tex")
LIMITS = {
    "Introduction": 1000,
    "Literature Review": 2500,
    "Design": 2000,
    "Implementation": 2500,
    "Evaluation": 2500,
    "Conclusion": 1000,
}


def remove_braced_command(source: str, name: str) -> str:
    marker = f"\\{name}{{"
    while marker in source:
        start = source.index(marker)
        pos = start + len(marker)
        depth = 1
        while pos < len(source) and depth:
            if source[pos] == "{" and source[pos - 1] != "\\":
                depth += 1
            elif source[pos] == "}" and source[pos - 1] != "\\":
                depth -= 1
            pos += 1
        source = source[:start] + source[pos:]
    return source


def clean(section: str) -> str:
    section = re.sub(r"(?<!\\)%.*", "", section)
    for name in ("caption", "label", "cite", "ref", "pageref", "url"):
        section = remove_braced_command(section, name)
    section = re.sub(r"\\begin\{(?:equation\*?|align\*?)\}.*?\\end\{(?:equation\*?|align\*?)\}", " ", section, flags=re.S)
    section = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", section, flags=re.S)
    section = re.sub(r"\\includegraphics(?:\[[^]]*\])?\{[^}]*\}", " ", section)
    section = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{[^}]*\}", " ", section)
    section = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", section)
    section = re.sub(r"\\(?:section|subsection|subsubsection)\*?(?:\[[^]]*\])?\{([^}]*)\}", r" \1 ", section)
    section = re.sub(r"\$[^$]*\$", " ", section, flags=re.S)
    section = re.sub(r"\\\[.*?\\\]", " ", section, flags=re.S)
    section = re.sub(r"\\(?:setlength|renewcommand)\{[^}]*\}\{[^}]*\}", " ", section)
    section = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", section)
    section = re.sub(r"\\[^A-Za-z]", " ", section)
    return section.replace("~", " ").replace("&", " ")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    markers = list(re.finditer(r"^\\section\{([^}]*)\}", source, flags=re.M))
    total = 0
    errors = []
    for i, marker in enumerate(markers):
        title = marker.group(1)
        name = re.sub(r"\s*\(\d+/\d+ words\)$", "", title)
        if name not in LIMITS:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else source.index("\\begin{thebibliography}", marker.end())
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", clean(source[marker.end():end]))
        count = len(words)
        total += count
        print(f"{name}: {count}/{LIMITS[name]}")
        if count > LIMITS[name]:
            errors.append(f"{name} exceeds its limit")
        declared = re.search(r"\((\d+)/(\d+) words\)$", title)
        if not declared or (int(declared.group(1)), int(declared.group(2))) != (count, LIMITS[name]):
            errors.append(f"{name} title count does not match the current source")
        if "--detail" in sys.argv:
            parts = list(re.finditer(r"^\\subsection\{([^}]*)\}", source[marker.end():end], flags=re.M))
            body = source[marker.end():end]
            for j, part in enumerate(parts):
                stop = parts[j + 1].start() if j + 1 < len(parts) else len(body)
                n = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", clean(body[part.start():stop])))
                print(f"  {part.group(1)}: {n}")
            for j, table in enumerate(re.finditer(r"\\begin\{table\}.*?\\end\{table\}", body, flags=re.S), 1):
                n = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", clean(table.group())))
                print(f"    Table {j}: {n}")
    print(f"Total six chapters: {total}/10500")
    if total > 10500:
        errors.append("total exceeds 10500 words")
    if errors and "--check" in sys.argv:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
