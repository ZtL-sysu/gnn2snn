"""Render the critical-site manuscript Markdown to a single LaTeX article.

Run from the paper directory when the source tables and figure assets are staged:
    python render_tex.py

This script intentionally performs no compilation or external conversion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "manuscript.md"
CAPTIONS = ROOT / "captions.json"
OUTPUT = ROOT / "main.tex"

TABLE_KEYS = ("historical", "realizability", "allocation", "mechanism_cost")
FIGURES = {
    "framework": ("../figures/framework.png", "png"),
    "ideal_sensitivity": ("../figures/ideal_sensitivity.pdf", "pdf"),
    "allocation_contrasts": ("../figures/allocation_contrasts.pdf", "pdf"),
    "mechanism_runtime": ("../figures/mechanism_runtime.pdf", "pdf"),
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,top=0.8in,bottom=0.8in,left=0.75in,right=0.75in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{caption}
\usepackage{adjustbox}
\usepackage[section]{placeins}
\usepackage{url}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setcounter{secnumdepth}{3}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.7}
\title{Critical-Site Sensitivity and Realizable Calibration in GCN-to-SNN Conversion}
\author{}
\date{}

"""


def text_to_tex(text: str) -> str:
    """Convert the small Markdown/TeX subset used in the manuscript prose."""
    # Turn the one un-delimited exponent expression in prose into inline math.
    text = re.sub(
        r"(?<![\w$])([A-Za-z][A-Za-z0-9]*)\^\{([^{}]+)\}([A-Za-z0-9]+)",
        r"$\1^{\2}\3$",
        text,
    )

    # Protect existing math and TeX references while transforming prose.
    chunks = re.split(r"(\$[^$]*\$|\\ref\{[^{}]+\})", text)
    for index, chunk in enumerate(chunks):
        if not chunk or chunk.startswith("$") or chunk.startswith(r"\ref{"):
            continue
        chunk = re.sub(
            r"\[((?:[A-Za-z][A-Za-z0-9_-]*)(?:\s*,\s*[A-Za-z][A-Za-z0-9_-]*)*)\]",
            lambda match: r"\citep{" + re.sub(r"\s*,\s*", ",", match.group(1)) + "}",
            chunk,
        )
        chunk = chunk.replace("“", "``").replace("”", "''")
        chunk = chunk.replace("’", "'").replace("‘", "`")
        chunk = chunk.replace("–", "--").replace("—", "---")
        chunk = chunk.replace("−", r"\textminus{}")
        # Escape ordinary prose percent signs, but preserve existing TeX \%.
        chunk = re.sub(r"(?<!\\)%", r"\\%", chunk)
        chunks[index] = chunk
    return "".join(chunks)


def read_assets() -> dict[str, str]:
    with CAPTIONS.open(encoding="utf-8") as stream:
        captions = json.load(stream)
    needed = set(TABLE_KEYS) | set(FIGURES)
    missing = sorted(needed.difference(captions))
    if missing:
        raise ValueError(f"captions.json is missing keys: {', '.join(missing)}")
    for key in TABLE_KEYS:
        if not (ROOT / "tables" / f"{key}.tex").is_file():
            raise FileNotFoundError(ROOT / "tables" / f"{key}.tex")
    return captions


def make_table(key: str, caption: str) -> str:
    return "\n".join(
        (
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{text_to_tex(caption)}}}",
            rf"\label{{tab:{key}}}",
            rf"\begin{{adjustbox}}{{max width=\linewidth}}\input{{tables/{key}.tex}}\end{{adjustbox}}",
            r"\end{table}",
        )
    )


def make_figure(key: str, caption: str) -> str:
    path, _extension = FIGURES[key]
    return "\n".join(
        (
            r"\begin{figure}[!htbp]",
            r"\centering",
            rf"\includegraphics[width=0.96\linewidth]{{{path}}}",
            rf"\caption{{{text_to_tex(caption)}}}",
            rf"\label{{fig:{key}}}",
            r"\end{figure}",
        )
    )


def render_body(markdown: str, captions: dict[str, str]) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    abstract = False

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(part.strip() for part in paragraph)
            if joined:
                output.append(text_to_tex(joined))
                output.append("")
            paragraph.clear()

    def flush_abstract() -> None:
        nonlocal abstract
        flush_paragraph()
        if abstract:
            output.append(r"\end{abstract}")
            output.append("")
            abstract = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush_paragraph()
            i += 1
            continue

        if line.startswith("# "):
            flush_abstract()
            # Title is emitted by \maketitle from the preamble.
            i += 1
            continue

        if line == "## Abstract":
            flush_paragraph()
            output.extend((r"\begin{abstract}", ""))
            abstract = True
            i += 1
            continue

        heading = re.match(r"^##\s+(\d+)\s+(.+)$", line)
        if heading:
            flush_abstract()
            output.extend((rf"\section{{{text_to_tex(heading.group(2))}}}", ""))
            i += 1
            continue

        subsection = re.match(r"^###\s+(\d+(?:\.\d+)*)\s+(.+)$", line)
        if subsection:
            flush_paragraph()
            number, title = subsection.groups()
            output.extend((rf"\subsection*{{{number}\quad {text_to_tex(title)}}}", ""))
            i += 1
            continue

        table = re.fullmatch(r"\[TABLE:([A-Za-z0-9_]+)\]", line)
        if table:
            flush_paragraph()
            key = table.group(1)
            if key not in TABLE_KEYS:
                raise ValueError(f"Unexpected table marker: {key}")
            output.extend((make_table(key, captions[key]), ""))
            i += 1
            continue

        figure = re.fullmatch(r"\[FIGURE:([A-Za-z0-9_]+)\]", line)
        if figure:
            flush_paragraph()
            key = figure.group(1)
            if key not in FIGURES:
                raise ValueError(f"Unexpected figure marker: {key}")
            output.extend((make_figure(key, captions[key]), ""))
            i += 1
            continue

        if line == r"\[":
            flush_paragraph()
            display_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                display_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Unclosed display math block")
            output.extend((r"\[", "\n".join(display_lines), r"\]", ""))
            i += 1
            continue

        section = re.match(r"^##\s+(.+)$", line)
        if section:
            # Abstract is the only unnumbered level-two heading permitted.
            raise ValueError(f"Unexpected level-two heading: {line}")

        if line.startswith("### "):
            raise ValueError(f"Malformed numbered subsection: {line}")

        paragraph.append(line)
        i += 1

    flush_abstract()
    flush_paragraph()

    # Enforce exactly six numbered main sections and one of each source marker.
    body = "\n".join(output)
    if len(re.findall(r"(?m)^\\section\{", body)) != 6:
        raise ValueError("Manuscript must contain exactly six numbered main sections")
    for key in TABLE_KEYS:
        if body.count(rf"\label{{tab:{key}}}") != 1:
            raise ValueError(f"Expected exactly one table marker for {key}")
    for key in FIGURES:
        if body.count(rf"\label{{fig:{key}}}") != 1:
            raise ValueError(f"Expected exactly one figure marker for {key}")
    return body


def main() -> None:
    captions = read_assets()
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    body = render_body(manuscript, captions)
    document = PREAMBLE + r"\begin{document}" + "\n" + r"\maketitle" + "\n\n"
    document += body
    document += "\n" + r"\bibliographystyle{unsrtnat}" + "\n"
    document += r"\bibliography{references}" + "\n" + r"\end{document}" + "\n"
    OUTPUT.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
