# Paper

**Selective Belief Propagation for Efficient Truth Maintenance in LLM Agents**

Yuhan Feng — S.-T. Yau High School Science Award 2026 (Computer Science),
Zhuzhou Nanfang High School.

- `main.pdf` — the final 38-page report as submitted (SHA-256
  `5ff79d0b0ae489418771e630b320e4ab52106de2f04552655abc6c2abe41c2f1`).
- `main.tex` — LaTeX source (standard `article` class, embedded
  bibliography, no custom style file).
- `figures/` — the ten figures referenced by the source.

## Build

From this directory:

```bash
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-references
```

Any standard TeX distribution (MiKTeX or TeX Live) works. The preamble uses
only common packages (`geometry`, `amsmath`, `booktabs`, `graphicx`,
`tikz`, `algorithm`, `natbib`, `hyperref`, `cleveref`, `enumitem`).

## Code and reproduction

The implementation and the offline reproduction harness live in the parent
repository. See the top-level `README.md` and `REPRODUCE.md`. The paper's
LLM-derived numbers are replayed from the frozen prompt cache with zero API
calls:

```bash
python scripts/reproduce_paper_tables.py
```
